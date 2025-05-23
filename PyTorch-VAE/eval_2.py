import torch
from torch import nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
from ignite.metrics import FID, InceptionScore
from ignite.engine import Engine
import os
from tqdm import tqdm
from models.vanilla_vae import VanillaVAE
from torch.utils.data import DataLoader, Dataset
from typing import List
from torchvision.models import Inception_V3_Weights

from visualise_generated import visualize_generated_images

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load VAE model
model = VanillaVAE(in_channels=3, latent_dim=128).to(device)
checkpoint = torch.load("/mntdata/rishabh/ece285/PyTorch-VAE/logs/VanillaVAE/version_29/checkpoints/last.ckpt", map_location=device)
state_dict = checkpoint['state_dict']
if any(k.startswith('model.') for k in state_dict):
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# Image transformations
transform_to_tensor = transforms.Compose([
    transforms.Resize((299, 299)),  # Required for InceptionV3
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.487, 0.487, 0.487], std=[0.245, 0.245, 0.245]) 
])

# Wrapper for InceptionV3 to extract features
from torchvision.models import Inception_V3_Weights

# Wrapper for InceptionV3 to extract features
class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        # Load InceptionV3 with the most up-to-date weights
        weights = Inception_V3_Weights.DEFAULT
        inception = models.inception_v3(weights=weights)
        
        # Replace the final classification layer with Identity to extract features
        inception.fc = nn.Identity()
        
        # Assign the modified model
        self.model = inception

    def forward(self, x):
        # Ensure input is 4D and pass through the feature extractor
        if len(x.shape) != 4:
            raise ValueError("Input to InceptionV3FeatureExtractor must be a 4D tensor [batch_size, channels, height, width].")
        outputs = self.model(x)
        if isinstance(outputs, tuple):  # Handle InceptionOutputs
            outputs = outputs[0]  # Select the logits
        return outputs
    

# Load all real images
def load_all_real_images(base_dir: str, transform, batch_size=32, num_workers=4):
    """
    Load all real images from the directory structure:
    base_dir/train/NORMAL, base_dir/train/PNEUMONIA, etc.

    Args:
        base_dir (str): Base directory containing train, test, val subdirectories.
        transform (callable): Transformations to apply to the images.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        torch.Tensor: Tensor containing all loaded images.
    """
    real_images = []
    sub_dirs = ['test','val']
    classes = ['NORMAL', 'PNEUMONIA']

    for sub_dir in sub_dirs:
        for cls in classes:
            image_dir = os.path.join(base_dir, sub_dir, cls)
            if os.path.exists(image_dir):
                dataset = RealImageDataset(image_dir, transform)
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True  # Speeds up data transfer to GPU
                )
                for batch in tqdm(dataloader, desc=f"Loading images from {image_dir}"):
                    real_images.append(batch.to(device))  # Move batch to GPU

    return torch.cat(real_images, dim=0)

# Generate images
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
def transform_to_tensor_gpu(tensor):
    # Resize the tensor to (299, 299)
    tensor = F.interpolate(tensor.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False).squeeze(0)
    # Normalize the tensor
    normalize = Normalize(mean=[0.487, 0.487, 0.487], std=[0.245, 0.245, 0.245])
    tensor = normalize(tensor)
    return tensor

def generate_images(num_images, batch_size=32):
    """Generate images using the VAE model, fully processed on the GPU."""
    generated_images = []
    for _ in tqdm(range(0, num_images, batch_size), desc="Generating images"):
        with torch.no_grad():
            current_batch = min(batch_size, num_images - len(generated_images))
            # Generate samples on the GPU
            samples = model.sample(current_batch, device)  # Samples are already on the GPU
            
            # Normalize samples to [0, 1] range directly on the GPU
            samples = (samples - samples.amin(dim=(1, 2, 3), keepdim=True)) / (
                samples.amax(dim=(1, 2, 3), keepdim=True) - samples.amin(dim=(1, 2, 3), keepdim=True) + 1e-8
            )
            
            # Apply transformations and append to the list
            for img in samples:
                img = transform_to_tensor_gpu(img)  # Apply GPU transformations
                generated_images.append(img)
    
    # Stack all generated images into a single tensor
    return torch.stack(generated_images, dim=0).to(device)

# RealImageDataset class
class RealImageDataset(Dataset):
    def __init__(self, image_dir: str, transform):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)

# FID and IS evaluation
def evaluate_fid_is(real_images, generated_images):
    """Evaluate FID and Inception Score."""
    if len(real_images.shape) != 4 or len(generated_images.shape) != 4:
        raise ValueError("Input tensors must have shape [batch_size, channels, height, width].")

    # FID Metric
    feature_extractor = InceptionV3FeatureExtractor().to(device)
    fid_metric = FID(num_features=2048, feature_extractor=feature_extractor, device=device)
    fid_evaluator = Engine(lambda engine, batch: batch)
    fid_metric.attach(fid_evaluator, "fid")

    # IS Metric
    is_metric = InceptionScore(device=device)
    is_evaluator = Engine(lambda engine, batch: batch)
    is_metric.attach(is_evaluator, "is")

    # Run FID evaluation
    print("Computing FID...")
    fid_state = fid_evaluator.run([[real_images, generated_images]]).metrics
    fid_score = fid_state["fid"]

    # Run IS evaluation
    print("Computing Inception Score...")
    is_state = is_evaluator.run([generated_images]).metrics
    is_score = is_state["is"]

    return fid_score, is_score

def evaluate_fid_is_in_batches(real_images, generated_images, batch_size=32):
    """Evaluate FID and Inception Score in batches."""
    # Initialize FID Metric
    feature_extractor = InceptionV3FeatureExtractor().to(device)
    fid_metric = FID(num_features=2048, feature_extractor=feature_extractor, device=device)

    # Initialize IS Metric
    is_metric = InceptionScore(device=device)

    # Process real and generated images in batches for FID
    print("Computing FID in batches...")
    for i in tqdm(range(0, len(real_images), batch_size), desc="FID Batches"):
        real_batch = real_images[i:i + batch_size].to(device)
        gen_batch = generated_images[i:i + batch_size].to(device)
        fid_metric.update((gen_batch, real_batch))  # Incrementally update FID metric

    fid_score = fid_metric.compute()  # Compute final FID score

    # Process generated images in batches for IS
    print("Computing Inception Score in batches...")
    for i in tqdm(range(0, len(generated_images), batch_size), desc="IS Batches"):
        gen_batch = generated_images[i:i + batch_size].to(device)
        is_metric.update(gen_batch)  # Incrementally update IS metric

    is_score = is_metric.compute()  # Compute final IS score

    return fid_score, is_score

# Main evaluation
if __name__ == "__main__":
    # Load all real images
    print("\nLoading all real images...")
    real_images = load_all_real_images(
        base_dir="/mntdata/rishabh/ece285/PyTorch-VAE/data/chest_xray/",
        transform=transform_to_tensor
    )

    # Count the number of real images
    num_images = real_images.size(0)
    print(f"Number of real images: {num_images}")

    # Generate images
    print("\nGenerating evaluation images...")
    generated_images = generate_images(num_images=num_images)

    # Ensure tensors have the correct shape
    assert len(real_images.shape) == 4, "real_images must be a 4D tensor"
    assert len(generated_images.shape) == 4, "generated_images must be a 4D tensor"

    # Evaluate FID and IS
    fid_score, is_score = evaluate_fid_is_in_batches(real_images, generated_images)

    print(f"\nFID Score: {fid_score:.2f}")
    print(f"Inception Score: {is_score:.2f}")

    # Visualize generated images
    print("\nVisualizing generated images...")
    visualize_generated_images(generated_images, grid_size=10, save_path="./generated_samples/generated_images_grid_128_test_val.png")