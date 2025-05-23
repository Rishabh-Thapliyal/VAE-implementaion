import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from ignite.metrics import FID, InceptionScore
from ignite.engine import Engine
import os
from tqdm import tqdm
from models.vanilla_vae import VanillaVAE
from torch import Tensor
from typing import Tuple, List, Union

# 1. Initialize your VAE model (from previous code)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = VanillaVAE(in_channels= 3, latent_dim= 128).to(device)

# Load checkpoint (with prefix handling)
checkpoint = torch.load("/mntdata/rishabh/ece285/PyTorch-VAE/logs/VanillaVAE/version_9/checkpoints/last.ckpt", map_location=device)
# /mntdata/rishabh/ece285/PyTorch-VAE/logs/VanillaVAE/version_9/checkpoints/epoch=9-step=210.ckpt
state_dict = checkpoint['state_dict']
if any(k.startswith('model.') for k in state_dict):
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# 2. Image generation functions
def tensor_to_pil(img_tensor):
    """Convert model output tensor to PIL Image"""
    img = img_tensor.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 127.5 + 127.5).astype(np.uint8)
    return Image.fromarray(img)

def generate_images(num_images=5000, batch_size=32):
    """Generate PIL Images for evaluation"""
    images = []
    for _ in tqdm(range(0, num_images, batch_size), desc="Generating images"):
        with torch.no_grad():
            current_batch = min(batch_size, num_images - len(images))
            samples = model.sample(current_batch, device)
            for img in samples:
                # images.append(tensor_to_pil(img))
                ndarr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                images.append(im)
    return images, samples

# 3. Load real images (for FID calculation)
def load_real_images(image_dir, num_images=5000):
    """Load real images from directory"""
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpeg', '.png'))]
    images = []
    for path in tqdm(image_paths[:num_images], desc="Loading real images"):
        img = Image.open(path).convert('RGB')
        # images.append(transform(img))
        images.append(img)
    return images


# 1. FID requires pairs of (real_images, generated_images)
def fid_prepare_batch(batch: Tuple[List[Tensor], List[Tensor]], device):
    real_batch, fake_batch = batch
    # Stack lists of tensors into batch tensors
    real_batch = torch.stack(real_batch).to(device)
    fake_batch = torch.stack(fake_batch).to(device)
    return (real_batch, fake_batch)

def fid_evaluation_step(engine: Engine, batch: Tuple[Tensor, Tensor]):
    real, fake = batch
    return (real, fake)  # Return both real and fake for FID comparison

# 2. IS only needs generated images
def is_prepare_batch(batch: List[Tensor], device):
    return torch.stack(batch).to(device)

def is_evaluation_step(engine: Engine, batch: Tensor):
    return batch  # Just return generated images for IS

# 3. Create separate engines
fid_evaluator = Engine(fid_evaluation_step)
is_evaluator = Engine(is_evaluation_step)

# 4. Setup metrics
# num_features=128, feature_extractor=None,
fid_metric = FID(num_features=2048, device=device)
is_metric = InceptionScore(device=device)

# Attach metrics
fid_metric.attach(fid_evaluator, "fid")
is_metric.attach(is_evaluator, "is")

# ----------
print("\nGenerating evaluation images...")
num_images = 10

generated_images, generated_samples = generate_images(num_images=2*num_images)

real_images_normal = load_real_images("/mntdata/rishabh/ece285/PyTorch-VAE/data/chest_xray/test/NORMAL/", 
                                num_images=num_images)
real_images_pneumonia = load_real_images("/mntdata/rishabh/ece285/PyTorch-VAE/data/chest_xray/test/PNEUMONIA/", 
                                num_images=num_images)

real_images = real_images_normal + real_images_pneumonia

# Convert PIL images to tensors (required by Ignite)
transform_to_tensor = transforms.Compose([
        #    transforms.CenterCrop(1000),
    transforms.Resize(64),
    transforms.ToTensor(),
])

generated_tensors = [transform_to_tensor(img) for img in generated_images]
real_tensors = [transform_to_tensor(img) for img in real_images]
# generated_tensors = generated_images

# 5. Run evaluation properly
print("Computing FID...")
fid_state = fid_evaluator.run([fid_prepare_batch([real_tensors, generated_tensors], device)]).metrics
print(f"FID: {fid_state['fid']:.2f}")

print("Computing IS...")
is_state = is_evaluator.run([is_prepare_batch(generated_tensors, device)]).metrics

print(f"Inception Score: {is_state['is']:.2f}")

# 7. Save some samples for visual inspection
output_dir = "./generated_samples"
os.makedirs(output_dir, exist_ok=True)
for i, img in enumerate(generated_images[:num_images]):
    img.save(os.path.join(output_dir, f"sample_{i:04d}.png"))
print(f"\nSaved sample images to {output_dir}")

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

fid = FrechetInceptionDistance(normalize=False).to(device)  # Set normalize=False since we'll handle normalization
inception = InceptionScore().to(device)

fid.update(real_tensors[0], real=True)
fid.update(generate_images, real=False)

fid_score = fid.compute().item()


# import torchvision.utils as vutils
# from torchvision.transforms import ToPILImage
# import os

# # Assuming generated_images is a list of PIL Images
# # First convert PIL Images to tensors
# # tensor_images = [transforms.ToTensor()(img) for img in generated_images[:num_images]]

# # Create grid (4 images per row by default)

# grid = vutils.make_grid(generated_tensors, nrow=8, padding=1)
# vutils.save_image(grid, os.path.join(output_dir, "sample_grid.png"))
# print(f"\nSaved grid image to {grid_path}")