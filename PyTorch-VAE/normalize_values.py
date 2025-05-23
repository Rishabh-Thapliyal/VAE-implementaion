from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def compute_mean_std(image_dir, transform=None):
    """
    Compute the mean and standard deviation of a dataset.

    Args:
        image_dir (str): Path to the directory containing images.
        transform (callable, optional): Transformations to apply to the images.

    Returns:
        tuple: Mean and standard deviation for each channel.
    """
    pixel_sum = 0
    pixel_squared_sum = 0
    num_pixels = 0

    # Iterate through all images
    for root, _, files in os.walk(image_dir):
        for file in tqdm(files, desc="Processing images"):
            if file.endswith(('.jpeg', '.jpg', '.png')):
                img = Image.open(os.path.join(root, file)).convert('RGB')
                if transform:
                    img = transform(img)
                else:
                    img = transforms.ToTensor()(img)  # Default transform to tensor

                # Update sums
                pixel_sum += img.sum(dim=(1, 2))
                pixel_squared_sum += (img ** 2).sum(dim=(1, 2))
                num_pixels += img.shape[1] * img.shape[2]

    # Compute mean and std
    mean = pixel_sum / num_pixels
    std = (pixel_squared_sum / num_pixels - mean ** 2).sqrt()

    return mean.numpy(), std.numpy()

# Example usage
image_dir = "/mntdata/rishabh/ece285/PyTorch-VAE/data/chest_xray/chest_xray/"
mean, std = compute_mean_std(image_dir)
print(f"Mean: {mean}, Std: {std}")