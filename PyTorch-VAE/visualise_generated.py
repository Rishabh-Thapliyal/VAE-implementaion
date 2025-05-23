import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch

def visualize_generated_images(images, grid_size=10, save_path=None):
    """
    Visualize generated images in a grid format.

    Args:
        images (torch.Tensor): Tensor of generated images with shape [N, C, H, W].
        grid_size (int): Number of rows and columns in the grid (default: 10x10).
        save_path (str, optional): Path to save the plotted grid as an image. If None, the grid is displayed.
    """
    import torch
    # Select a random subset of images to fit the grid
    num_images = grid_size * grid_size
    if images.size(0) < num_images:
        raise ValueError(f"Not enough images to fill a {grid_size}x{grid_size} grid. Got {images.size(0)} images.")
    
    # Denormalize the images (assuming they were normalized for InceptionV3)
    images = images[:num_images].clone()  # Select the first `num_images`
    images = images * torch.tensor([0.245, 0.245, 0.245], device=images.device).view(1, -1, 1, 1)  # Multiply by std
    images = images + torch.tensor([0.487, 0.487, 0.487], device=images.device).view(1, -1, 1, 1)  # Add mean
    images = images.clamp(0, 1)  # Clamp to [0, 1] range

    # Create a grid of images
    grid = vutils.make_grid(images, nrow=grid_size, padding=2, normalize=False)

    # Plot the grid
    plt.figure(figsize=(grid_size, grid_size))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())  # Convert to HWC format for plotting

    # Save or display the grid
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Grid saved to {save_path}")
    else:
        plt.show()