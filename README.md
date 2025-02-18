# Diffusion Model for Generation of Pixel Art-like images

This project focuses on gathering a dataset and training a text-guided diffusion model for generating pixel art-like images. The goal is to develop a dedicated diffusion model for pixel art-like images without the computational cost of running a large Stable Diffusion (SD) model. Since pixel art has a very limited color palette and minimal detail, a small diffusion model, when paired with the appropriate text embedding model, should perform well.

Eventually, this model will be used in another project involving food items. 

# Dataset

For the dataset, we need paired text and pixel art images. Since collecting this type of data is challenging, we use a pseudo-image approach that mimics pixel art. A common method is to apply nearest-neighbor interpolation to downsample images to a very low resolution, such as 64×64. While this theoretically reduces details, the images often still appear too realistic.

A more refined approach considers that pixel art typically features uniform colors across large regions, lacking the high-frequency details found in real-life images. To replicate this characteristic, I first downsample the image using nearest-neighbor interpolation and then apply a mode-searching segmentation algorithm (QuickShift). Using the resulting segmentation mask, I compute the average color for each region and apply that uniform color throughout the region. This process generates a pixel art-like pseudo-target that can be used for training the model. A sample of this is shown below:

![fixed](https://github.com/user-attachments/assets/38fc7c0d-d043-44d7-bce7-71501b4cf373)
