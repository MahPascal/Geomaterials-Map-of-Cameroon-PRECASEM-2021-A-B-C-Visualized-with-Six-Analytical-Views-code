# Geomaterials-Map-of-Cameroon-PRECASEM-2021-A-B-C-Visualized-with-Six-Analytical-Views-code
Geomaterials Map of Cameroon (PRECASEM, 2021) A, B, C, Visualized with Six Analytical Views code


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

def resize_images_to_smallest(images):
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    new_height = min(heights)
    new_width = min(widths)
    resized_images = [cv2.resize(img, (new_width, new_height)) for img in images]
    return resized_images

def enhanced_visualization(fig_num, gray, img, title):
    fig = plt.figure(fig_num, figsize=(21, 15))

    # Meshgrid for 3D
    x, y = np.meshgrid(np.arange(gray.shape[1]), np.arange(gray.shape[0]))
    z = gray.flatten()

    # Subplot 1: 3D Scatter (projection='3d')
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(x.flatten(), y.flatten(), z, c=z, cmap='viridis', marker='o')
    ax1.set_title(f"3D Scatter: {title}")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Pixel Intensity')

    # Subplot 2: Original Image (BGR to RGB)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Original (RGB): {title}")
    ax2.axis('off')

    # Subplot 3: Grayscale image
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(gray, cmap='gray')
    ax3.set_title(f"Grayscale: {title}")
    ax3.axis('off')

    # Subplot 4: Edge Detection (Canny)
    edges = cv2.Canny(gray, 100, 200)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(edges, cmap='gray')
    ax4.set_title("Edge Detection (Canny)")
    ax4.axis('off')

    # Subplot 5: Histogram of pixel intensity
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(gray.ravel(), bins=256, range=(0, 256), color='gray')
    ax5.set_title("Pixel Intensity Histogram")
    ax5.set_xlabel("Pixel Value")
    ax5.set_ylabel("Frequency")

    # Subplot 6: Heatmap
    ax6 = fig.add_subplot(2, 3, 6)
    heatmap = ax6.imshow(gray, cmap='hot')
    fig.colorbar(heatmap, ax=ax6)
    ax6.set_title("Heatmap (Hot colormap)")
    ax6.axis('off')

    plt.suptitle(title, fontsize=12)
    plt.show()

def visualize_geomaps_with_6_views(image_paths, image_titles):
    images = [cv2.imread(path) for path in image_paths]
    resized_images = resize_images_to_smallest(images)
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in resized_images]

    for i in range(len(resized_images)):
        enhanced_visualization(i+1, grays[i], resized_images[i], image_titles[i])

# === Example usage with up to 9 maps ===
image_paths = [
    "/content/Geomaterials map of Cameroon (PRECASEM, 2021) A.jpg",
    "/content/Geomaterials map of Cameroon (PRECASEM, 2021) B.jpg",
    "/content/Geomaterials map of Cameroon (PRECASEM, 2021) C.jpg",
]

image_titles = [
    "Geomaterials map of Cameroon (PRECASEM, 2021) A",
    "Geomaterials map of Cameroon (PRECASEM, 2021) B",
    "Geomaterials map of Cameroon (PRECASEM, 2021) C",
]

visualize_geomaps_with_6_views(image_paths, image_titles)

