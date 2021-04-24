import matplotlib.pyplot as plt

from skimage import filters
from skimage.color import rgb2gray

from PIL import Image
from numpy import asarray

# Open the image form working directory
file_image = Image.open("images/ppkeise.jpeg")

# convert image to numpy array
image_multicolor = asarray(file_image)
# convert image to shades of gray
image = rgb2gray(image_multicolor)

edge_roberts = filters.scharr(image)
edge_sobel = filters.sobel(image)

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title("Scharr Edge Detection")

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title("Sobel Edge Detection")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()