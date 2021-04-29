# Filter link: https://scikit-image.org/docs/stable/auto_examples/edges/plot_edge_filter.html#sphx-glr-auto-examples-edges-plot-edge-filter-py
from PIL import Image
from numpy import asarray
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Open the image form working directory
file_image = Image.open("images/ppkeise.jpeg")

# convert image to numpy array
image_multicolor = asarray(file_image)

# convert image to shades of gray
image = rgb2gray(image_multicolor)

edge_roberts = filters.scharr(image)
edge_sobel = filters.sobel(image)

_, figures = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))

figures[0].imshow(edge_roberts, cmap=plt.cm.gray)
figures[0].set_title("Scharr Edge Detection")

figures[1].imshow(edge_sobel, cmap=plt.cm.gray)
figures[1].set_title("Sobel Edge Detection")

for figure in figures:
    figure.axis("off")

plt.tight_layout()
plt.show()