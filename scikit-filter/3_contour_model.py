# Filter link: https://scikit-image.org/docs/stable/auto_examples/edges/plot_active_contours.html#sphx-glr-auto-examples-edges-plot-active-contours-py
from numpy import asarray
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
import numpy

# Open the image form working directory
image = Image.open("images/ppkeise.jpeg")

# convert image to numpy array
numpay_image = asarray(image)

# convert image to shades of gray
image = rgb2gray(numpay_image)

frequency = numpy.linspace(0, 2 * numpy.pi, 400)
r = 180 + 180 * numpy.sin(frequency)
c = 200 + 180 * numpy.cos(frequency)
initial = numpy.array([r, c]).T

snake = active_contour(gaussian(image, 4), initial, alpha=0.01, beta=1, gamma=0.01)

_, figure = plt.subplots(figsize=(7, 7))
figure.imshow(image, cmap=plt.cm.gray)
figure.plot(initial[:, 1], initial[:, 0], "--g", lw=3)
figure.plot(snake[:, 1], snake[:, 0], "-r", lw=3)
figure.set_xticks([]), figure.set_yticks([])
figure.axis([0, image.shape[1], image.shape[0], 0])

plt.show()