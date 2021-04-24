import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from PIL import Image
from numpy import asarray

# Open the image form working directory
image = Image.open("images/ppkeise.jpeg")

# convert image to numpy array
numpay_image = asarray(image)

# convert image to shades of gray
img = rgb2gray(numpay_image)


s = np.linspace(0, 2 * np.pi, 400)
r = 180 + 180 * np.sin(s)
c = 200 + 180 * np.cos(s)
init = np.array([r, c]).T

snake = active_contour(gaussian(img, 4), init, alpha=0.01, beta=1, gamma=0.01)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], "--r", lw=3)
ax.plot(snake[:, 1], snake[:, 0], "-b", lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()