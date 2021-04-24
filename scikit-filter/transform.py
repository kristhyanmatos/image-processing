# Documentação
# Link: https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html#sphx-glr-auto-examples-transform-plot-rescale-py

import matplotlib.pyplot as plt

from skimage.transform import rescale, resize, downscale_local_mean

from skimage.color import rgb2gray

from PIL import Image
from numpy import asarray

# Open the image form working directory
file_image = Image.open("images/ppkeise.jpeg")

# convert image to numpy array
image_multicolor = asarray(file_image)
# convert image to shades of gray
image = rgb2gray(image_multicolor)

#  Dimensionar de forma proporcional os eixos
image_rescaled = rescale(image, 0.25, anti_aliasing=False)

#  Dimensionar individualmente cada eixo
image_resized = resize(
    image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True
)
# Número de amostras
image_downscaled = downscale_local_mean(image, (5, 4))

fig, axes = plt.subplots(nrows=2, ncols=2)

ax = axes.ravel()

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original image")

ax[1].imshow(image_rescaled, cmap="gray")
ax[1].set_title("Rescaled image (aliasing)")

ax[2].imshow(image_resized, cmap="gray")
ax[2].set_title("Resized image (no aliasing)")

ax[3].imshow(image_downscaled, cmap="gray")
ax[3].set_title("Downscaled image (no aliasing)")

ax[0].set_xlim(0, 512)
ax[0].set_ylim(512, 0)
plt.tight_layout()
plt.show()