# Filter link: https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html#sphx-glr-auto-examples-transform-plot-rescale-py
from PIL import Image
from numpy import asarray
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt

# Open the image form working directory
image_file = Image.open("images/ppkeise.jpeg")

# convert image to numpy array
image_multicolor = asarray(image_file)

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

_, figures = plt.subplots(nrows=2, ncols=2)

figure = figures.ravel()

figure[0].imshow(image, cmap="gray")
figure[0].set_title("Original image")

figure[1].imshow(image_rescaled, cmap="gray")
figure[1].set_title("Rescaled image (aliasing)")

figure[2].imshow(image_resized, cmap="gray")
figure[2].set_title("Resized image (no aliasing)")

figure[3].imshow(image_downscaled, cmap="gray")
figure[3].set_title("Downscaled image (no aliasing)")

figure[0].set_xlim(0, 512)
figure[0].set_ylim(512, 0)
plt.tight_layout()
plt.show()