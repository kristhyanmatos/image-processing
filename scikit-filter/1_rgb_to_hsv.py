# Filter link: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_rgb_to_hsv.html#sphx-glr-auto-examples-color-exposure-plot-rgb-to-hsv-py
from skimage.color import rgb2hsv
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt

# Open the image form working directory
image_file = Image.open("images/ppkeise.jpeg")

# convert image to numpy array
image_rgb = asarray(image_file)

image_hsv = rgb2hsv(image_rgb)
image_hsv_hue = image_hsv[:, :, 0]
image_hsv_value = image_hsv[:, :, 2]

figure, (figure_sub_1, figure_sub_2, figure_sub_3) = plt.subplots(
    ncols=3, figsize=(8, 2)
)

figure_sub_1.imshow(image_rgb)
figure_sub_1.set_title("RGB image")
figure_sub_1.axis("off")
figure_sub_2.imshow(image_hsv_hue, cmap="hsv")
figure_sub_2.set_title("Hue channel")
figure_sub_2.axis("off")
figure_sub_3.imshow(image_hsv_value)
figure_sub_3.set_title("Value channel")
figure_sub_3.axis("off")

figure.tight_layout()

# Cut
hue_threshold = 0.1

binary_img = image_hsv_hue > hue_threshold

figure, (figure_sub_1, figure_sub_2) = plt.subplots(ncols=2, figsize=(8, 3))

figure_sub_1.hist(image_hsv_hue.ravel(), 512)
figure_sub_1.set_title("Histogram of the Hue channel with threshold")
figure_sub_1.axvline(x=hue_threshold, color="r", linestyle="dashed", linewidth=2)
figure_sub_1.set_xbound(0, 0.12)
figure_sub_2.imshow(binary_img)
figure_sub_2.set_title("Hue-thresholded image")
figure_sub_2.axis("off")

figure.tight_layout()

figure, figure_sub_1 = plt.subplots(figsize=(4, 3))

value_threshold = 0.1
binary_img = (image_hsv_hue > hue_threshold) | (image_hsv_value < value_threshold)

figure_sub_1.imshow(binary_img)
figure_sub_1.set_title("Hue and value thresholded image")
figure_sub_1.axis("off")

figure.tight_layout()
plt.show()