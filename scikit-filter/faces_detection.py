# Filter link: https://scikit-image.org/docs/stable/auto_examples/applications/plot_face_detection.html#sphx-glr-auto-examples-applications-plot-face-detection-py
from skimage import data
from matplotlib import patches
from skimage.feature import Cascade
import matplotlib.pyplot as plt

# Load the trained file from the module root.
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade.
detector = Cascade(trained_file)

from PIL import Image
from numpy import asarray

# Open the image form working directory
file_image = Image.open("images/pp.jpg")

# convert image to numpy array
image = asarray(file_image)

detected = detector.detect_multi_scale(
    img=image, scale_factor=1.2, step_ratio=1, min_size=(60, 60), max_size=(500, 500)
)

plt.imshow(image)
iamge_shape = plt.gca()
plt.set_cmap("gray")

for patch in detected:

    iamge_shape.add_patch(
        patches.Rectangle(
            (patch["c"], patch["r"]),
            patch["width"],
            patch["height"],
            fill=False,
            color="g",
            linewidth=2,
        )
    )

plt.show()