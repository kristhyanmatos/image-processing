from skimage import data
from skimage.feature import Cascade

import matplotlib.pyplot as plt
from matplotlib import patches

# Load the trained file from the module root.
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade.
detector = Cascade(trained_file)

from PIL import Image
from numpy import asarray

# Open the image form working directory
file_image = Image.open("images/pp.jpg")

# convert image to numpy array
img = asarray(file_image)

detected = detector.detect_multi_scale(
    img=img, scale_factor=1.2, step_ratio=1, min_size=(60, 60), max_size=(500, 500)
)

plt.imshow(img)
img_desc = plt.gca()
plt.set_cmap("gray")

for patch in detected:

    img_desc.add_patch(
        patches.Rectangle(
            (patch["c"], patch["r"]),
            patch["width"],
            patch["height"],
            fill=False,
            color="r",
            linewidth=2,
        )
    )

plt.show()