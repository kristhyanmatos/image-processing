# Filter link: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_histogram_matching.html#sphx-glr-auto-examples-color-exposure-plot-histogram-matching-py
import cv2
from skimage.exposure import match_histograms
from PIL import Image
from numpy import asarray

video = cv2.VideoCapture(0)
image = Image.open("images/pp.jpg")

image_reference = asarray(image)


while True:
    conect, frame = video.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    matched = match_histograms(frame, image_reference, multichannel=True)

    cv2.imshow("Video filter gray", frame_gray)
    cv2.imshow("Video", matched)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

video.release()
cv2.destroyAllWindows()