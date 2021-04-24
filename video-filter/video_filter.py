# Filter link: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_histogram_matching.html#sphx-glr-auto-examples-color-exposure-plot-histogram-matching-py
import cv2
import dlib
from skimage.exposure import match_histograms
from PIL import Image
from numpy import asarray

video = cv2.VideoCapture(0)
detectar_rostos = dlib.get_frontal_face_detector()
image = Image.open("images/pp.jpg")

imagem_referencia = asarray(image)


while True:
    conect, frame = video.read()

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    matched = match_histograms(frame, imagem_referencia, multichannel=True)

    cv2.imshow("Video filter gray", grayFrame)
    cv2.imshow("Video", matched)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

video.release()
cv2.destroyAllWindows()