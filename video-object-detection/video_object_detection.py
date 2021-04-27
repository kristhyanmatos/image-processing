# Link Base: https://learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/
import cv2
import sys
from PIL import Image
from numpy import asarray
from random import randint
from skimage.exposure import match_histograms

cap = cv2.VideoCapture(0)
image = Image.open("images/pp.jpg")
image_reference = asarray(image)


def scale_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def scale_filter(frame):
    return match_histograms(frame, image_reference, multichannel=True)


def selectROIfromFrame(frame):
    # box (x, y, lar, alt)
    box = cv2.selectROI("SELECT ROI", frame, fromCenter=False, showCrosshair=False)
    print(box)
    return box


if __name__ == "__main__":

    conect, first_frame = cap.read()
    # Gray Scale
    # first_frame = scale_gray(first_frame)

    # Match Histogram Scale
    # first_frame = scale_filter(first_frame)

    if not conect:
        sys.exit()

    # Select boxes
    bboxes = []
    colors = []

    while True:
        box = selectROIfromFrame(first_frame)
        bboxes.append(box)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if k == 113:  # q is pressed
            break

    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()

    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(cv2.legacy.TrackerCSRT_create(), first_frame, bbox)

    index = 1
    print("Press x to quit")
    while cap.isOpened():
        conect, frame = cap.read()

        # Gray Scale
        # frame = scale_gray(frame)

        # Match Histogram Scale
        # frame = scale_filter(frame)

        if not conect:
            break
        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)
        if success:
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        else:
            print("[", index, "]", "Failed")
            index = index + 1

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) == ord("x"):
            break

    cv2.destroyAllWindows()