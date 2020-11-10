# Imports
import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist
from scipy.spatial import ConvexHull


def eye_size(eye):
    eyeWidth = dist.euclidean(eye[0], eye[3])
    hull = ConvexHull(eye)
    eyeCenter = np.mean(eye[hull.vertices, :], axis=0)

    eyeCenter = eyeCenter.astype(int)

    return int(eyeWidth), eyeCenter


def place_eye(frame, eyeCenter, eyeSize):
    eyeSize = int(eyeSize*1.5)

    x1 = int(eyeCenter[0, 0] - (eyeSize/2))
    x2 = int(eyeCenter[0, 0] + (eyeSize/2))
    y1 = int(eyeCenter[0, 1] - (eyeSize/2))
    y2 = int(eyeCenter[0, 1] + (eyeSize/2))

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

        # re-calculate the size to avoid clipping
    eyeOverlayWidth = x2 - x1
    eyeOverlayHeight = y2 - y1

    # calculate the masks for the overlay
    eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(eyeOverlay, eyeOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst


# Path to image and pre trained models.
image_path = "osama.jpg"
cascade_path = "haarcascade_frontalface_default.xml"
predictor_path= "shape_predictor_68_face_landmarks.dat"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascade_path)

# create the landmark predictor
predictor = dlib.shape_predictor(predictor_path)

# Read the image
image = cv2.imread(image_path)
# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
  gray,
  scaleFactor=1.05,
  minNeighbors=5,
  minSize=(100, 100),
  flags=cv2.CASCADE_SCALE_IMAGE
)
print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Converting the OpenCV rectangle coordinates to Dlib rectangle
dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

detected_landmarks = predictor(image, dlib_rect).parts()
landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
# copying the image so we can see side-by-side
image_copy = image.copy()

for idx, point in enumerate(landmarks):
  pos = (point[0, 0], point[0, 1])

  # annotate the positions
  cv2.putText(image_copy, str(idx), pos,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=(0, 0, 255))

  # draw points on the landmark positions
  cv2.circle(image_copy, pos, 3, color=(0, 255, 255))

# cv2.imshow('Original image', image)
# cv2.imshow('Point Draw image', image_copy)

# If you look closely at the numbers (and check with several images), you’ll notice that the feature points are always coming in the same order. That is,
# 1. Points 0 to 16 is the Jawline
# 2. Points 17 to 21 is the Right Eyebrow
# 3. Points 22 to 26 is the Left Eyebrow
# 4. Points 27 to 35 is the Nose
# 5. Points 36 to 41 is the Right Eye
# 6. Points 42 to 47 is the Left Eye
# 7. Points 48 to 60 is Outline of the Mouth
# 8. Points 61 to 67 is the Inner line of the Mouth

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

landmarks = np.matrix([[p.x, p.y]
              for p in predictor(image, dlib_rect).parts()])

landmarks_display = landmarks[RIGHT_EYE_POINTS + LEFT_EYE_POINTS]

RADIUS = 2
COLOR = (0, 255, 255)

for idx, point in enumerate(landmarks_display):
    pos = (point[0, 0], point[0, 1])
    cv2.circle(image, pos, RADIUS, color=COLOR, thickness=-1)
# cv2.imshow('Only eyes', image)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(predictor_path)

# ---------------------------------------------------------
# Load and pre-process the eye-overlay
# ---------------------------------------------------------
# Load the image to be used as our overlay
imgEye = cv2.imread('Eye.png', -1)

# Create the mask from the overlay image
orig_mask = imgEye[:, :, 3]

# Create the inverted mask for the overlay image
orig_mask_inv = cv2.bitwise_not(orig_mask)

# Convert the overlay image image to BGR
# and save the original image size
imgEye = imgEye[:, :, 0:3]
origEyeHeight, origEyeWidth = imgEye.shape[:2]

# Start capturing the WebCam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            x = rect.left()
            y = rect.top()
            x1 = rect.right()
            y1 = rect.bottom()

            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])

            left_eye = landmarks[LEFT_EYE_POINTS]
            right_eye = landmarks[RIGHT_EYE_POINTS]

            leftEyeSize, leftEyeCenter = eye_size(left_eye)
            rightEyeSize, rightEyeCenter = eye_size(right_eye)

            place_eye(frame, leftEyeCenter, leftEyeSize)
            place_eye(frame, rightEyeCenter, rightEyeSize)

        cv2.imshow("Faces with Overlay", frame)

    ch = 0xFF & cv2.waitKey(1)

    if ch == ord('q'):
        break

# cv2.waitKey(0)
cv2.destroyAllWindows()
