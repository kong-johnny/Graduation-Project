from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# # STEP 1: Import the necessary modules.
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # STEP 2: Create an HandLandmarker object.
# base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
# options = vision.HandLandmarkerOptions(base_options=base_options,
#                                        num_hands=2)
# detector = vision.HandLandmarker.create_from_options(options)

# # # convert the image from rgb to bgr
# # image = cv2.imread("testhand3.png")
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # cv2.imwrite("testhand3-bgr.png", image)

# # STEP 3: Load the input image.
# image = mp.Image.create_from_file("testhand3-bgr.png")

# # STEP 4: Detect hand landmarks from the input image.
# detection_result = detector.detect(image)

# # STEP 5: Process the classification result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# # cv2.imwrite(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.imwrite("testhand_mediapipe_result.png", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# import mediapipe as mp

# Use OpenCV’s VideoCapture to load the input video.

# Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
# You’ll need it to calculate the timestamp for each frame.

# Loop through each frame in the video using VideoCapture#read()

# Convert the frame received from OpenCV to a MediaPipe’s Image object.
# mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)

def get_hand_landmarks(image: np.ndarray):
  # STEP 1: Import the necessary modules.
  import mediapipe as mp
  from mediapipe.tasks import python
  from mediapipe.tasks.python import vision

  # STEP 2: Create an HandLandmarker object.
  base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
  options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
  detector = vision.HandLandmarker.create_from_options(options)

  # # convert the image from rgb to bgr
  # image = cv2.imread("testhand3.png")
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # cv2.imwrite("testhand3-bgr.png", image)

  # STEP 3: Load the input image.
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

  # STEP 4: Detect hand landmarks from the input image.
  detection_result = detector.detect(image)

  # STEP 5: Process the classification result. In this case, visualize it.
  annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
  # cv2.imwrite(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#   cv2.imwrite("testhand_mediapipe_result.png", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
  return annotated_image

if __name__ == "__main__":
    image = cv2.imread("testhand4.jpg")
    result = get_hand_landmarks(image)
    cv2.imwrite("testhand_mediapipe_result.png", result)