import cv2
from rmn import RMN
import numpy as np

# load RMN model
m = RMN()

# read video
video_path = '/public/home/huangshisheng/gaussian-splatting/endingDesign/record_video/out1709367446.0161347.avi'
cap = cv2.VideoCapture(video_path)

# get video info
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# define output video
output_path = '/public/home/huangshisheng/gaussian-splatting/endingDesign/record_video/output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
cnt = 0

# get the frames count
frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames: ", frames_count)
import tqdm
for i in tqdm.tqdm(range(frames_count)):

# while True:
    ret, frame = cap.read()
    if not ret:
        break
    cnt += 1
    print("Processing frame: ", cnt, end=' ')

    # # crop frame as 1/4
    # h, w, _ = frame.shape
    # left = w // 4
    # right = w - w // 4
    # top = h // 4
    # bottom = h - h // 4
    # frame = frame[top:bottom, left:right]
    # frame = cv2.resize(frame, (frame_width, frame_height))

    # cv2.imwrite('frame.jpg', frame)
    # exit()

    from handpose_mediapipe import get_hand_landmarks
    
    face_frame = get_hand_landmarks(frame)

    # facial expression recognition
    expression_frame = m.detect_emotion_for_single_frame(face_frame)

    
    
    # overlay expression on the frame
    final_frame = m.draw(frame, expression_frame)
    
    # write the frame into the output video
    out.write(final_frame)
    
    # # show the frame
    # cv2.imshow('Frame', final_frame)
    
    # # quit by pressing 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# release the video capture and video writer
cap.release()
out.release()
cv2.destroyAllWindows()
