import insightface
from insightface.app import FaceAnalysis 
from insightface.data import get_image as ins_get_image
import argparse

app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
# img = ins_get_image('t1')
# faces = app.get(img)
# list ((face.embedding, face.bbox, face.kps) for face in faces)

import cv2
import numpy as np

import rmn
m = rmn.RMN()

# img = cv2.imread("testhand.png")
# faces = app.get(img)
# print(faces)
# exit()
# rimg = app.draw_on(img, faces)
# cv2.imwrite("testhand_facedet_output.jpg", rimg)

def process_video(video_path, output_video_path, output_npy_path, keep_no_face_frame = False):
    # init video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("video info: ", fps, frames_count, width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # init faces_info
    faces_info = []

    from pose_estimator import PoseEstimator
    pose_estimator = PoseEstimator(img_size=(width, height))
    loss_frame = 0
    # for test
    # frames_count = 1000
    # while True:
    import tqdm
    for i in tqdm.tqdm(range(frames_count)):
        # read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # crop frame to 1/4
        left = int(width/4)
        right = int(width/4*3)
        top = int(height/4)
        bottom = int(height/4*3)
        frame = frame[top:bottom, left:right]
        frame = cv2.resize(frame, (width, height))


        # get face info
        # img = ins_get_image(frame)
        faces = app.get(frame)
        # if faces == []:
        #     continue
        # for face in faces:
        #     faces_info.append(face)
        if faces != []:
            

        # sort faces by det_score
            faces.sort(key=lambda x: (x['kps'][2][0] - width/2)**2 + (x['kps'][2][1] - height/2))
            
            face_frame = app.draw_on(frame, faces[0:1])
            # print(faces)
            # crop face from frame
            face_box = faces[0]['bbox']
            merge = 0.2
            face_box[0] = max(0, face_box[0] - merge * (face_box[2] - face_box[0]))
            face_box[1] = max(0, face_box[1] - merge * (face_box[3] - face_box[1]))
            face_box[2] = min(width, face_box[2] + merge * (face_box[2] - face_box[0]))
            face_box[3] = min(height, face_box[3] + merge * (face_box[3] - face_box[1]))
            face_img = frame[int(face_box[1]):int(face_box[3]), int(face_box[0]):int(face_box[2])]

        

            # get facial expression
            res = m.detect_emotion_for_single_frame(face_img)
            # print(type(res))
            if res != [] and 'kps' in faces[0] and 'landmark_3d_68' in faces[0]:
                faces_info.append(faces[0])
                exp = [res[0]['emo_label'], res[0]['emo_proba']]
                if exp[1] < 0.6 and (exp[0] == 'sad' or exp[0] == 'fear' or exp[0] == 'angry'):
                    exp[0] = 'neutral'
                # print(exp)
                faces_info[-1]['expression'] = exp
                # cv2.imwrite('face_tmp.jpg', face_img)
            # if 'kps' in faces[0]:
                kps = faces[0]['kps']
                eyeBaseDistance = 65
                left_x = int(kps[0][0])
                right_x = int(kps[1][0])
                pixel_dist = max(abs(right_x-left_x), 0.1)
                # print(pixel_dist, kps)
                distance = (eyeBaseDistance / pixel_dist) * 1.5
                faces_info[-1]['distance'] = distance
                # print distance on face_frame
                cv2.putText(face_frame, f"distance: {distance:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # print expression on face_frame
                cv2.putText(face_frame, f"expression: {exp[0]}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


                # calculate the angle
                landmark = faces[0]['landmark_3d_68']
                rotation_vector, translation_vector = pose_estimator.solve_pose_by_68_points(landmark)
                # print(rotation_vector, translation_vector)

                
                rmat, jac = cv2.Rodrigues(rotation_vector)
                # print(rmat, jac)
                nose_w = pose_estimator.cal_3d_position(kps[2], rmat, translation_vector)
                cv2.putText(face_frame, f"nose (lr, fr, ud): {nose_w}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # print(nose_w, distance)
                nose_w[0] = (kps[2][0] - width/2) * 7 / width
                nose_w[1] = distance
                nose_w[2] = 1.7
                faces_info[-1]['nose_w'] = nose_w
                # exit()
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                

                faces_info[-1]['angle'] = angles
                # print angle on face_frame
                # cv2.putText(face_frame, f"angle: {angles}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # judge left or right
                # if angles[1] > 0:
                #     cv2.putText(face_frame, f"right: {abs(angles[1])}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # else:
                #     cv2.putText(face_frame, f"left: {abs(angles[1])}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(face_frame, f"yaw: {angles[1]}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # judge up or down
                # if angles[0] > 0:
                #     cv2.putText(face_frame, f"down: {abs(angles[0])}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # else:
                #     cv2.putText(face_frame, f"up: {abs(angles[0])}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(face_frame, f"pitch: {angles[0]}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # judge tilt
                # if angles[2] > 0:
                #     cv2.putText(face_frame, f"tilt: {abs(angles[2])}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # else:
                #     cv2.putText(face_frame, f"not tilt: {abs(angles[2])}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(face_frame, f"roll: {angles[2]}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # draw a arrow as the direction from the center of the eyes along the angle
                # start_point = (int((kps[0][0] + kps[1][0])/2), int((kps[0][1] + kps[1][1])/2))
                # end_point = (int(start_point[0] + 50 * np.cos(angles[1])), int(start_point[1] + 50 * np.sin(angles[1])))
                # cv2.arrowedLine(face_frame, start_point, end_point, (255, 255, 255), 2)
                out.write(face_frame)

        else:
            loss_frame += 1
            if keep_no_face_frame:
                out.write(frame)
        # if i == 200:    
        #     exit()
        
        # write video
        # out.write(face_frame)

    # release video
    cap.release()
    out.release()
    # sace faces_info list to csv
    import csv
    with open(output_npy_path, 'w', newline='') as csvfile:
        fieldnames = faces_info[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for face in faces_info:
            writer.writerow(face)
    # import pandas as pd
    # df = pd.DataFrame(faces_info, columns=faces_info[0].keys())
    # df.to_csv(output_npy_path, index=False)
    # log
    print(f"视频处理完成，人脸信息已保存到 {output_npy_path}")

# from IPython.display import display, Image
# path = "./ppdepic_output.jpg"
# img = Image(path)
# display(img)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate heatmap for face position')
    parser.add_argument('--video_path', type=str, default='/huangshisheng/ldq/HexPlane/endingDesign/data_raw/level1/数学与应用数学05号_湖北师范大学_潘慈忠.wmv', help='path to the video')
    parser.add_argument('--output_video_path', type=str, default='test_video_output.mp4', help='path to the output video')
    parser.add_argument('--output_csv_path', type=str, default='test_faces_info.csv', help='path to the output csv file')
    parser.add_argument('--keep_no_face_frame', type=bool, default=False, help='keep the frame without face')
    args = parser.parse_args()
    video_path = args.video_path
    output_video_path = args.output_video_path
    output_csv_path = args.output_csv_path
    keep_no_face_frame = args.keep_no_face_frame
    process_video(video_path, output_video_path, output_csv_path, keep_no_face_frame)

# video_path = '/huangshisheng/ldq/HexPlane/endingDesign/data_raw/level1/数学与应用数学05号_湖北师范大学_潘慈忠.wmv'  
# output_video_path = 'test_video_v2.mp4' 
# output_csv_path = 'test_faces_info.csv'

# process_video(video_path, output_video_path, output_csv_path)

# output_video_path = 'test_video_no_face_v2.mp4'

# process_video(video_path, output_video_path, output_csv_path, keep_no_face_frame = True)