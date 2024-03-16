from mmpose.apis import MMPoseInferencer

model = MMPoseInferencer("rtmpose-m_8xb256-210e_hand5-256x256")

imagepath = "testhand4.jpg"

# ##
# defaultdict(<class 'list'>, 
#             {'visualization': None, 
#              'predictions': 
#                   [[{'keypoints': 
#                       [[67.4736328125, 1467.98876953125], [79.96875, 1449.24609375], [79.96875, 1436.7509765625], [86.21630859375, 1455.49365234375], [92.4638671875, 1474.236328125], [73.72119140625, 1449.24609375], [67.4736328125, 1461.7412109375], [73.72119140625, 1467.98876953125], [79.96875, 1480.48388671875], [73.72119140625, 1455.49365234375], [67.4736328125, 1474.236328125], [73.72119140625, 1480.48388671875], [86.21630859375, 1480.48388671875], [67.4736328125, 1467.98876953125], [73.72119140625, 1480.48388671875], [73.72119140625, 1480.48388671875], [73.72119140625, 1486.7314453125], [61.22607421875, 1480.48388671875], [61.22607421875, 1486.7314453125], [61.22607421875, 1486.7314453125], [67.4736328125, 1492.97900390625]], 
#               'keypoint_scores': 
#                       [0.587124228477478, 0.6191263198852539, 0.6422615051269531, 0.6474728584289551, 0.6125402450561523, 0.6502897143363953, 0.6509199142456055, 0.6388750076293945, 0.5897471904754639, 0.6270714402198792, 0.6422634720802307, 0.6230915784835815, 0.5590161085128784, 0.6262540817260742, 0.6207543611526489, 0.6276723146438599, 0.6035902500152588, 0.6079323291778564, 0.626400351524353, 0.6136724948883057, 0.6248399615287781], 
#               'bbox': 
#                   ([0.0, 0.0, 2559.0, 1599.0],), 
#               'bbox_score': 
#                   1.0}]]})
# ##
result_generator = model(imagepath, visualization=True)
result = next(result_generator)


# print(result)
# exit()

import cv2
# draw the result
image = cv2.imread(imagepath)
keypoints = result['predictions'][0][0]['keypoints']
keypoint_scores = result['predictions'][0][0]['keypoint_scores']
bbox = result['predictions'][0][0]['bbox'][0]
bbox_score = result['predictions'][0][0]['bbox_score']
for i, (x, y) in enumerate(keypoints):
    
    if keypoint_scores[i] > 0.5:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    else:
        # draw the keypoints with low confidence in red
        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
cv2.putText(image, f"{bbox_score:.2f}", (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
# cv2.imwrite("testhand.png", image)
cv2.imwrite("testhand4_result.png", image)
