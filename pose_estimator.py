"""
! author: enpei
! Date: 2021-12-23
封装常用工具，降低Demo复杂度

"""
import cv2
import numpy as np


class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        self.model_points_68 = self._get_full_model_points()
        self.image_points = None

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None

    def _get_full_model_points(self, filename='asserts/model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

    def show_3d_model(self):
        import matplotlib.pyplot as plt


        x = self.model_points_68[:, 0]
        y = self.model_points_68[:, 1]
        z = self.model_points_68[:, 2]

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        ax.scatter3D(x, y, z)
        plt.show()

    # 利用相似三角形估算距离
    
    def get_distance(self,eyeBaseDistance):
        image_points = self.image_points
        left_x = int(image_points[36][0])
        right_x = int(image_points[45][0])

        pixel_dist = abs(right_x-left_x)

        return (eyeBaseDistance / pixel_dist) * 1.5


    def get_image_points(self, landmarks):
        landmarks_list = []
        # if landmarks is numpy array
        if isinstance(landmarks, np.ndarray):
            for n in range(0, 68):
                x = landmarks[n, 0]
                y = landmarks[n, 1]
                landmarks_list.append([x, y])
            image_points = np.array(landmarks_list, dtype="double")
            self.image_points = image_points
            return image_points
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_list.append([x,y])

        image_points = np.array(landmarks_list,dtype="double")

        self.image_points = image_points

        return image_points



    def solve_pose_by_68_points(self, landmarks):

        image_points = self.get_image_points(landmarks)
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def draw_annotation_box(self, image, rotation_vector, translation_vector, is_watch, line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        color = (255, 0, 255) if is_watch =='是' else (0, 255, 0)

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    # 坐标系
    def draw_axes(self, img, R, t):
        img	= cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeefs, R, t, 200)

    # 方向指针
    def draw_pointer(self,img, R, t):
        point1 = ( int(self.image_points[30][0]), int(self.image_points[30][1]))
        
        nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 300.0)]), R, t, self.camera_matrix, self.dist_coeefs)

        point2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
        cv2.line(img, point1, point2, (0,255,0), 4)

    def cal_3d_position(self, nose_position: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        turn the pos of nose to world coo

        参数:
            - point: 形为(x, y)的点坐标。
            - R: np.ndarray，3x3的旋转矩阵。
            - t: np.ndarray，3x1的平移向量。

        返回:
            - world_point: (X, Y, Z)，变换后的点坐标。
        """
        # print(nose_position.shape)
        assert(nose_position.shape == (2,)), "nose_position must be a 2 array"
        assert(R.shape == (3, 3)), "R must be a 3x3 array"
        assert(t.shape == (3, 1)), "t must be a 3x1 array"

        # 将二维点扩展为齐次坐标 (x, y, 1)
        nose_position_homogeneous = np.array([nose_position[0], nose_position[1], 1.0])
        # print("nose pos h: ", nose_position_homogeneous)
        # (u, v, 1)^T = M [ R (x, y, z) + t ]
        
        M = np.matrix(self.camera_matrix)
        nose_position_homogeneous = np.dot(M.I, nose_position_homogeneous)
        # print("M^-1 * nose pos h: ", nose_position_homogeneous)
        nose_position_homogeneous = nose_position_homogeneous - t.reshape((1, 3))
        # print("nose - t: ", nose_position_homogeneous)
        R = np.matrix(R)
        # print(R, R.I)
        # 使用旋转矩阵和平移向量计算变换后的坐标
        nose_position_homogeneous = nose_position_homogeneous.reshape((3, 1))
        world_point = np.dot(R.I, nose_position_homogeneous)

        return world_point



    
    