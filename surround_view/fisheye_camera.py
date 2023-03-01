import os
import numpy as np
import cv2

from . import param_settings as settings


class FisheyeCameraModel(object):

    """
    鱼眼相机模型，用于去畸变，射影变换和翻转相机坐标系
    Fisheye camera model, for undistorted, projecting and flipping camera frames.
    """

    def __init__(self, camera_param_file, camera_name):
        """
        初始化相机模型, 加载参数
        :param camera_param_file: 相机参数yaml文件路径
        :param camera_name: 相机名称
        """

        if not os.path.isfile(camera_param_file):  # 判断是否是一个文件夹路径
            raise ValueError("Cannot find camera param file")

        if camera_name not in settings.camera_names:  # 判断相机名称是否正确
            raise ValueError("Unknown camera name: {}".format(camera_name))

        self.camera_file = camera_param_file
        self.camera_name = camera_name
        self.camera_matrix = None  # 定义相机内参矩阵
        self.dist_coeffs = None  # 定义畸变参数
        self.resolution = None  # 定义分辨率
        self.project_matrix = None  # 射影矩阵
        self.scale_xy = (1.0, 1.0)  # 校正后画面横纵向缩放比
        self.shift_xy = (0, 0)  # 矫正后画面中心的横向和纵向平移距离
        self.undistort_maps = None  # 去畸变矩阵

        self.project_shape = settings.project_shapes[self.camera_name]  # ?射影形状
        self.load_camera_params()  # 加载yaml文件参数

    def load_camera_params(self):
        """
        从yaml文件中加载相机参数
        :return: none
        """

        fs = cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("camera_matrix").mat()  # 相机内参矩阵,3x3
        self.dist_coeffs = fs.getNode("dist_coeffs").mat()  # 畸变参数,4x1
        self.resolution = fs.getNode("resolution").mat().flatten()  # 分辨率,1x2

        project_matrix = fs.getNode("project_matrix").mat()  # 射影矩阵,3x3
        if project_matrix is not None:
            self.project_matrix = project_matrix

        scale_xy = fs.getNode("scale_xy").mat()  # 2x1
        if scale_xy is not None:
            self.scale_xy = scale_xy

        shift_xy = fs.getNode("shift_xy").mat()  # 2x1
        if shift_xy is not None:
            self.shift_xy = shift_xy

        fs.release()
        self.update_undistort_maps()

    def update_undistort_maps(self):
        """
        更新去畸变映射表
        """
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= self.scale_xy[0]  # fx
        new_matrix[1, 1] *= self.scale_xy[1]  # fy
        new_matrix[0, 2] += self.shift_xy[0]  # cx
        new_matrix[1, 2] += self.shift_xy[1]  # cy
        width, height = self.resolution

        # 得到畸变图像与去畸变图像的映射表，后面可以用cv2.fisheye.remap()将畸变图像映射为去畸变图像
        self.undistort_maps = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            np.eye(3),
            new_matrix,
            (width, height),
            cv2.CV_16SC2
        )
        return self

    def set_scale_and_shift(self, scale_xy=(1.0, 1.0), shift_xy=(0, 0)):
        """
        设置校正后图像的纵横缩放比与画面中心平移距离
        """
        self.scale_xy = scale_xy
        self.shift_xy = shift_xy
        self.update_undistort_maps()
        return self

    def undistort(self, image):
        """
        去畸变
        @param [input]  image:原始图像
        @return result:去畸变图像
        """
        result = cv2.remap(image, *self.undistort_maps, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        return result

    def project(self, image):
        """
        进行射影透视变换
        @param [input]  image:原始图像
        @return result:去畸变图像
        """
        # cv2.warpPerspective(输入图像, 射影矩阵, 输出图像大小)
        result = cv2.warpPerspective(image, self.project_matrix, self.project_shape)
        return result

    def flip(self, image):
        """
        把每个相机输出的图像旋转到前后左右对应的角度上
        """
        if self.camera_name == "front":  # 前方图像直接copy
            return image.copy()

        elif self.camera_name == "back":  # 把图像先上下翻转，再左右翻转
            return image.copy()[::-1, ::-1, :]  # img[:,::-1,:]左右翻转，img[::-1,,:]上下翻转

        elif self.camera_name == "left":  # 把图像先转置，再上下翻转，效果上相当于向左旋转90度
            return cv2.transpose(image)[::-1]

        else:
            return np.flip(cv2.transpose(image), 1)  # 把图像先转置，再左右翻转，效果上相当于向右旋转90度

    def save_data(self):
        """
        把相机数据写入yaml文件中储存
        """
        fs = cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", self.camera_matrix)
        fs.write("dist_coeffs", self.dist_coeffs)
        fs.write("resolution", self.resolution)
        fs.write("project_matrix", self.project_matrix)
        fs.write("scale_xy", np.float32(self.scale_xy))
        fs.write("shift_xy", np.float32(self.shift_xy))
        fs.release()
