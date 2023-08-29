import os
import numpy as np
import cv2
from PIL import Image
from PyQt5.QtCore import QMutex, QWaitCondition, QMutexLocker
from .base_thread import BaseThread
from .imagebuffer import Buffer
from . import param_settings as settings
from .param_settings import xl, xr, yt, yb
from . import utils


class ProjectedImageBuffer(object):
    """
    用于同步来自不同摄像机的 processing thread
    """

    def __init__(self, drop_if_full=True, buffer_size=8):
        self.drop_if_full = drop_if_full  # 队列满的话就把新数据直接丢弃
        self.buffer = Buffer(buffer_size)
        self.sync_devices = set()  # 创建一个集合保存需要进行线程同步的设备id
        self.wc = QWaitCondition()
        self.mutex = QMutex()
        self.arrived = 0  # 计数器,
        self.current_frames = dict()  # 当前帧字典,键为设备id,值为图像帧

    def bind_thread(self, thread):
        with QMutexLocker(self.mutex):
            self.sync_devices.add(thread.device_id)  # 将设备id添加到同步设备

        name = thread.camera_model.camera_name  # 获取相机的名字
        shape = settings.project_shapes[name]  # 获取射影形状(宽,高)
        # 在字典中创建一个numpy数组,数组维度为(高,宽,通道数3),类型为uint8,即范围为0-255,初始全为0
        self.current_frames[thread.device_id] = np.zeros(shape[::-1] + (3,), np.uint8)
        # 线程管理器,将不同相机的process thread线程绑定到此类对象上,方便统一管理
        thread.proc_buffer_manager = self

    def get(self):
        """
        从缓冲区读取数据
        """
        return self.buffer.get()

    def set_frame_for_device(self, device_id, frame):
        """
        将图像帧添加到current_frames字典中的设备id上
        """
        if device_id not in self.sync_devices:
            raise ValueError("Device not held by the buffer: {}".format(device_id))
        self.current_frames[device_id] = frame

    def sync(self, device_id):
        self.mutex.lock()
        # 只在指定的设备间(sync_devices)进行同步
        if device_id in self.sync_devices:
            # 每个线程处理完调用sync方法, arrived+1,
            self.arrived += 1
            # we are the last to arrive: wake all waiting threads
            if self.arrived == len(self.sync_devices):
                self.buffer.add(self.current_frames, self.drop_if_full)  # 添加到缓冲区,后续BirdView线程还要读取数据进行处理
                self.wc.wakeAll()
            # 当还有设备没有处理完,进行等待
            else:
                self.wc.wait(self.mutex)
            # 同步完-1
            self.arrived -= 1
        self.mutex.unlock()

    def wake_all(self):
        """
        唤醒所有线程
        """
        with QMutexLocker(self.mutex):
            self.wc.wakeAll()

    def __contains__(self, device_id):
        """
        判断设备id是否在同步设备中
        """
        return device_id in self.sync_devices

    def __str__(self):
        """
        返回类名,同步的设备id
        """
        return (self.__class__.__name__ + ":\n" + \
                "devices: {}\n".format(self.sync_devices))


"""
图像数组为,(高,宽)即(y,x),下面几个函数分别将每个相机的图像分为了三部分
"""


def FI(front_image):
    """
    front_Image的左
    """
    return front_image[:, :xl]


def FII(front_image):
    """
    front_Image的右
    """
    return front_image[:, xr:]


def FM(front_image):
    """
    front_Image的中
    """
    return front_image[:, xl:xr]


def BIII(back_image):
    """
    back_Image的左
    """
    return back_image[:, :xl]


def BIV(back_image):
    """
    back_Image的右
    """
    return back_image[:, xr:]


def BM(back_image):
    """
    back_Image的中
    """
    return back_image[:, xl:xr]


def LI(left_image):
    """
    left_Image的上
    """
    return left_image[:yt, :]


def LIII(left_image):
    """
    left_Image的下
    """
    return left_image[yb:, :]


def LM(left_image):
    """
    left_Image的中
    """
    return left_image[yt:yb, :]


def RII(right_image):
    """
    right_Image的上
    """
    return right_image[:yt, :]


def RIV(right_image):
    """
    right_Image的下
    """
    return right_image[yb:, :]


def RM(right_image):
    """
    right_Image的中
    """
    return right_image[yt:yb, :]


class BirdView(BaseThread):
    """
    鸟瞰图线程,将处理后的四个相机的图像进行鸟瞰图生成相关的操作
    """

    def __init__(self,
                 proc_buffer_manager=None,
                 drop_if_full=True,
                 buffer_size=8,
                 parent=None):
        super(BirdView, self).__init__(parent)
        self.proc_buffer_manager = proc_buffer_manager
        self.drop_if_full = drop_if_full
        self.buffer = Buffer(buffer_size)
        # 创建一个numpy数组,数组维度为(高,宽,通道数3),类型为uint8,即范围为0-255,初始全为0
        self.image = np.zeros((settings.total_h, settings.total_w, 3), np.uint8)
        self.weights = None
        self.masks = None
        self.car_image = settings.car_image
        self.frames = None

    def get(self):
        """
        从缓冲区读取数据
        """
        return self.buffer.get()

    def update_frames(self, images):
        """
        更新当前帧
        """
        self.frames = images

    def load_weights_and_masks(self, weights_image, masks_image):
        """
        加载权重和掩码图片,参数都为路径
        """
        # 先转化为RGBA图像,然后除以255,这步后GMat为(高,宽,RGBA4)的3维数组
        GMat = np.asarray(Image.open(weights_image).convert("RGBA"), dtype=float) / 255.0
        # 将GMat沿着第三个维度进行拼接,结束后weights是一个四维数组(高,宽,重复三个一样的,RGBA),array(1)~array(4)分别是RGBA
        # array(1)为例, 是R通道的元素, 最小单位是3个一样的单个像素上的R通道值
        # 因为乘权重矩阵时候,是把RGB三个通道上都相乘,所以是三个一样的值
        self.weights = [np.stack((GMat[:, :, k],
                                  GMat[:, :, k],
                                  GMat[:, :, k]), axis=2)
                        for k in range(4)]

        Mmat = np.asarray(Image.open(masks_image).convert("RGBA"), dtype=float)
        Mmat = utils.convert_binary_to_bool(Mmat)
        self.masks = [Mmat[:, :, k] for k in range(4)]  # 三维数组, (高,宽,RGBA)

    def merge(self, imA, imB, k):
        """
        重叠部分乘以权重矩阵
        """
        G = self.weights[k]
        return (imA * G + imB * (1 - G)).astype(np.uint8)

    @property  # 装饰器,可以把方法直接当成类的属性调用
    def FL(self):
        return self.image[:yt, :xl]

    @property
    def F(self):
        return self.image[:yt, xl:xr]

    @property
    def FR(self):
        return self.image[:yt, xr:]

    @property
    def BL(self):
        return self.image[yb:, :xl]

    @property
    def B(self):
        return self.image[yb:, xl:xr]

    @property
    def BR(self):
        return self.image[yb:, xr:]

    @property
    def L(self):
        return self.image[yt:yb, :xl]

    @property
    def R(self):
        return self.image[yt:yb, xr:]

    @property
    def C(self):
        return self.image[yt:yb, xl:xr]

    def stitch_all_parts(self):
        """
        处理所有部分
        """
        front, back, left, right = self.frames
        np.copyto(self.F, FM(front))  # 将FM函数处理好的数组赋值给F变量(这里用了装饰器把函数装饰为变量)
        np.copyto(self.B, BM(back))
        np.copyto(self.L, LM(left))
        np.copyto(self.R, RM(right))
        np.copyto(self.FL, self.merge(FI(front), LI(left), 0))
        np.copyto(self.FR, self.merge(FII(front), RII(right), 1))
        np.copyto(self.BL, self.merge(BIII(back), LIII(left), 2))
        np.copyto(self.BR, self.merge(BIV(back), RIV(right), 3))

    def stitch_all_parts_simple(self):
        """
        简单的1/2拼接融合重叠区域
        """
        front, back, left, right = self.frames
        np.copyto(self.F, FM(front))
        np.copyto(self.B, BM(back))
        np.copyto(self.L, LM(left))
        np.copyto(self.R, RM(right))
        np.copyto(self.FL, (FI(front) * 1 / 2 + LI(left) * 1 / 2).astype(np.uint8))
        np.copyto(self.FR, (FII(front) * 1 / 2 + RII(right) * 1 / 2).astype(np.uint8))
        np.copyto(self.BL, (BIII(back) * 1 / 2 + LIII(left) * 1 / 2).astype(np.uint8))
        np.copyto(self.BR, (BIV(back) * 1 / 2 + RIV(right) * 1 / 2).astype(np.uint8))

    def copy_car_image(self):
        """
        把车的照片copy到C中
        """
        np.copyto(self.C, self.car_image)

    def make_luminance_balance(self):
        """
        亮度平衡
        """

        def tune(x):
            if x >= 1:
                return x * np.exp((1 - x) * 0.5)
            else:
                return x * np.exp((1 - x) * 0.8)

        front, back, left, right = self.frames
        m1, m2, m3, m4 = self.masks
        Fb, Fg, Fr = cv2.split(front)
        Bb, Bg, Br = cv2.split(back)
        Lb, Lg, Lr = cv2.split(left)
        Rb, Rg, Rr = cv2.split(right)

        a1 = utils.mean_luminance_ratio(RII(Rb), FII(Fb), m2)
        a2 = utils.mean_luminance_ratio(RII(Rg), FII(Fg), m2)
        a3 = utils.mean_luminance_ratio(RII(Rr), FII(Fr), m2)

        b1 = utils.mean_luminance_ratio(BIV(Bb), RIV(Rb), m4)
        b2 = utils.mean_luminance_ratio(BIV(Bg), RIV(Rg), m4)
        b3 = utils.mean_luminance_ratio(BIV(Br), RIV(Rr), m4)

        c1 = utils.mean_luminance_ratio(LIII(Lb), BIII(Bb), m3)
        c2 = utils.mean_luminance_ratio(LIII(Lg), BIII(Bg), m3)
        c3 = utils.mean_luminance_ratio(LIII(Lr), BIII(Br), m3)

        d1 = utils.mean_luminance_ratio(FI(Fb), LI(Lb), m1)
        d2 = utils.mean_luminance_ratio(FI(Fg), LI(Lg), m1)
        d3 = utils.mean_luminance_ratio(FI(Fr), LI(Lr), m1)

        t1 = (a1 * b1 * c1 * d1) ** 0.25
        t2 = (a2 * b2 * c2 * d2) ** 0.25
        t3 = (a3 * b3 * c3 * d3) ** 0.25

        x1 = t1 / (d1 / a1) ** 0.5
        x2 = t2 / (d2 / a2) ** 0.5
        x3 = t3 / (d3 / a3) ** 0.5

        x1 = tune(x1)
        x2 = tune(x2)
        x3 = tune(x3)

        Fb = utils.adjust_luminance(Fb, x1)
        Fg = utils.adjust_luminance(Fg, x2)
        Fr = utils.adjust_luminance(Fr, x3)

        y1 = t1 / (b1 / c1) ** 0.5
        y2 = t2 / (b2 / c2) ** 0.5
        y3 = t3 / (b3 / c3) ** 0.5

        y1 = tune(y1)
        y2 = tune(y2)
        y3 = tune(y3)

        Bb = utils.adjust_luminance(Bb, y1)
        Bg = utils.adjust_luminance(Bg, y2)
        Br = utils.adjust_luminance(Br, y3)

        z1 = t1 / (c1 / d1) ** 0.5
        z2 = t2 / (c2 / d2) ** 0.5
        z3 = t3 / (c3 / d3) ** 0.5

        z1 = tune(z1)
        z2 = tune(z2)
        z3 = tune(z3)

        Lb = utils.adjust_luminance(Lb, z1)
        Lg = utils.adjust_luminance(Lg, z2)
        Lr = utils.adjust_luminance(Lr, z3)

        w1 = t1 / (a1 / b1) ** 0.5
        w2 = t2 / (a2 / b2) ** 0.5
        w3 = t3 / (a3 / b3) ** 0.5

        w1 = tune(w1)
        w2 = tune(w2)
        w3 = tune(w3)

        Rb = utils.adjust_luminance(Rb, w1)
        Rg = utils.adjust_luminance(Rg, w2)
        Rr = utils.adjust_luminance(Rr, w3)

        self.frames = [cv2.merge((Fb, Fg, Fr)),
                       cv2.merge((Bb, Bg, Br)),
                       cv2.merge((Lb, Lg, Lr)),
                       cv2.merge((Rb, Rg, Rr))]
        return self

    def get_weights_and_masks(self, images):
        front, back, left, right = images
        G0, M0 = utils.get_weight_mask_matrix(FI(front), LI(left))
        G1, M1 = utils.get_weight_mask_matrix(FII(front), RII(right))
        G2, M2 = utils.get_weight_mask_matrix(BIII(back), LIII(left))
        G3, M3 = utils.get_weight_mask_matrix(BIV(back), RIV(right))
        self.weights = [np.stack((G, G, G), axis=2) for G in (G0, G1, G2, G3)]
        self.masks = [(M / 255.0).astype(int) for M in (M0, M1, M2, M3)]
        return np.stack((G0, G1, G2, G3), axis=2), np.stack((M0, M1, M2, M3), axis=2)

    def make_white_balance(self):
        """
        白平衡
        """
        self.image = utils.make_white_balance(self.image)

    def run(self):
        if self.proc_buffer_manager is None:
            # 这个线程需要一个缓冲区去运行
            raise ValueError("This thread requires a buffer of projected images to run")

        while True:  # 无线循环,处理图像
            self.stop_mutex.lock()
            if self.stopped:  # 如果其他地方将stopped改为true,则停止循环,停止处理图像
                self.stopped = False
                self.stop_mutex.unlock()
                break
            self.stop_mutex.unlock()  # 前后上锁防止其他线程改变停止信号

            # elapsed():计时结束，返回自上次调用start（）或restart（）以来经过的毫秒数。
            self.processing_time = self.clock.elapsed()
            self.clock.start()

            self.processing_mutex.lock()
            self.update_frames(self.proc_buffer_manager.get().values())  # 更新当前帧,得到保存了处理后的前后左右四个相机图像的帧
            if settings.SAVE_BRIDVIEW_PROCESS:
                self.stitch_all_parts_simple()
                self.copy_car_image()
                tmp_img1 = self.image.copy()
                self.image = np.zeros((settings.total_h, settings.total_w, 3), np.uint8)
                cv2.imwrite(f"{settings.WORK_PATH}/paper_need_img/birdview_merge_half.jpg", tmp_img1)
                self.stitch_all_parts()
                self.copy_car_image()
                tmp_img2 = self.image.copy()
                self.image = np.zeros((settings.total_h, settings.total_w, 3), np.uint8)
                cv2.imwrite(f"{settings.WORK_PATH}/paper_need_img/birdview_without_balance.jpg", tmp_img2)
            self.make_luminance_balance().stitch_all_parts()  # 亮度平衡并拼接所有部分
            self.make_white_balance()  # 白平衡
            self.copy_car_image()  # 添加车的图像
            if settings.SAVE_BRIDVIEW_PROCESS:
                tmp_img3 = self.image.copy()
                cv2.imwrite(f"{settings.WORK_PATH}/paper_need_img/birdview_with_balance.jpg", tmp_img3)
                settings.SAVE_BRIDVIEW_PROCESS = False
            self.buffer.add(self.image.copy(), self.drop_if_full)  # 添加处理好的image到缓冲区
            self.processing_mutex.unlock()

            # 更新统计数据
            self.update_fps(self.processing_time)
            self.stat_data.frames_processed_count += 1
            # 通知GUI更新统计数据
            self.update_statistics_gui.emit(self.stat_data)
