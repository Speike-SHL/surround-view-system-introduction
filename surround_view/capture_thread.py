import cv2
from PyQt5.QtCore import qDebug

from .base_thread import BaseThread
from .structures import ImageFrame
from .utils import gstreamer_pipeline


class CaptureThread(BaseThread):
    """
    捕获线程，
    """
    def __init__(self,
                 device_id,
                 flip_method=2,
                 drop_if_full=True,
                 api_preference=cv2.CAP_GSTREAMER,
                 resolution=None,
                 use_gst=False,
                 parent=None):
        """
        device_id: 相机设备id
        flip_method: 0是不变，2是180度旋转
        use_gst: 是否使用gstreamer打开相机
        drop_if_full: 如果缓冲区满是否丢弃新帧
        api_preference: cv2.CAP_GSTREAMER for csi cameras, usually cv2.CAP_ANY would suffice.
        resolution: 相机分辨率 (width, height).
        """
        super(CaptureThread, self).__init__(parent)
        self.device_id = device_id
        self.flip_method = flip_method
        self.use_gst = use_gst  # 是否使用gstreamer打开相机
        self.drop_if_full = drop_if_full
        self.api_preference = api_preference
        self.resolution = resolution
        self.cap = cv2.VideoCapture()
        self.buffer_manager = None  # 会被绑定到MultiBufferManager对象上,用于将此捕获线程与其他相机同步

    def run(self):
        if self.buffer_manager is None:
            # 该线程未绑定任何线程管理器
            raise ValueError("This thread has not been binded to any buffer manager yet")

        while True:  # 无线循环,从相机中一直读取帧
            self.stop_mutex.lock()
            if self.stopped:  # 如果其他地方将stopped改为true,则停止循环,停止读取相机数据
                self.stopped = False
                self.stop_mutex.unlock()
                break
            self.stop_mutex.unlock()  # 前后上锁防止其他线程改变停止信号

            self.processing_time = self.clock.elapsed()  # 保存处理时间
            self.clock.start()  # 开启定时器,用于计算捕获速率

            # 如果该线程启用了同步,则调用MultiBufferManager中的sync方法等待同步
            self.buffer_manager.sync(self.device_id)

            if not self.cap.grab():  # 抓取一帧到缓冲区,如果抓取不成功,继续等待下一帧到来
                continue

            _, frame = self.cap.retrieve()  # 从缓冲区读取一帧图像解码为opencv的格式
            img_frame = ImageFrame(self.clock.msecsSinceStartOfDay(), frame)  # 用时间戳和图像构造ImageFrame数据
            self.buffer_manager.get_device(self.device_id).add(img_frame, self.drop_if_full)  # 向缓冲区添加数据

            # 更新统计数据
            self.update_fps(self.processing_time)  # 用处理时间调用Base_Thread线程中的`update_fps`方法更新平均帧率
            self.stat_data.frames_processed_count += 1
            # 通知GUI更新统计数据
            self.update_statistics_gui.emit(self.stat_data)

        qDebug("Stopping capture thread...")

    def connect_camera(self):
        """
        连接相机
        """
        if self.use_gst:
            options = gstreamer_pipeline(cam_id=self.device_id, flip_method=self.flip_method)
            self.cap.open(options, self.api_preference)  # 打开相机
        else:
            self.cap.open(self.device_id)

        if not self.cap.isOpened():  # 如果相机没有打开, 返回False
            qDebug("Cannot open camera {}".format(self.device_id))
            return False
        else:
            if self.resolution is not None:  # 尝试设置相机的分辨率
                width, height = self.resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                # 如果分辨率不支持，某些相机可能会关闭
                if not self.cap.isOpened():
                    qDebug("Resolution not supported by camera device: {}".format(self.resolution))
                    return False
            else:  # 使用默认的分辨率
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.resolution = (width, height)

        return True

    def disconnect_camera(self):
        """
        关闭相机,成功关闭返回True,相机本来就没开返回False
        """
        if self.cap.isOpened():
            self.cap.release()
            return True
        else:
            return False

    def is_camera_connected(self):
        """
        判断相机是否连接
        """
        return self.cap.isOpened()
