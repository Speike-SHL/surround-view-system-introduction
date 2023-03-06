import cv2
from PyQt5.QtCore import qDebug, QMutex

from .base_thread import BaseThread


class CameraProcessingThread(BaseThread):

    """
    处理单个相机图像的线程, 如去畸变, 射影, 翻转
    """

    def __init__(self,
                 capture_buffer_manager,
                 device_id,
                 camera_model,
                 drop_if_full=True,
                 parent=None):
        """
        capture_buffer_manager: `MultiBufferManager` 对象的实例
        device_id: 要处理的相机设备号
        camera_model: `FisheyeCameraModel` 对象的实例
        drop_if_full: 缓冲区满是否删除新数据还是等待缓冲区有位置
        """
        super(CameraProcessingThread, self).__init__(parent)
        self.capture_buffer_manager = capture_buffer_manager
        self.device_id = device_id
        self.camera_model = camera_model
        self.drop_if_full = drop_if_full
        # `ProjectedImageBuffer` 对象的实例
        self.proc_buffer_manager = None

    def run(self):
        if self.proc_buffer_manager is None:
            # 这个线程还没有绑定到任何线程
            raise ValueError("This thread has not been binded to any processing thread yet")

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

            self.processing_mutex.lock()  # 处理线程上锁
            raw_frame = self.capture_buffer_manager.get_device(self.device_id).get()  # 原始图像
            und_frame = self.camera_model.undistort(raw_frame.image)  # 去畸变
            pro_frame = self.camera_model.project(und_frame)  # 射影变换
            flip_frame = self.camera_model.flip(pro_frame)  # 翻转图像
            self.processing_mutex.unlock()  # 处理线程解锁

            self.proc_buffer_manager.sync(self.device_id)  # 调用ProjectedImageBuffer类的`sync`方法进行线程同步
            # 将处理后的图像添加到current_frame中对应的device_id上
            self.proc_buffer_manager.set_frame_for_device(self.device_id, flip_frame)

            # 更新统计数据
            self.update_fps(self.processing_time)
            self.stat_data.frames_processed_count += 1
            # 通知GUI更新统计数据
            self.update_statistics_gui.emit(self.stat_data)
