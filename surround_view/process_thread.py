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
        drop_if_full: 如果缓存区满,则删除
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

        while True:
            # ?这一段是干嘛的
            self.stop_mutex.lock()  # 线程锁
            if self.stopped:
                self.stopped = False
                self.stop_mutex.unlock()
                break
            self.stop_mutex.unlock()
            # ?这一段是干嘛的

            # elapsed():计时结束，返回自上次调用start（）或restart（）以来经过的毫秒数。
            self.processing_time = self.clock.elapsed()
            self.clock.start()

            self.processing_mutex.lock()  # 处理线程上锁
            raw_frame = self.capture_buffer_manager.get_device(self.device_id).get()  # 原始图像
            und_frame = self.camera_model.undistort(raw_frame.image)  # 去畸变
            pro_frame = self.camera_model.project(und_frame)  # 射影变换
            flip_frame = self.camera_model.flip(pro_frame)  # 翻转图像
            self.processing_mutex.unlock()  # 处理线程解锁

            self.proc_buffer_manager.sync(self.device_id)
            self.proc_buffer_manager.set_frame_for_device(self.device_id, flip_frame)

            # 更新统计数据
            self.update_fps(self.processing_time)
            self.stat_data.frames_processed_count += 1
            # 通知GUI更新统计数据
            self.update_statistics_gui.emit(self.stat_data)
