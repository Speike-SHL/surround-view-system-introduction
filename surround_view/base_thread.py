import cv2
from queue import Queue
from PyQt5.QtCore import (QThread, QTime, QMutex, pyqtSignal, QMutexLocker)
from .structures import ThreadStatisticsData


class BaseThread(QThread):
    """
    所有类型线程的基类，如 capture, processing, stitching 等线程。
    主要用于收集线程的统计数据
    """

    FPS_STAT_QUEUE_LENGTH = 32  # 帧队列长度为32

    update_statistics_gui = pyqtSignal(ThreadStatisticsData)  # 创建信号->更新统计gui,信号类型为ThreadStatisticsData

    def __init__(self, parent=None):
        super(BaseThread, self).__init__(parent)
        self.stopped = False  # ?线程停止标志
        self.stop_mutex = QMutex()  # 创建线程锁
        self.clock = QTime()
        self.fps = Queue()  # 创建帧队列
        self.processing_time = 0
        self.processing_mutex = QMutex()  # 创建线程锁
        self.fps_sum = 0
        self.stat_data = ThreadStatisticsData()

    def stop(self):  # 线程上锁并将线程停止标志变为True
        with QMutexLocker(self.stop_mutex):  # QMutexLocker能将锁自动销毁,要传入QMutex对象
            self.stopped = True

    def update_fps(self, dt):
        # 将帧率填充进队列
        if dt > 0:
            self.fps.put(1000 / dt)  # 1000/每帧时间=帧率

        # 超出队列长度时丢弃队列中的冗余对象
        if self.fps.qsize() > self.FPS_STAT_QUEUE_LENGTH:
            self.fps.get()

        # 更新统计数据
        if self.fps.qsize() == self.FPS_STAT_QUEUE_LENGTH:  # 帧队列中满的话,就计算一次平均帧率
            while not self.fps.empty():  # 队列不空的话
                self.fps_sum += self.fps.get()

            self.stat_data.average_fps = round(self.fps_sum / self.FPS_STAT_QUEUE_LENGTH, 2)
            self.fps_sum = 0
