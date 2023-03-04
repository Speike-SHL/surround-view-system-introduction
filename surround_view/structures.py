
class ImageFrame(object):                   # 图像帧格式
    def __init__(self, timestamp, image):
        self.timestamp = timestamp          # 时间戳
        self.image = image                  # 图像


class ThreadStatisticsData(object):         # 线程统计数据

    def __init__(self):
        self.average_fps = 0                # 平均帧率
        self.frames_processed_count = 0     # 处理过的帧数
