from PyQt5.QtCore import QSemaphore, QMutex
from PyQt5.QtCore import QMutexLocker, QWaitCondition
from queue import Queue
"""
QWaitCondition 用于线程同步,
QWaitCondition主要有两个方法：wait和wakeAll。调用wait方法会使当前线程阻塞等待，直到有其他线程调用wakeAll方法来唤醒它；
调用wakeAll方法会唤醒所有等待的线程。通常在使用QWaitCondition时需要配合QMutex使用，以保证线程安全。
"""

class Buffer(object):
    """
    缓冲区对象,提供生产者和消费者线程操作缓冲区的一系列方法, 生产者线程为 capture_thread, 消费者线程为CameraProcessingThread
    """
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size  # 缓冲区大小
        self.free_slots = QSemaphore(self.buffer_size)  # 初始时刻空闲槽数量
        self.used_slots = QSemaphore(0)  # 初始时刻已使用槽数量, 为0时等待阻塞
        # ?有什么用
        # 答：主要在`clear`方法中使用,在`add`和`get`方法前后调用,这样在使用clear方法时,可以知道是否有线程正在操作缓冲区
        self.clear_buffer_add = QSemaphore(1)
        self.clear_buffer_get = QSemaphore(1)
        self.queue_mutex = QMutex()  # 创建队列互斥锁
        self.queue = Queue(self.buffer_size)  # 创建一个队列

    def add(self, data, drop_if_full=False):
        """
        向缓冲区添加数据
        """
        self.clear_buffer_add.acquire()
        if drop_if_full:  # 队列满的话就把新数据直接丢弃
            if self.free_slots.tryAcquire():  # tryAcquire不会阻塞，有可用许可返回true,使用该许可并-1,否则返回False
                self.queue_mutex.lock()
                self.queue.put(data)
                self.queue_mutex.unlock()
                self.used_slots.release()  # 已使用槽位+1
        else:  # 如果不是直接丢弃模式，就阻塞等待队列的位置
            self.free_slots.acquire()
            self.queue_mutex.lock()
            self.queue.put(data)
            self.queue_mutex.unlock()
            self.used_slots.release()

        self.clear_buffer_add.release()

    def get(self):
        """
        从缓冲区获取数据
        """
        self.clear_buffer_get.acquire()
        self.used_slots.acquire()  # 获取已使用槽位
        self.queue_mutex.lock()
        data = self.queue.get()
        self.queue_mutex.unlock()
        self.free_slots.release()  # 空闲槽位+1
        self.clear_buffer_get.release()
        return data

    def clear(self):
        # 检查缓冲区是否包含数据
        if self.queue.qsize() > 0:
            # 停止向缓冲区添加数据,如果有线程正在调用`add`方法添加数据，返回False
            if self.clear_buffer_add.tryAcquire():
                # 停止从缓冲区获取数据,如果有线程正在调用`get`方法获取数据,则返回False
                if self.clear_buffer_get.tryAcquire():
                    # 释放队列中的所有槽位
                    self.free_slots.release(self.queue.qsize())
                    # ?执行完之后,free_slots不是变为0了,上步已经恢复了free_slots了啊
                    self.free_slots.acquire(self.buffer_size)
                    # 把used_slots变为0
                    self.used_slots.acquire(self.queue.qsize())
                    # 清除缓冲区
                    for _ in range(self.queue.qsize()):
                        self.queue.get()
                    # ?为什么先获取后释放
                    self.free_slots.release(self.buffer_size)
                    # 恢复`get`方法的使用
                    self.clear_buffer_get.release()
                else:
                    return False
                # 恢复`add`方法的使用
                self.clear_buffer_add.release()
                return True
            else:
                return False
        else:
            return False

    def size(self):
        """
        返回队列大小
        """
        return self.queue.qsize()

    def maxsize(self):
        """
        返回缓存区最大值
        """
        return self.buffer_size

    def isfull(self):
        """
        判断队列是否满
        """
        return self.queue.qsize() == self.buffer_size

    def isempty(self):
        """
        判断队列是否为空
        """
        return self.queue.qsize() == 0


class MultiBufferManager(object):

    """
    用于同步来自不同摄像机的 capture_thread
    """

    def __init__(self, do_sync=True):
        self.sync_devices = set()  # 创建一个集合保存需要进行线程同步的设备id
        self.do_sync = do_sync  # 是否需要同步
        self.wc = QWaitCondition()
        self.mutex = QMutex()
        self.arrived = 0  # 计数器,
        self.buffer_maps = dict()  # 创建一个字典,键为设备id,值为设备对应的Buffer对象

    def bind_thread(self, thread, buffer_size, sync=True):
        """
        为该线程中的设备创建缓冲区,并绑定到MultiBufferManager对象,方便线程同步
        """
        self.create_buffer_for_device(thread.device_id, buffer_size, sync)
        # 将thread线程中的buffer_manager绑定到创建的MultiBufferManager对象,方便使用一个对象进行多个线程的管理
        thread.buffer_manager = self

    def create_buffer_for_device(self, device_id, buffer_size, sync=True):
        """
        为设备创建缓冲区,并把需要执行同步的设备id添加到sync_devices中
        """
        if sync:
            with QMutexLocker(self.mutex):
                self.sync_devices.add(device_id)

        self.buffer_maps[device_id] = Buffer(buffer_size)

    def get_device(self, device_id):
        """
        获取指定设备id的缓冲区对象
        """
        return self.buffer_maps[device_id]

    def remove_device(self, device_id):
        """
        将设备id对应的设备缓冲区删除,移除同步,并通知等待的线程
        """
        self.buffer_maps.pop(device_id)
        with QMutexLocker(self.mutex):
            if device_id in self.sync_devices:
                self.sync_devices.remove(device_id)
                self.wc.wakeAll()  # ?为什么在这里唤醒,好像是避免这个设备已经移除,其他线程还在等待同步

    def sync(self, device_id):
        # 只在指定的设备间执行同步
        self.mutex.lock()
        if device_id in self.sync_devices:
            # 每个线程处理完调用sync方法, arrived+1,
            self.arrived += 1
            # 当要同步的设备都调用了sync, 则执行同步, 唤醒所有线程
            if self.do_sync and self.arrived == len(self.sync_devices):
                self.wc.wakeAll()
            # 当还有设备没有处理完,进行等待
            else:
                self.wc.wait(self.mutex)
            # 同步完-1
            self.arrived -= 1
        self.mutex.unlock()

    def wake_all(self):
        """
        唤醒等待的线程
        """
        with QMutexLocker(self.mutex):
            self.wc.wakeAll()

    def set_sync(self, enable):
        """
        将同步标志置1
        """
        self.do_sync = enable

    def sync_enabled(self):
        """
        返回当前是否设置了线程同步
        """
        return self.do_sync

    def sync_enabled_for_device(self, device_id):
        """
        返回device_id是否设置了线程同步
        """
        return device_id in self.sync_devices

    def __contains__(self, device_id):
        """
        object类中的内置方法,在使用关键字`in`时被调用
        这里返回device_id是否在buffer_maps中
        """
        return device_id in self.buffer_maps

    def __str__(self):
        """
        在使用 str() 或者 print() 函数时被调用
        打印:
        类名, 是否启用同步, 创建了缓冲区的设备id, 执行同步的设备id
        """
        return (self.__class__.__name__ + ":\n" + \
                "sync: {}\n".format(self.do_sync) + \
                "devices: {}\n".format(tuple(self.buffer_maps.keys())) + \
                "sync enabled devices: {}".format(self.sync_devices))
