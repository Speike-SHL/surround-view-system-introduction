"""
~~~~~~~~~~~~~~~~~~~~~~~~
用于在实车上运行的最终版本
~~~~~~~~~~~~~~~~~~~~~~~~
"""
import os
import cv2
from surround_view import CaptureThread, CameraProcessingThread
from surround_view import FisheyeCameraModel, BirdView
from surround_view import MultiBufferManager, ProjectedImageBuffer
import surround_view.param_settings as settings

yamls_dir = os.path.join(os.getcwd(), "yaml")  # yaml文件的路径
camera_ids = [4, 3, 5, 6]  # ? 相机的设备id 为什么用4356
flip_methods = [0, 2, 0, 2]  # 0表示不变，2表示180度翻转
names = settings.camera_names  # 相机名称,["front", "back", "left", "right"]
cameras_files = [os.path.join(yamls_dir, name + ".yaml") for name in names]  # 相机参数的yaml文件
# 使用FisheyeCameraModel创建相机模型对象，传入相机参数yaml文件和相机名称
camera_models = [FisheyeCameraModel(camera_file, name) for camera_file, name in zip(cameras_files, names)]


"""
首先，程序使用CaptureThread类创建了一组线程，每个线程分别绑定到一个相机上，并设置缓冲区大小为8。
然后，程序创建了一个MultiBufferManager对象来管理这些缓冲区，并开始连接相机并启动线程。
接着，程序使用CameraProcessingThread类创建了一组线程，每个线程分别绑定到一个相机模型上，并将它们添加到一个ProjectedImageBuffer对象中。
然后，程序创建了一个BirdView对象，用于生成鸟瞰图，并通过load_weights_and_masks方法加载权重和掩码。
最后，程序使用cv2库显示鸟瞰图，并实时更新各个相机和鸟瞰图的帧率。
"""
def main():
    capture_tds = [CaptureThread(camera_id, flip_method) for camera_id, flip_method in zip(camera_ids, flip_methods)]
    capture_buffer_manager = MultiBufferManager()
    for td in capture_tds:
        capture_buffer_manager.bind_thread(td, buffer_size=8)
        if (td.connect_camera()):
            td.start()

    proc_buffer_manager = ProjectedImageBuffer()
    process_tds = [CameraProcessingThread(capture_buffer_manager,
                                          camera_id,
                                          camera_model)
                   for camera_id, camera_model in zip(camera_ids, camera_models)]
    for td in process_tds:
        proc_buffer_manager.bind_thread(td)
        td.start()

    birdview = BirdView(proc_buffer_manager)
    birdview.load_weights_and_masks("./weights.png", "./masks.png")
    birdview.start()
    while True:
        img = cv2.resize(birdview.get(), (300, 400))
        cv2.imshow("birdview", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        for td in capture_tds:
            print("camera {} fps: {}\n".format(td.device_id, td.stat_data.average_fps), end="\r")

        for td in process_tds:
            print("process {} fps: {}\n".format(td.device_id, td.stat_data.average_fps), end="\r")

        print("birdview fps: {}".format(birdview.stat_data.average_fps))

    for td in process_tds:
        td.stop()

    for td in capture_tds:
        td.stop()
        td.disconnect_camera()


if __name__ == "__main__":
    main()
