#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   automatic_parking.py
@Time    :   2023/08/25 11:42:06
@Author  :   Speike 
@Contact :   shao-haoluo@foxmail.com
@Desc    :   None
"""
import os
import cv2
import time
from surround_view import CaptureThread, CameraProcessingThread
from surround_view import FisheyeCameraModel, BirdView
from surround_view import MultiBufferManager, ProjectedImageBuffer
import surround_view.param_settings as settings

import torch
import pprint
import numpy as np
from pathlib import Path
from psdet.utils.config import get_config
from psdet.utils.common import get_logger
from psdet.models.builder import build_model

yaml_dirs = os.path.join(os.getcwd(), "yaml")  # yaml文件的路径
camera_ids = [701, 702, 703, 704]  # ? 相机的设备id 为什么用4356,是从test_cameras.py中读出来的，每次都有不同
flip_methods = [0, 2, 0, 2]  # 0表示不变，2表示180度翻转
names = settings.camera_names  # 相机名称,["front", "back", "left", "right"]
cameras_files = [os.path.join(yaml_dirs, name + ".yaml") for name in names]  # 相机参数的yaml文件
# 使用FisheyeCameraModel创建4个相机模型对象，传入相机参数yaml文件和相机名称
camera_models = [FisheyeCameraModel(camera_file, name) for camera_file, name in zip(cameras_files, names)]

"""
首先，程序使用CaptureThread类创建了一组4个线程，每个线程分别绑定到一个相机上，并设置缓冲区大小为8。
然后，程序创建了一个MultiBufferManager对象来管理这些缓冲区，并开始连接相机并启动线程。
接着，程序使用CameraProcessingThread类创建了一组4个线程，每个线程分别绑定到一个相机模型上，并将它们添加到一个ProjectedImageBuffer对象中。
然后，程序创建了一个BirdView线程1个，用于生成鸟瞰图，并通过load_weights_and_masks方法加载权重和掩码。
最后，程序使用cv2库显示鸟瞰图，并实时更新各个相机和鸟瞰图的帧率。
"""


def main():
    out = None  # 鸟瞰图视频流对象
    # 创建4个相机捕获线程
    capture_tds = [CaptureThread(camera_id, flip_method) for camera_id, flip_method in zip(camera_ids, flip_methods)]
    # 创建相机捕获线程的线程管理对象
    capture_buffer_manager = MultiBufferManager()
    # 把4个相机捕获线程绑定到线程管理对象上
    for td in capture_tds:
        capture_buffer_manager.bind_thread(td, buffer_size=8)
        if td.connect_camera():  # 连接相机
            td.start()  # 开启相机捕获线程
            print("开启一个相机捕获线程")
        else:
            print("相机未连接")
            return
    print("--------------相机捕获线程全部启动成功---------------")
    # 创建图像处理线程的线程管理对象
    proc_buffer_manager = ProjectedImageBuffer()
    # 创建4个图像处理线程
    process_tds = [CameraProcessingThread(capture_buffer_manager,
                                          camera_id,
                                          camera_model)
                   for camera_id, camera_model in zip(camera_ids, camera_models)]
    # 把4个图像处理线程绑定到线程管理对象上
    for td in process_tds:
        proc_buffer_manager.bind_thread(td)
        td.start()  # 开启图像处理线程
        time.sleep(0.2)
        print("开启一个图像处理线程")
    print("--------------图像处理线程全部启动成功---------------")
    # 创建鸟瞰图处理线程
    time.sleep(1)  # todo:不知道为什么，如果此处不延时开启线程，会导致矩阵相乘出错，是因为图像处理线程还没来得及处理？
    birdview = BirdView(proc_buffer_manager)
    # 加载权重矩阵和mask矩阵
    birdview.load_weights_and_masks("./weights.png", "./masks.png")
    birdview.start()  # 开启鸟瞰图线程
    print("----------------鸟瞰图线程启动成功------------------")
    while True:
        img = cv2.resize(birdview.get(), (settings.WIDTH, settings.HEIGHT))

        # ---------------NEW-----------------
        # 用模型检测车位
        img, parkings_points = get_parkingslot(img)

        cv2.imshow("birdview", img)  # 显示鸟瞰图

        key = cv2.waitKey(1) & 0xFF  # 每一毫秒检查一下用户是否按键
        if key == ord("q"):  # 用户按下“q”终止程序运行
            break
        elif key == ord('i'):  # 用户按下"i"键保存图像
            print("saving image----")
            saveImg()
        elif key == ord('r'):  # 开始录制拼接后的鸟瞰图
            print("recording video---")
            settings.IS_RECORDING = True
            out = cv2.VideoWriter(f"{settings.WORK_PATH}/paper_need_img/birdview.mp4", cv2.VideoWriter_fourcc(*'MP4V'),
                                  settings.FPS, (settings.WIDTH, settings.HEIGHT))
        elif key == ord('s'):  # 按下”s“键停止录制
            print("stop recording video---")
            settings.IS_RECORDING = False
            if out is not None:
                out.release()
                out = None

        if settings.IS_RECORDING:
            if out is not None:
                out.write(img)

        # for td in capture_tds:  # 显示相机捕获线程的设备id和对应平均帧率
        #     print("camera {} fps: {}\n".format(td.device_id, td.stat_data.average_fps), end="\r")
        #
        # for td in process_tds:  # 显示图像处理线程的设备id和对应平均帧率
        #     print("process {} fps: {}\n".format(td.device_id, td.stat_data.average_fps), end="\r")
        # 显示鸟瞰图线程的平均帧率
        print("birdview fps: {}".format(birdview.stat_data.average_fps))

    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    for td in process_tds:  # 终止循环时结束线程
        td.stop()

    for td in capture_tds:  # 终止循环时结束线程并断开相机
        td.stop()
        td.disconnect_camera()


# 用于保存论文中所需图片
def saveImg():
    settings.SAVE_RAW = 4
    settings.SAVE_UNDISTORTED = 4
    settings.SAVE_PROJECTION = 4
    settings.SAVE_BRIDVIEW_PROCESS = True


def draw_parking_slot(image, pred_dicts):
    """
    绘制车位检测结果
    :param image: 原始图像
    :param pred_dicts: 模型输出的预测结果
    :return: 绘制了车位检测结果的图像以及字典类型的多车位的四个点坐标
    """
    slots_pred = pred_dicts['slots_pred']

    width = 512  # 图像的宽 进模型前被resize成了512
    height = 512  # 图像的高 进模型前被resize成了512
    VSLOT_MIN_DIST = 0.044771278151623496
    VSLOT_MAX_DIST = 0.1099427457599304
    HSLOT_MIN_DIST = 0.15057789144568634
    HSLOT_MAX_DIST = 0.44449496544202816

    SHORT_SEPARATOR_LENGTH = 0.199519231
    LONG_SEPARATOR_LENGTH = 0.46875
    junctions = []
    parkings_points = {}  # ---------------NEW-----------------
    for j in range(len(slots_pred[0])):
        position = slots_pred[0][j][1]
        # 由模型输出的position计算出车位入口线的两个端点p0,p1
        p0_x = width * position[0] - 0.5
        p0_y = height * position[1] - 0.5
        p1_x = width * position[2] - 0.5
        p1_y = height * position[3] - 0.5
        vec = np.array([p1_x - p0_x, p1_y - p0_y])
        vec = vec / np.linalg.norm(vec)  # p0 和 p1 之间的方向向量, 代表车库的入口线方向
        # 入口线长度的平方 (p0_x - p1_x)^2 + (p0_y - p1_y)^2
        distance = (position[0] - position[2]) ** 2 + (position[1] - position[3]) ** 2

        # 根据入口线长度判断是纵向车位还是侧方车位，从而确定侧线的长度
        if VSLOT_MIN_DIST <= distance <= VSLOT_MAX_DIST:
            separating_length = LONG_SEPARATOR_LENGTH  # separating_length 代表车位侧线的长度
        else:
            separating_length = SHORT_SEPARATOR_LENGTH

        # 由入口点p0,p1和方向向量vec以及侧线长度separating_length计算出车位底部两个点p2,p3
        p2_x = p0_x + height * separating_length * vec[1]
        p2_y = p0_y - width * separating_length * vec[0]
        p3_x = p1_x + height * separating_length * vec[1]
        p3_y = p1_y - width * separating_length * vec[0]
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        p3_x = int(round(p3_x))
        p3_y = int(round(p3_y))
        cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (255, 0, 0), 2)
        cv2.line(image, (p0_x, p0_y), (p2_x, p2_y), (255, 0, 0), 2)
        cv2.line(image, (p1_x, p1_y), (p3_x, p3_y), (255, 0, 0), 2)

        # cv2.circle(image, (p0_x, p0_y), 3,  (0, 0, 255), 4)
        junctions.append((p0_x, p0_y))
        junctions.append((p1_x, p1_y))

        # ---------------NEW-----------------
        parkings_points[j] = [(p0_x, p0_y), (p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y)]

    for junction in junctions:
        cv2.circle(image, junction, 3, (0, 0, 255), 4)

    return image, parkings_points


def get_parkingslot(image):
    start_time = time.time()
    with torch.no_grad():
        image0 = cv2.resize(image, (512, 512))
        image = image0 / 255.
        data_dict = {'image': torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).cuda()}
        pred_dicts, ret_dict = model(data_dict)
        print(f"模型车位检测耗时：{time.time() - start_time}")
        # 绘制车位检测结果
        start_time = time.time()
        image, parkings_points = draw_parking_slot(image0, pred_dicts)
        print(f"绘制车位检测结果耗时：{time.time() - start_time}")
    return image, parkings_points


if __name__ == "__main__":
    cfg = get_config()
    logger = get_logger(cfg.log_dir, cfg.tag)
    logger.info(pprint.pformat(cfg))

    # 加载模型
    model = build_model(cfg.model)
    logger.info(model)
    model.load_params_from_file(filename=cfg.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    main()

## 执行时要指定模型和配置文件
# python automatic_parking.py
#                       -c config/ps_gat.yaml
#                       -m model1_checkpoint_epoch_200.pth
