#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np

# 关于相机曝光问题：首先要在官方给出的相机设置软件中，把相机曝光设置成手动
# 因为鱼眼相机视场大，进光多，所以白天要把曝光设置的很低。只有相机曝光设置
# 成了手动，opencv中才能进行更改。同时,opencv还可以写一个函数，自动识别
# 当前捕获的图片亮度，自动调节曝光。
# 不过目前的问题是，opencv曝光设置成功了，但是读取的曝光参数一直是-6


# def isLight(src):     # TODO:写一个函数，通过计算图片的亮度，自动调节曝光
#   gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#   total = gray.size
#   sum_val = np.sum(gray.astype(float) - 128)
#   avg = sum_val / total
#   ls = np.zeros(256, dtype=int)
#
#   for i in range(gray.shape[0]):
#     for j in range(gray.shape[1]):
#       x = int(gray[i, j])
#       ls[x] += 1
#
#   total = np.sum(np.abs(np.arange(256) - 128 - avg) * ls)
#   mean = total / total
#   cast = abs(avg / mean)
#
#   print("亮度异常值:", cast)
#
#   if cast > 1:
#     if avg > 0:
#       print("亮度异常 过亮", avg)
#       return 1
#     else:
#       print("亮度异常 过暗", avg)
#       return -1
#   else:
#     print("normal")
#     return 0

# 自动搜索获取安装的相机列表
def get_cam_lst(cam_lst=range(0, 24)):
    arr = []
    for iCam in cam_lst:
        # cap = cv2.VideoCapture(iCam, cv2.CAP_DSHOW)
        cap = cv2.VideoCapture(iCam)
        ret, frame = cap.read()
        if ret:
            print(f"设备{iCam}已打开！")
            cv2.imshow(f"video{iCam}", frame)
            cv2.waitKey(500)
            arr.append(iCam)
        cap.release()
    cv2.destroyAllWindows()
    return arr


def show_cam_img(caps, cam_list):
    print(
        "INFO: Press 'q' to quit! Press 's' to save a picture, 'n' to change to next camera device!"
    )
    idx = 0
    error_count = 0
    while True:
        cap_device = caps[idx]
        time.sleep(0.1)
        ret, frame = cap_device.read()
        if ret:
            # isLight(frame)
            cv2.imshow("video", frame)
            # print(f"相机{idx} 曝光{cap_device.get(cv2.CAP_PROP_EXPOSURE)}")
            # print(f"相机{idx} 亮度{cap_device.get(cv2.CAP_PROP_BRIGHTNESS)}")
        else:
            print("ERROR: failed read frame!")
            time.sleep(1)
            error_count += 1

        if error_count > 20:
            print("ERROR: too many failed read frame!")
            break

        # quit the test
        c = cv2.waitKey(1)
        if c == ord("q"):
            break

        # change to next camera device
        if c == ord("n"):
            error_count = 0
            idx += 1
            if idx >= len(caps):
                idx = 0
            continue

        # save the picture
        if c == ord("s"):
            if ret:
                name = "video{0}_{1}.png".format(
                    cam_list[idx], time.strftime(
                        "%Y-%m-%d_%H_%M_%S", time.localtime())
                )
                image_name = os.path.join(os.getcwd(), "images", name)
                cv2.imwrite(image_name, frame)  # 保存可能有问题，自己指定路径和文件名
                print(f"save file to {image_name}")
    
    cv2.destroyAllWindows()


# 初始化相机，设置一些参数
def init_caps(cam_list, resolution=(640, 480)):
    caps = []
    for iCam in cam_list:
        # cap = cv2.VideoCapture(iCam, cv2.CAP_DSHOW)
        cap = cv2.VideoCapture(iCam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_EXPOSURE, -1)          # TODO: 在相机软件中设置曝光为手动后,每次在这里手动调节曝光
        # cap.set(cv2.CAP_PROP_BRIGHTNESS, -50)
        # cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        caps.append(cap)

    for cap in caps:
        print(f"设备{cap}已打开！，分辨率为{cap.get(3)}*{cap.get(4)}, 帧率为{cap.get(5)}, "
              f"曝光为{cap.get(cv2.CAP_PROP_EXPOSURE)}, 亮度为{cap.get(cv2.CAP_PROP_BRIGHTNESS)}")

    return caps



# 最后销毁相机对象
def deinit_caps(cap_list):
    for cap in cap_list:
        cap.release()


def show_cameras(video_list=None):
    if video_list == None:
        print("Start to search all available camera devices, please wait... ")
        cam_list = get_cam_lst()
        err_msg = "cannot find any video device!"
    else:
        cam_list = get_cam_lst(video_list)
        err_msg = (
            "cannot find available video device in list: {0}!".format(
                video_list)
            + "\nPlease check the video devices in /dev/v4l/by-path/ folder!"
        )

    if len(cam_list) < 1:
        print("ERROR: " + err_msg)
        return

    print("Available video device list is {}".format(cam_list))
    caps = init_caps(cam_list)
    show_cam_img(caps, cam_list)
    deinit_caps(caps)


if __name__ == "__main__":
    # 可以使用如下指令指定打开的相机号
    # show_cameras([2, 6, 10, 14])

    # 也可以自动搜索全部的相机,`q`退出，`s`保存一张图片，`n`切换下张照片
    show_cameras()
