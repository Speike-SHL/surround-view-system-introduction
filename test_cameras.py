#!/usr/bin/env python3

import cv2
import time

# 自动搜索获取安装的相机列表用于初始化
def get_cam_lst(cam_lst=range(0, 24)):
    arr = []
    for iCam in cam_lst:
        cap = cv2.VideoCapture(iCam)
        if not cap.read()[0]:
            continue
        else:
            arr.append(iCam)

        cap.release()
    return arr

def show_cam_img(caps, cam_list):
    print("INFO: Press 'q' to quit! Press 's' to save a picture, 'n' to change to next camera device!")
    idx = 0
    while True:
        cap_device = caps[idx]
        ret, frame = cap_device.read()
        if ret:
            cv2.imshow('video', frame)
        else:
            print("ERROR: failed read frame!")

        # quit the test
        c = cv2.waitKey(1)
        if c == ord('q'):
            break

        # change to next camera device
        if c == ord('n'):
            idx += 1
            if idx >= len(caps):
                idx = 0
            continue

        # save the picture
        if c == ord('s'):
            if ret:
                name = 'video{0}_{1}.png'.format(cam_list[idx],
                            time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
                cv2.imwrite("E:\\front.png", frame) #保存可能有问题，自己指定路径和文件名
                print("saved file: %s!" %name)

    cv2.destroyAllWindows()

def init_caps(cam_list, resolution=(640,480)):
    caps = []
    for iCam in cam_list:
        cap = cv2.VideoCapture(iCam)
        cap.set(3, resolution[0])
        cap.set(4, resolution[1])
        caps.append(cap)

    return caps

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
        err_msg = "cannot find available video device in list: {0}!".format(video_list) +\
                    "\nPlease check the video devices in /dev/v4l/by-path/ folder!"

    if len(cam_list) < 1:
        print("ERROR: " + err_msg)
        return

    print("Available video device list is {}".format(cam_list))
    caps = init_caps(cam_list)
    show_cam_img(caps, cam_list)
    deinit_caps(caps)

if __name__ == "__main__":
    # 可以使用如下指令指定打开的相机号
    #show_cameras([2, 6, 10, 14])

    # 也可以自动搜索全部的相机,`q`退出，`s`保存一张图片，`n`切换下张照片
    show_cameras()
