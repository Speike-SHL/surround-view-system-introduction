#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   automatic_parking.py
@Time    :   2023/08/25 11:42:06
@Author  :   Speike 
@Contact :   shao-haoluo@foxmail.com
@Desc    :   None
"""
import cv2
import time


def get_cam_list(cam_list=range(0, 1000)):
    arr = []
    for iCam in cam_list:
        cap = cv2.VideoCapture(iCam)
        if not cap.read()[0]:
            continue
        else:
            arr.append(iCam)

        cap.release()
    return arr


def show_cam_img(cam_list, resolution=(640, 480)):
    caps = []
    for iCam in cam_list:
        cap = cv2.VideoCapture(iCam)
        cap.set(6, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
        cap.set(3, resolution[0])
        cap.set(4, resolution[1])
        caps.append(cap)
    while True:
        for i in range(len(cam_list)):
            # cap = cv2.VideoCapture(cam_list[i])
            # cap.set(3, resolution[0])
            # cap.set(4, resolution[1])
            # ret, frame = cap.read()
            # cap.release()
            if caps[i].isOpened():
                print(f"video{cam_list[i]} is opened!")
            else:
                print(f"video{cam_list[i]} is not opened!")
            ret, frame = caps[i].read()
            if ret:
                cv2.imshow(f"video{cam_list[i]}", frame)
            else:
                print(f"ERROR: failed read frame from video{cam_list[i]}!")
            cv2.waitKey(1)


def search_cameras(video_list=None):
    if video_list == None:
        print("开始寻找所有可用的相机设备...")
        cam_list = get_cam_list()
        print(f"找到{len(cam_list)}个相机设备，分别是：{cam_list}")
        show_cam_img(cam_list)
    else:
        print("您指定了相机设备列表，开始寻找...")
        cam_list = get_cam_list(video_list)
        # 对比video_list和cam_list，找出不匹配的相机设备
        diff = list(set(video_list).difference(set(cam_list)))
        if len(diff) == 0:
            print(f"成功找到所有指定的相机设备，分别是：{cam_list}")
            show_cam_img(cam_list)
        else:
            print(f"找到{len(cam_list)}个相机设备，分别是：{cam_list}")
            print(f"未找到{len(diff)}个相机设备，分别是：{diff}")
            show_cam_img(cam_list)


if __name__ == "__main__":
    search_cameras()
