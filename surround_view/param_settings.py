"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
用于设置投影区域的各参数，见doc中的解释
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import os
import cv2

camera_names = ["front", "back", "left", "right"]

# --------------------------------------------------------------------
# # 这两个参数决定了在鸟瞰图中向标定板的外侧看多远。这两个值越大，鸟瞰图看的范围就越大，相应地远处的物体被投影后的形变也越严重，所以应酌情选择
# shift_w = 300
# shift_h = 300
#
# # size of the gap between the calibration pattern and the car in horizontal and vertical directions
# # 标定板内侧边缘与车辆左右两侧的距离，标定板内侧边缘与车辆前后方的距离
# inn_shift_w = 20
# inn_shift_h = 50
#
# # 拼接后图像的总宽和高
# total_w = 600 + 2 * shift_w
# total_h = 1000 + 2 * shift_h
#
# # 整个鸟瞰图左上角为原点，车辆四个角在鸟瞰图中的位置，(xl,yt),(xr,yt),(xl,yb),(xr,yb)
# xl = shift_w + 180 + inn_shift_w
# xr = total_w - xl
# yt = shift_h + 200 + inn_shift_h
# yb = total_h - yt
shift_w = 100
shift_h = 100
inn_shift_w = 2
inn_shift_h = 53
total_w = 160 + 2 * shift_w
total_h = 273 + 2 * shift_h
xl = shift_w + 47 + inn_shift_w
xr = total_w - xl
yt = shift_h + 47 + inn_shift_h
yb = total_h - yt
# --------------------------------------------------------------------

# 每个相机的射影后图像的宽，长
project_shapes = {
    "front": (total_w, yt),
    "back": (total_w, yt),
    "left": (total_h, xl),
    "right": (total_h, xl)
}

# 计算射影矩阵时选取的关键点，在运行get_projection_map.py时，需要按顺序在图中点击下面的点
# project_keypoints = {
#     "front": [(shift_w + 120, shift_h),
#               (shift_w + 480, shift_h),
#               (shift_w + 120, shift_h + 160),
#               (shift_w + 480, shift_h + 160)],
#
#     "back":  [(shift_w + 120, shift_h),
#               (shift_w + 480, shift_h),
#               (shift_w + 120, shift_h + 160),
#               (shift_w + 480, shift_h + 160)],
#
#     "left":  [(shift_h + 280, shift_w),
#               (shift_h + 840, shift_w),
#               (shift_h + 280, shift_w + 160),
#               (shift_h + 840, shift_w + 160)],
#
#     "right": [(shift_h + 160, shift_w),
#               (shift_h + 720, shift_w),
#               (shift_h + 160, shift_w + 160),
#               (shift_h + 720, shift_w + 160)]
# }
project_keypoints = {
    "front": [(shift_w + 23, shift_h),
              (shift_w + 136, shift_h),
              (shift_w + 23, shift_h + 68),
              (shift_w + 136, shift_h + 68)],

    "back": [(shift_w + 7.7, shift_h),
             (shift_w + 152, shift_h),
             (shift_w + 7.7, shift_h + 89.8),
             (shift_w + 152, shift_h + 89.8)],

    "left": [(shift_h + 39, shift_w),
             (shift_h + 233, shift_w),
             (shift_h + 39, shift_w + 39),
             (shift_h + 233, shift_w + 39)],

    "right": [(shift_h + 39, shift_w),
              (shift_h + 233, shift_w),
              (shift_h + 39, shift_w + 39),
              (shift_h + 233, shift_w + 39)]
}

# 读取images文件夹下car.png这个图像，并把它大小改为设置的car的大小，即car四个点坐标计算的长宽
car_image = cv2.imread(os.path.join(os.getcwd(), "images", "car.png"))
car_image = cv2.resize(car_image, (xr - xl, int(1.4 * (yb - yt))))
# car_image = cv2.resize(car_image, (xr - xl,yb -yt))

# ----------------------------用于保存图像---------------------------------
# -------------------不要在这里更改下面的值,主程序会修改-----------------------
WORK_PATH = os.getcwd()
SAVE_RAW = 0                    # 是否保存原始图像,process_thread.py
SAVE_UNDISTORTED = 0            # 是否保存去畸变的图像,process_thread.py
SAVE_PROJECTION = 0             # 是否保存射影变换后的图像,process_thread.py
SAVE_BRIDVIEW_PROCESS = False   # 是否保存鸟瞰图处理的对比结果
IS_RECORDING = False            # 是否保存鸟瞰图视频
FPS = 30                        # 鸟瞰图视频的帧率, 可根据实际处理速度修改
WIDTH = 600                     # 鸟瞰图视频的宽
HEIGHT = 800                    # 鸟瞰图视频的高