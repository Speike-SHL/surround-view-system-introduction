"""
~~~~~~~~~~~~~~~~~~~~~~~~~~
Fisheye Camera calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage:
    python calibrate_camera.py \
        -i 0 \
        -grid 9x6 \
        -o fisheye.yaml \
        -framestep 20 \
        --resolution 640x480
        --fisheye
"""
import argparse
import os
import numpy as np
import cv2
from surround_view import CaptureThread, MultiBufferManager
import surround_view.utils as utils


# 将把相机参数文件保存到yaml目录
TARGET_DIR = os.path.join(os.getcwd(), "yaml")

# 默认的文件名为camera_params.yaml
DEFAULT_PARAM_FILE = os.path.join(TARGET_DIR, "camera_params.yaml")


def main():
    # argparse模块是python用来读取命令行输入参数的模块
    parser = argparse.ArgumentParser()

    # 输入视频流
    parser.add_argument("-i", "--input", type=int, default=0,
                        help="input camera device")

    # 棋盘格大小
    parser.add_argument("-grid", "--grid", default="9x6",
                        help="size of the calibrate grid pattern")

    parser.add_argument("-r", "--resolution", default="640x480",
                        help="resolution of the camera image")

    # 每framestep读取一帧进行标定
    parser.add_argument("-framestep", type=int, default=20,
                        help="use every nth frame in the video")

    parser.add_argument("-o", "--output", default=DEFAULT_PARAM_FILE,
                        help="path to output yaml file")

    parser.add_argument("-fisheye", "--fisheye", action="store_true",
                        help="set true if this is a fisheye camera")

    parser.add_argument("-flip", "--flip", default=0, type=int,
                        help="flip method of the camera")

    parser.add_argument("--no_gst", action="store_true",
                        help="set true if not use gstreamer for the camera capture")

    args = parser.parse_args()

    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)

    # 设置标定图像左上角的提示字体
    text1 = "press c to calibrate"
    text2 = "press q to quit"
    text3 = "device: {}".format(args.input)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体格式
    fontscale = 0.6

    # 获取分辨率
    resolution_str = args.resolution.split("x")
    W = int(resolution_str[0])
    H = int(resolution_str[1])
    """
    获取网格大小，并创建三维的全0数组grid_points储存三维点坐标
    第三行理解如下：
    假设用户在命令行中输入了以下参数：--resolution 640x480 -grid 6x4
    W = 640
    H = 480
    grid_size = (6, 4)
    grid_points = array([[[0., 0., 0.],
                          [1., 0., 0.],
                          [2., 0., 0.],
                          [3., 0., 0.],
                          [4., 0., 0.],
                          [5., 0., 0.],
                          [0., 1., 0.],
                          [1., 1., 0.],
                          [2., 1., 0.],
                          [3., 1., 0.],
                          [4., 1., 0.],
                          [5., 1., 0.],
                          [0., 2., 0.],
                          [1., 2., 0.],
                          [2., 2., 0.],
                          [3., 2., 0.],
                          [4., 2., 0.],
                          [5., 2., 0.],
                          [0., 3., 0.],
                          [1., 3., 0.],
                          [2., 3., 0.],
                          [3., 3., 0.],
                          [4., 3., 0.],
                          [5., 3., 0.]]], dtype=float32)
    这里grid_points是一个形状为(1, 24, 3)的numpy数组，它表示了一个6x4的棋盘格子中每个格子的3D空间坐标。
    第一维是1表示只有一张棋盘格子图像，第二维是24表示一共有24个格子，第三维是3表示坐标是三维(x, y, z)。
    grid_points数组的前两列(x,y)分别是这个二维棋盘格子中的每个格子的行坐标和列坐标，第三列(z轴)全部为0，
    表示所有格子都在棋盘格子的平面上。
    """
    grid_size = tuple(int(x) for x in args.grid.split("x"))
    grid_points = np.zeros((1, np.prod(grid_size), 3), np.float32)
    grid_points[0, :, :2] = np.indices(grid_size).T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    device = args.input  # 设备号
    cap_thread = CaptureThread(device_id=device,
                               flip_method=args.flip,
                               resolution=(W, H),
                               use_gst=not args.no_gst,
                               )
    buffer_manager = MultiBufferManager()
    buffer_manager.bind_thread(cap_thread, buffer_size=8)
    if cap_thread.connect_camera():
        cap_thread.start()
    else:
        print("cannot open device")
        return

    quit = False
    do_calib = False
    i = -1
    while True:
        i += 1
        img = buffer_manager.get_device(device).get().image
        if i % args.framestep != 0:
            continue

        print("searching for chessboard corners in frame " + str(i) + "...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 寻找标定板角点
        found, corners = cv2.findChessboardCorners(
            gray,
            grid_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FILTER_QUADS
            # cv2.CALIB_CB_FAST_CHECK
        )
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            imgpoints.append(corners)
            objpoints.append(grid_points)
            print(f"OK! you have found {len(objpoints)}, left {12-len(objpoints)}.",)
            # 在标定板上画圆显示
            cv2.drawChessboardCorners(img, grid_size, corners, found)

        # 在标定图像左上角显示文字
        cv2.putText(img, text1, (20, 70), font, fontscale, (255, 200, 0), 2)
        cv2.putText(img, text2, (20, 110), font, fontscale, (255, 200, 0), 2)
        cv2.putText(img, text3, (20, 30), font, fontscale, (255, 200, 0), 2)
        cv2.imshow("corners", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            print("\nPerforming calibration...\n")
            N_OK = len(objpoints)
            if N_OK < 12:
                print("Less than 12 corners (%d) detected, calibration failed" %(N_OK))
                continue
            else:
                do_calib = True
                break

        elif key == ord("q"):
            quit = True
            break

    if quit:
        cap_thread.stop()
        cap_thread.disconnect_camera()
        cv2.destroyAllWindows()

    # 做标定
    if do_calib:
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
        calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                             cv2.fisheye.CALIB_CHECK_COND +
                             cv2.fisheye.CALIB_FIX_SKEW)

        if args.fisheye:
            ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                (W, H),
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        else:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints,
                imgpoints,
                (W, H),
                None,
                None)

        if ret:
            fs = cv2.FileStorage(args.output, cv2.FILE_STORAGE_WRITE)
            fs.write("resolution", np.int32([W, H]))
            fs.write("camera_matrix", K)
            fs.write("dist_coeffs", D)
            fs.release()
            print("successfully saved camera data")
            cv2.putText(img, "Success!", (220, 240), font, 2, (0, 0, 255), 2)

        else:
            cv2.putText(img, "Failed!", (220, 240), font, 2, (0, 0, 255), 2)

        cv2.imshow("corners", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
