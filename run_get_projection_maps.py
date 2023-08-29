"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Manually select points to get the projection matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import argparse
import os
import numpy as np
import cv2
from surround_view import FisheyeCameraModel, PointSelector, display_image
import surround_view.param_settings as settings

def callback(x):
    pass

def get_projection_map(camera_model, image):
    und_image = camera_model.undistort(image)
    name = camera_model.camera_name
    gui = PointSelector(und_image, title=name)
    dst_points = settings.project_keypoints[name]
    choice = gui.loop()
    if choice > 0:
        src = np.float32(gui.keypoints)
        dst = np.float32(dst_points)
        camera_model.project_matrix = cv2.getPerspectiveTransform(src, dst)
        proj_image = camera_model.project(und_image)

        ret = display_image("Bird's View", proj_image)
        if ret > 0:
            cv2.destroyAllWindows()
            return True
    cv2.destroyAllWindows()
    return False


def main():
    camera_names = settings.camera_names
    camera_files = [os.path.join(os.getcwd(), "yaml", camera_name + ".yaml") for camera_name in camera_names]
    image_files = [os.path.join(os.getcwd(), "images", camera_name + ".png") for camera_name in camera_names]
    images = [cv2.imread(image_file) for image_file in image_files]
    camera_models = [FisheyeCameraModel(camera_file, camera_name) for camera_file, camera_name in zip(camera_files, camera_names)]

    for camera_model, image in zip(camera_models, images):
        winName = f"{camera_model.camera_name} Camera  select <scale(0~200 --> 0~2)> <shift(0~400 --> -200~200)>"
        cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('scale_x', winName, 100, 200, callback)    # 映射到0~2
        cv2.createTrackbar('scale_y', winName, 100, 200, callback)    # 映射到0~2
        cv2.createTrackbar('shift_x', winName, 200, 400, callback)  # 映射到-200~200
        cv2.createTrackbar('shift_y', winName, 200, 400, callback)  # 映射到-200~200
        print(f"select {winName} , Press 'q' to exit the program and 'Enter' to confirm.")
        while True:
            scale = [cv2.getTrackbarPos('scale_x', winName)/100.0,
                     cv2.getTrackbarPos('scale_y', winName)/100.0]
            shift = [cv2.getTrackbarPos('shift_x', winName)-200,
                     cv2.getTrackbarPos('shift_y', winName)-200]
            camera_model.set_scale_and_shift(scale, shift)
            und_image = camera_model.undistort(image)
            cv2.imshow(winName, und_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("EXIT")
                cv2.destroyAllWindows()
                return False
            if key == 13:
                print(f"{camera_model.camera_name} Camera , scale={scale}, shift={shift}")
                cv2.destroyAllWindows()
                break
        success = get_projection_map(camera_model, image)
        if success:
            print("saving projection matrix to yaml")
            camera_model.save_data()
        else:
            print("failed to compute the projection map")


if __name__ == "__main__":
    main()
