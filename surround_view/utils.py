import cv2
import numpy as np


def gstreamer_pipeline(cam_id=0, capture_width=960, capture_height=640, framerate=60, flip_method=2):
    """
    gstreamer_pipeline：多媒体管道
    使用 libgstreamer 打开 csi相机
    """
    # ?gstreamer管道的描述
    return ("nvarguscamerasrc sensor-id={} ! ".format(cam_id) + \
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (capture_width,
               capture_height,
               framerate,
               flip_method
            )
    )


def convert_binary_to_bool(mask):
    """
    将二进制图像(只有一个通道, 且像素为0或255)转换为二进制(所有像素都为0或1)
    """
    return (mask.astype(np.float) / 255.0).astype(np.int)  # 用astype转化数据类型,先转为float,除了后转为整型


def adjust_luminance(gray, factor):
    """
    通过一个factor调整灰度图的亮度
    """
    return np.minimum((gray * factor), 255).astype(np.uint8)  # ?怎么操作的


def get_mean_statistisc(gray, mask):
    """
    获取一个由 mask 矩阵定义区域内的灰度图像的总 values
    这个 mask 矩阵的值必须为0或1
    """
    # todo:  这个函数会造成报错，需要处理, 如果鸟瞰图线程启动不加演示，会导致数组维度不一样
    # print(f"gray.shape: {gray.shape}, mask.shape:{mask.shape}")
    return np.sum(gray * mask)


def mean_luminance_ratio(grayA, grayB, mask):
    """
    mask 矩阵定义区域内的总亮度比例
    """
    return get_mean_statistisc(grayA, mask) / get_mean_statistisc(grayB, mask)


def get_mask(img):
    """
    把一个图像转化为 mask 数组
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图像转化为灰度图
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)  # 转化为 mask 矩阵, 只有0或255的灰度
    return mask


def get_overlap_region_mask(imA, imB):
    """
    给定两张同样大小的图像,获取它们的重叠区域,然后把该区域转化为一个 mask 数组
    """
    overlap = cv2.bitwise_and(imA, imB)
    mask = get_mask(overlap)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
    return mask


def get_outmost_polygon_boundary(img):
    """
    给定一个描述了两个图像重叠区域内的 mask 图像,获取该区域的最外围图像
    """
    mask = get_mask(img)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
    cnts, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # 得到面积最大的轮廓
    C = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]

    # 多边形近似
    polygon = cv2.approxPolyDP(C, 0.009 * cv2.arcLength(C, True), True)

    return polygon


def get_weight_mask_matrix(imA, imB, dist_threshold=5):
    """
    得到平滑组合两个图像 A 和 B 的权重矩阵
    """
    overlapMask = get_overlap_region_mask(imA, imB)
    overlapMaskInv = cv2.bitwise_not(overlapMask)
    indices = np.where(overlapMask == 255)

    imA_diff = cv2.bitwise_and(imA, imA, mask=overlapMaskInv)
    imB_diff = cv2.bitwise_and(imB, imB, mask=overlapMaskInv)

    G = get_mask(imA).astype(np.float32) / 255.0

    polyA = get_outmost_polygon_boundary(imA_diff)
    polyB = get_outmost_polygon_boundary(imB_diff)

    for y, x in zip(*indices):
        distToB = cv2.pointPolygonTest(polyB, (float(x), float(y)), True)
        if distToB < dist_threshold:
            distToA = cv2.pointPolygonTest(polyA, (float(x), float(y)), True)
            distToB *= distToB
            distToA *= distToA
            G[y, x] = distToB / (distToA + distToB)

    return G, overlapMask


def make_white_balance(image):
    """
    根据图像通道的 means 调整其白平衡
    """
    B, G, R = cv2.split(image)
    m1 = np.mean(B)
    m2 = np.mean(G)
    m3 = np.mean(R)
    K = (m1 + m2 + m3) / 3
    c1 = K / m1
    c2 = K / m2
    c3 = K / m3
    B = adjust_luminance(B, c1)
    G = adjust_luminance(G, c2)
    R = adjust_luminance(R, c3)
    return cv2.merge((B, G, R))
