# 导入必要的包
import numpy as np
import imutils
import cv2


class SingleMotionDetector:
    def __init__(self, accumWeight=0.5):
        # 存储累计重量因子
        self.accumWeight = accumWeight

        # 初始化背景模型
        self.bg = None

    def update(self, image):
        # 如果背景模型为空，则初始化它
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # 通过累积加权平均值来更新背景模型

        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def detect(self, image, tVal=25):
        # 计算背景模型与传入图像之间的绝对差值，然后对差值图像进行阈值处理
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]

        # 进行一系列的腐蚀和膨胀来去除小水珠
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 在阈值图像中寻找轮廓，初始化运动的最小和最大边界框区域
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        # 如果没有找到等高线，则返回None
        if len(cnts) == 0:
            return None

        # 否则，在等高线上循环
        for c in cnts:
            # 计算轮廓的边界框，并使用它来更新最小和最大边界框区域
            (x, y, w, h) = cv2.boundingRect(c)
            (minX, minY) = (min(minX, x), min(minY, y))
            (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

        # 否则，返回带边界框的阈值图像的元组
        return (thresh, (minX, minY, maxX, maxY))