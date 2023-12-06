import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from calib import *
import torch
import time
import torch.nn as nn
from pathlib import Path

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, dnn=False):
        from models.experimental import attempt_load #, attempt_download # scoped to avoid circular import
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix = Path(w).suffix.lower()
        suffixes = ['.pt', '.torchscript', '.onnx', '.engine', '.tflite', '.pb', '', '.mlmodel']
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        self.__dict__.update(locals())  # assign all variables to self
    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        y = self.model(im, augment=augment, visualize=visualize)
        return y if val else y[0]

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        if isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
            im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
            self.forward(im)  # warmup


def read_images(dir1, dir2, gray=True):
    img1 = cv.imread(dir1)
    img2 = cv.imread(dir2)
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    return img1, img2, img1_gray, img2_gray


def match_BM(stereo, img1, img2):
    """
    :param img1, img2: (height, width) grayscale
    """
    
    disparity_BM = stereo.compute(img1, img2)
    return disparity_BM


def match_SGBM(stereo, img1, img2):
    """
    :param img1, img2: (height, width, color)
    """
    disparity_SGBM = stereo.compute(img1, img2)
    return disparity_SGBM


def match_ROI_BM(img1, img2, x1,x2,y1,y2):
    """
    :param roi: (obj_y_top, obj_y_bottom)
    """
    stereo = cv.StereoBM_create(
        numDisparities=256,
        blockSize=17
    )
    stereo.setPreFilterSize(9)
    stereo.setPreFilterCap(31)
    stereo.setTextureThreshold(10)
    stereo.setUniquenessRatio(12)
    stereo.setSpeckleRange(10)
    stereo.setSpeckleWindowSize(150)
    disparity_BM = stereo.compute(img1[y1:y2, x1:], img2[y1:y2,x1:])
    return disparity_BM


def match_ROI_SGBM(img1, img2, roi=None):
    """
    :param roi: (obj_y_top, obj_y_bottom)
    """
    blockSize = 11
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=384,
        blockSize=blockSize,
        uniquenessRatio=12,
        speckleWindowSize=150,
        speckleRange=10,
        disp12MaxDiff=0,
        P1=8 * 1 * blockSize * blockSize,
        P2=32 * 1 * blockSize * blockSize,
        mode=0
    )
    if roi is not None:
        disparity_SGBM = stereo.compute(img1[roi[0]:roi[1], :, :], img2[roi[0]:roi[1], :, :])
    else:
        disparity_SGBM = stereo.compute(img1, img2)
    return disparity_SGBM

def computeDistance(stereo, img1, img2, x1,x2,y1,y2, M=126.6, mode="median"):
    disparity_BM = stereo.compute(img1[y1:y2, x1:], img2[y1:y2,x1:])
    
    val = disparity_BM[:,:(x2-x1)] #disparity_BM[(y2-y1)//:3*(y2-y1)//4,(x2-x1)//4:3*(x2-x1)//4]
    print(val.shape, end=' ')
    val = val[val>0]/16
    print(val.shape)
    if mode == "median":
        dist = M / np.median(val)
    elif mode == "mean":
        dist = M / np.mean(val)
    elif mode == "thresh":
        dist = M / np.mean(val[val >= np.median(val)])
    return dist*1000
    
    
def fastBM(leftIm, rightIm, windowSize, maxDisp, x1,y1, x2,y2 ,centerRatio = 0.5,stride = 2, device='cuda'):
    # left/Right Im: Original size frame tensor
    # WindowSize: BM window
    # x1,y1: left top corner's coord. in original size
    # x2,y2: Right bottom corner's coord. in original size
    # stride: each of how many pixels computed,
    # center ratio: 0.5 computes the area x_size*0.5 X y_size*0.5
    # device = 'cpu' if cuda is not available
    with torch.no_grad():
        t = time.time()
        xLeftLim = int(x1+(x2-x1)*centerRatio/2)
        xRightLim = int(x2-(x2-x1)*centerRatio/2)
        yUpLim = int(y1+(y2-y1)*centerRatio/2)
        yDownLim = int(y2-(y2-y1)*centerRatio/2)
        leftIm = leftIm.to(device)
        rightIm = rightIm.to(device)
    
        allPixelDists = torch.zeros(len(range(yUpLim,yDownLim,stride)),len(range(xLeftLim,xRightLim,stride))).to(device)
        for xx,xId in enumerate(range(xLeftLim,xRightLim,stride)):
            for yy,yId in enumerate(range(yUpLim,yDownLim,stride)):
                filterIm = leftIm[:,max(0,yId-windowSize//2):min(yId+windowSize//2,leftIm.shape[1]-1),
                                  max(0,xId-windowSize//2):min(xId+windowSize//2,leftIm.shape[2]-1)]
                searchArea = rightIm[:,max(0,yId-windowSize//2):min(yId+windowSize//2,rightIm.shape[1]-1),
                                     max(xId-maxDisp,0):min(xId+windowSize//2,rightIm.shape[2]-1)].unfold(2,filterIm.shape[2],1).permute(2,0,1,3)
                diff = torch.sum(abs(searchArea-filterIm),dim=(1,2,3))
                allPixelDists[yy,xx] = searchArea.shape[0]-torch.argmin(diff)
        print(time.time()-t)
        return float(torch.median(allPixelDists))

def compute_distance(disparity_map, obj_pos, r=1, M=126.6, mode="median"):
    """
    :param disparity_map: (height, width)
    :param obj_pos: (x, y, width, height)
    """
    obj_pos = (int(obj_pos[0] + obj_pos[2] * (1 - r) / 2), int(obj_pos[1] + obj_pos[3] * (1 - r) / 2),
               int(obj_pos[2] * (1 + r) / 2), int(obj_pos[3] * (1 + r) / 2))
    obj_rect = disparity_map[obj_pos[1]:obj_pos[1] + obj_pos[3], obj_pos[0]:obj_pos[0] + obj_pos[2]]

    if mode == "median":
        dist = M / np.median(obj_rect)
    elif mode == "mean":
        dist = M / np.mean(obj_rect)
    elif mode == "thresh":
        dist = M / np.mean(obj_rect[obj_rect >= np.median(obj_rect)])
    return dist, obj_rect


if __name__ == '__main__':
    ikili = cv.imread("ikili.png")
    img1 = ikili[:, :ikili.shape[1] // 2]
    img2 = ikili[:, ikili.shape[1] // 2:]

    image_size = Resolution()
    image_size.width = img1.shape[1]
    image_size.height = img1.shape[0]

    serial_number = 20543369
    calibration_file = download_calibration_file(serial_number)
    camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(
        calibration_file, image_size)

    img1 = cv.remap(img1, map_left_x, map_left_y, interpolation=cv.INTER_LINEAR)
    img2 = cv.remap(img2, map_right_x, map_right_y, interpolation=cv.INTER_LINEAR)

    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
