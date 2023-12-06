import cv2
import numpy as np
import time
import sys
import torch
import torch.nn as nn
from pathlib import Path
from utils.general import non_max_suppression, scale_coords

import pyzed.sl as sl
from lol import *

#def main():
#torch.backends.cudnn.enabled = False
upCut = 2/5
drawDown = True
downCut = 1/5
modelTorchPath = "weights/best.pt"
half = True #Half precision
#imgsz = (640, 352) #multiple of 32
imgsz = (1024, 640) #multiple of 32
upCutPix = imgsz[1]*upCut
upCutPix = int(upCutPix-upCutPix%32)
downCutPix = imgsz[1]*(1-downCut)
downCutPix = int(downCutPix -downCutPix%32)
resizeType = 'nothing'#'nothing'#'aspectRatioShort'# string; one of {pad, aspectRatioLong, aspectRatioShort, nothing}
device = 'cuda'
conf_thres = 0.45
iou_thres = 0.45
setFps = 0# set 0 if no limit 
classes = [0, 1]
classNames = ["Pothole", "Speed Bump"]
agnostic_nms = 'store_true'
max_det = 1000
font = cv2.FONT_HERSHEY_PLAIN
colors=[(255,0,0),(0,0,255)]
thicknessBorder = 2
newAspectRatio = imgsz[0]/imgsz[1]
outputSize = (int(imgsz[0]*1.5),int(imgsz[1]*1.5))
saveAsVideo = True
depthAlgo = 'zed' # 'sgbm' # 'zed'
if depthAlgo != 'zed':
    from calib import *
    from lol import *
    serial_number = 20543369
    calibration_file = download_calibration_file(serial_number)
    image_size = Resolution()
    image_size.width = 1920
    image_size.height = 1080
    camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(
        calibration_file, image_size)
#    if depthAlgo == 'bm':
#        stereo = cv2.StereoBM_create(
#        numDisparities=256,
#        blockSize=17
#    )
#        stereo.setPreFilterSize(9)
#        stereo.setPreFilterCap(31)
#        stereo.setTextureThreshold(10)
#        stereo.setUniquenessRatio(12)
#        stereo.setSpeckleRange(10)
#        stereo.setSpeckleWindowSize(150)
#    elif depthAlgo =='sgbm':
#        blockSize = 11
#        stereo = cv2.StereoSGBM_create(
#        minDisparity=0,
#        numDisparities=384,
#        blockSize=blockSize,
#        uniquenessRatio=12,
#        speckleWindowSize=150,
#        speckleRange=10,
#        disp12MaxDiff=0,
#        P1=8 * 1 * blockSize * blockSize,
#        P2=32 * 1 * blockSize * blockSize,
#        mode=0
#    )
    
if saveAsVideo: 
    videoWriter = cv2.VideoWriter('output'+str(int(time.time()))+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 14, imgsz)


# Create a Camera object
filePath = "/home/yavuz/zed/yolo/HD1080_SN20543369_11-26-12-001.svo"
zed = sl.Camera(filePath)

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.set_from_svo_file(filePath) # !!

init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
init_params.camera_fps = 30  # Set fps at 20
init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Use ULTRA depth mode
init_params.coordinate_units = sl.UNIT.MILLIMETER # Use millimeter units (for depth measurements)

right_image = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4) if depthAlgo != 'zed' else None
left_image = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4) 
depth_map = sl.Mat() if depthAlgo == 'zed' else None
runtime_parameters = sl.RuntimeParameters()

# Open the camera
err = zed.open(init_params)
zed.set_svo_position(15500) # Video file start time
if err != sl.ERROR_CODE.SUCCESS:
    sys.exit(1)
 
model = DetectMultiBackend(modelTorchPath, device)
model.model.half() if half else model.model.float()

frameEnd = time.time()
idx=0
pred = []

fps_start = 10
fps_mean_length = 40
fps_arr = [fps_start]*fps_mean_length

while True:
    idx += 1
    frameStart = time.time()
            
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        frame = left_image.get_data()[:,:,:3]

            
    height, width, channels = frame.shape
    
    
    out = cv2.resize(frame,imgsz)
    if True: #idx % 2 == 0:
        im = torch.from_numpy(out).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im = im / 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        im = im.permute(0,3,1,2)# reorder dimensions
        pred = model(im[:,:,upCutPix:downCutPix,:])
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        pred = pred[0]        
        
    depth_value = 0
    #out = im.squeeze().permute(1,2,0).cpu().numpy()
    if len(pred):
        
        if depthAlgo == 'zed':
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        else:
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            frameR = right_image.get_data()[:,:,:3]
        
        for i in range(pred.shape[0]):
            det=pred[i,:].unsqueeze(0)
            # Rescale boxes from img_size to im0 size
            det[:,:4] = scale_coords(im.shape[2:], det[:,:4], (imgsz[1], imgsz[0])).round()
            detNp = det.cpu().numpy()
            xycoords1 = tuple([int(detNp[0,0]),int(detNp[0,1]+upCutPix)])
            xycoords2 = tuple([int(detNp[0,2]),int(detNp[0,3]+upCutPix)])

            x_resized = (xycoords1[0]+xycoords2[0])/2
            y_resized = (xycoords1[1] + xycoords2[1]) / 2
            x = x_resized*1920/imgsz[0]
            y = y_resized * 1080 / imgsz[1]

            xarray = np.linspace(xycoords1[0],xycoords2[0],xycoords2[0]-xycoords1[0],dtype="int32")
            x1=int(xycoords1[0]*1920/imgsz[0])
            x2 = int(xycoords2[0]*1920/imgsz[0])
            y1= int(xycoords1[1]* 1080 / imgsz[1])
            y2= int(xycoords2[1]* 1080 / imgsz[1])

            borderx1=int((x1+x)//2)
            borderx2=int((x2+x)//2)
            bordery1=int((y1+y)//2)
            bordery2=int((y2+y)//2)

            if depthAlgo == 'zed':
                dArr = (depth_map.get_data())
                dArr=dArr[bordery1:bordery2,borderx1:borderx2]
                dArr =dArr[np.isfinite(dArr)]
                dArr = dArr[~np.isnan(dArr)]
                depth_value=np.median(dArr)
            elif depthAlgo == 'bm':
                t = time.time()
                frame = cv2.remap(frame, map_left_x, map_left_y, interpolation=cv.INTER_LINEAR)
                print('Rect 1:', time.time()-t)
                t = time.time()
                frameR = cv2.remap(frameR, map_right_x, map_right_y, interpolation=cv.INTER_LINEAR)
                print('Rect 1:', time.time()-t)
                t = time.time()
                
                depth_value = computeDistance(stereo, cv.cvtColor(frame, cv.COLOR_BGR2GRAY),cv.cvtColor(frameR, cv.COLOR_BGR2GRAY),x1,x2,y1,y2, M=126.6, mode="median")
                print('Rect 1:', time.time()-t)
                t = time.time()
                
            elif depthAlgo == 'sgbm':
                depth_value = computeDistance(stereo, frame,frameR,x1,x2,y1,y2, M=126.6, mode="median"),
            elif depthAlgo == 'fastbm':
                frame = cv2.remap(frame, map_left_x, map_left_y, interpolation=cv.INTER_LINEAR)
                frameR = cv2.remap(frameR, map_right_x, map_right_y, interpolation=cv.INTER_LINEAR)
                M = 126.6
                depth_value = 1000*M/fastBM(torch.Tensor(frame/255.), torch.Tensor(frameR/255.), windowSize = 25, maxDisp = 256, x1=x1,y1=y1, x2=x2,y2=y2 ,centerRatio = 0.5,stride = 2 , device='cpu')
                
            
            confidence = detNp[0, -2]
            classCode = int(detNp[0,-1])
            label = classNames[classCode]
            if y>300:
                cv2.rectangle(out, xycoords1, xycoords2, colors[classCode], thickness=thicknessBorder)
                if np.isnan(depth_value) or depth_value/1000-4<0:
                    cv2.putText(out, label + " - "+ str(round(confidence, 2)), xycoords1, font, 1, colors[classCode], thickness=1)                
                else:
                    cv2.putText(out, label + " " + str(round(depth_value/1000-4,2)) + 'm '+ str(round(confidence, 2)), xycoords1, font, 1, colors[classCode], thickness=1)
    
    if drawDown:
        cv2.rectangle(out, (1,downCutPix), (imgsz[0]-10,downCutPix), colors[0], thickness=thicknessBorder)
        cv2.rectangle(out, (1,upCutPix), (imgsz[0]-10,upCutPix), colors[0], thickness=thicknessBorder)
    tempFrameEnd = time.time()
    fpsCalc = 1/(tempFrameEnd-frameEnd)
    fps_arr = [*fps_arr[1:], fpsCalc]
    frameEnd = tempFrameEnd
    cv2.putText(out, "FPS: " + str(round(sum(fps_arr)/fps_mean_length, 2)), (5, 30), font, 1, (0, 0, 0), 1)
    #out = cv2.resize(out, outputSize)
    cv2.imshow("EEE 493 | "+str(imgsz[0])+" x "+str(imgsz[1]), out) #out
    
    if saveAsVideo:
        videoWriter.write(out)
    cv2.imshow("EEE 493 | "+str(imgsz[0])+" x "+str(imgsz[1]), out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
           

videoWriter.release()
cv2.destroyAllWindows()

#if __name__ == "__main__":
#    main()




