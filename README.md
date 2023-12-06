# Distance-Map-Extraction-and-Road-Defect-Detection-Using-Stereo-Vision

####Details of implementation and results can be found in Report.pdf

### Abstract
With the advancements in computer vision and the capability of our processors, real-time methods
used in driving systems have become more important. Detecting the objects on the road and locat-
ing them is an open problem. In defense industry, especially for land vehicles, autonomous object
detection and distance estimation are crucial. The purpose of this project is to detect potholes
and speed bumps on the road and estimate their relative distances to the user by utilizing stereo
vision data. To this end, a stereo camera was used for data acquisition, and a personal computer
was used as a processing unit. The data from the stereo camera was first processed with a deep
learning-based object detection method. Then, the disparity map of two cameras was extracted
and used to estimate the distance from detected objects to the camera. The baseline algorithms
existing in the current literature and the built-in algorithm in the camera were tried. After getting
results from the object detection and distance estimation tasks, we matched their outputs on real-
time frames, and we projected them onto a two-dimensional surface considering perspective shifts.
We basically created a map on which the detected objects and their distances to the car are shown.
YOLOv5 was used as an object detection deep network. The network was trained on Google Colab
GPUs and tested on a personal computer. For simulation of our network, we tried our method on
both saved video and real-time experiments. We mainly used PyTorch and OpenCV frameworks
with Python programming language to train and test our network. We searched on the Internet for
training datasets and collected an additional dataset in Ankara. After a literature survey, we im-
plemented block matching (BM) and semi-global block matching (SGBM) in addition to the built-in
algorithm of the camera for distance estimation. We worked on improving the accuracy and speed
of all methods. We validated the results of each algorithm in a testing environment, the roads of
Ankara, by measuring the distances of the test objects. Also, we provided users with controls on
object detection region and confidence threshold through the interface. We used OpenCV and ZED
SDK, Python package of the selected camera, to obtain the visual data and extract disparity infor-
mation. The requirements and expected results are discussed in the report. Our aim is to achieve
all of the requirements in the testing environment, which is the roads of Ankara. The end product
is capable of detecting 80% of potholes and 85% of speed bumps. The final distance estimation
error is 5% up to 10 meters and 12% from 10 to 20 meters. The complete project with all work
packages including GUI works with 16 frame-per-second (FPS) in real-time tests. The outcomes
of this project can be used in a single car or swarm systems with and without a driver inside.



### Train

To run the model run the "train.ipynb"

### GUI Run

To run the stero vision model and detection model with integrated GUI :

    python codes/yolozed.py




