# **Finding Road Lane** 

The goal of the project is to use techniques in computer vision to create a pipeline which can identify the boundaries of the road lanes. The input to the pipeline is a stream of images which are taken by a single camera mounted at the center of the car. 
The specific steps taken to accomplish this goal are: 
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms and gradients to create a thresholded binary image.
* Apply a perspective transform to get a "birds-eye view" of the binary image.
* Detect lane pixels and fit a quadratic curve to find the lane boundary.
* Determine the curvature of the left/right lanes 
* Determine the vehicle position with respect to center.
* Stack the detected lane boundaries onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Create a video where lane boundaries are marked using the pipeline created in the project. 

[//]: # (Image References)

[image1]: ./src/output_images/camera_calibration_combined.jpg "Undistorted"
[image2]: ./src/output_images/thresholded_image_combined.jpg "Thresholded"
[image3]: ./src/output_images/perspective_combined.jpg "Perspective"
[image4]: ./src/output_images/road_stats.jpg "RoadStats"

---

### Camera calibration
The code for calibrating camera images can be found in the [Camera class](https://github.com/spookyQubit/AdvancedLaneDetection/blob/master/src/calibration.py). 
The camera matrix and the distortion coefficients are calculated using the chessboard images provided in [camera caliberation directory](https://github.com/spookyQubit/AdvancedLaneDetection/tree/master/camera_cal). 
Most important member function of this class is get_undistorted which takes in a distorted image and returns the undistorted version of the image.
The effect of undistortion can be seen in the figure below:
![alt text][image1] 

---
###

### Thresholding
After undistorting the image, the next step in the pipeline is to detect only the lanes on the road. To acheive this purpose, we experimented with multiple thresholdings on the image:
* Sobel X thresholding
* Sobel Y thresholding
* Sobel direction thresholding
* Sobel magnitude thresholding
* HLS color thresholding
All thresholding logics are implemented by an instance of the [Threshold class](https://github.com/spookyQubit/AdvancedLaneDetection/blob/master/src/threshold.py). Only Sobel X thresholding, Sobel Y thresholding and HLS color thresholding was used in the final part of the project. Although Masking class was developed to crop out region outside the lane lines, it was not used. Below is an example of a sample image with its corresponding thresholded counterpart. Note that particular attention is given to make the lane lines as distinctly visible as possible. 
![alt text][image2]

--- 
###

### Perspective Transform
In order to be able to find an appropriate fit to the lane lines, it is important to make a perspective transform of the images so that they appear parallel and not seem to taper off with distance. For this, we implemented the [PerspectiveTransform class](https://github.com/spookyQubit/AdvancedLaneDetection/blob/master/src/perspective_transform.py). This class performs a four point perspective transformation using cv2's getPerspectiveTransform funtionality to give us a bird's eye view of the image. An example of a throsholded image with its perspective transformed counterpart is shown below:
![alt text][image3] 

---
###

### Lanes as Line instances
We store the properties of the lane in instances of [Line class](https://github.com/spookyQubit/AdvancedLaneDetection/blob/master/src/line.py). Creating this class helps us in keeping track of the polynomial fit and radius of curvature of the lanes.

---
###

### Lane Pixels
After we have the perspective transformed image, we then try to find all the pixels belonging to each of the lanes. For this, we wrote the [WindowSearch class](https://github.com/spookyQubit/AdvancedLaneDetection/blob/master/src/window_search.py). The most important function in the class is get_lanes, which takes an image as an argument and returns two Line instances, one for the left lane and the other for the right lane. 

---
###

### Road sanity 
It is important to keep in mind the reality of the world in which the car is running and any pipeline which models the lanes on a roads should ensure that the model is consistant with that reality. Some of the crucial checks are
* the two lanes should be nearly parallel to each other
* the radius of curvature of the lanes on the highway should not be too small
* the fit of the lanes should not dramatically change within a small number of successive video frames 
In order to implement these requirement (specially the last one), it is necessary to keep track of the history of the images which the car has seen. 
All these road specific logics are implemented in the [Road class](https://github.com/spookyQubit/AdvancedLaneDetection/blob/master/src/road.py). 

---
###


### Main process image
The heart of the project, the entry point to the entire pipeline, is the process_image function implemented in [Process class](https://github.com/spookyQubit/AdvancedLaneDetection/blob/master/src/main.py). This is the function to which the VideoFileClip library passes each frame of the image. The final lane detected image superimposed on the original image with the lane stats is shown below:
![alt text][image4] 

---
###

### Conclusion
The project was a great introduction to techniques used in computer vision specially for the purpose of lane detection. The pipeline performed really well on the project video. One thing which really helped was to keep track of previous lanes and smoothing the current lane based on previously seen ones. Also, it was  	 
###
