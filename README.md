###
The goal of the project is to use techniques in computer vision to create a pipeline which can identify the boundaries of the road lanes on which a car is moving. 
The input to the pipeline is a stream of images which are taken by a single camera mounted at the middle of the car. 
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
---

### Camera calibration
The code for calibrating camera images can be found in the Camera class implemented in ./src/calibration.py. 
The camera matrix and the distortion coefficients are calculated using the chessboard images provided in ./camera_cal directory. 
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
All thresholding logics are implemented by an instance of the Threshold class implemented in "src/thresholding.py". Only Sobel X thresholding, Sobel Y thresholding and HLS color thresholding was used in the final part of the project. Although Masking class was developed to crop out region outside the lane lines, it was not used. Below is an example of a sample image with its corresponding thresholded counterpart. Note that particular attention is given to make the lane lines as distinctly visible as possible. 
![alt text][image2]
--- 
###

### Perspective Transform
In order to be able to find an appropriate fit to the lane lines, it is important to make a perspective transform of the images so that they appear parallel and not seem to taper off with distance. For this, we implemented the PerspectiveTransform class in "src/perspective_transform.py". This class performs a four point perspective transformation using cv2's getPerspectiveTransform funtionality to give us a bird's eye view of the image. An example of a throsholded image with its perspective transformed counterpart is shown below:
![alt text][image3] 
---
###

### Lanes as Line instances
We store the properties of the lane in instances of Line class implemented in "src/window_search.py". Creating this class helps us in keeping track of the polynomial fit and radius of curvature of the lanes.
---
###

### Lane Pixels
After we have the perspective transformed image, we then try to find all the pixels belonging to each of the lanes. For this, we wrote the WindowSearch class in "src/window_search.py". The most important function in the class is get_lanes, which takes an image as an argument and returns two Line instances, one for the left lane and the other for the right lane. 
---
###

### Road sanity 
It is important to keep in mind the reality of the world in which the car is running and any pipeline which models the lanes on a roads should ensure that the model is consistant with that reality. Some of the crucial checks are
* the two lanes should be nearly parallel to each other
* the radius of curvature of the lanes on the highway should not be too small
* the fit of the lanes should not dramatically change within a small number of successive video frames 
In order to implement these requirement (specially the last one), it is necessary to keep track of the history of the images which the car has seen. 
All these road specific logics are implemented in the Road class in "src/road.py". 
---
###


### Main process image
The heart of the project, the entry point to the entire pipeline, is the process_image function implemented in Process class in "src/main.py". This is the function to which the VideoFileClip library passes each frame of the image.  
###

### Conclusion
The project was a great introduction to techniques used in computer vision specially for the purpose of lane detection. The pipeline performed really well on the project video. One thing which really helped was to keep track of previous lanes and smoothing the current lane based on previously seen ones. Also, it was  	 
###


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!
