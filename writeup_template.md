# Vehicle Detection Project

## Udacity Self-Driving Car Nanodegree Project 4

---

The scope of this project is to develop a pipeline to process a video stream from a forward-facing camera mounted on the front of a car, and output an annotated video which detects vehicles.

The goals / steps of this project are the following:

1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier;
2. Also apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
3. Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
4. Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
5. Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/data_exploration.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/combined_features.png
[image4]: ./output_images/bboxes_and_heat_1.png
[image5]: ./output_images/bboxes_and_heat_2.png
[image6]: ./output_images/bboxes_and_heat_3.png
[image7]: ./output_images/bboxes_and_heat_4.png
[image8]: ./output_images/bboxes_and_heat_5.png
[image9]: ./output_images/bboxes_and_heat_6.png
[video1]: ./project_video.gif

---

## Install

This project requires **Python 3.5** with the following libraries/dependencies installed:

- [Numpy](http://www.numpy.org/)
- [Matplotlib](http://matplotlib.org/)
- [OpenCV](http://opencv.org/)
- [MoviePy](http://zulko.github.io/moviepy/)
- [scipy](https://www.scipy.org/)
- [skimage](http://scikit-image.org/)
- [sklearn](http://scikit-learn.org/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/).

## Code

- `vehicle-detection.ipynb` - The main notebook of the project.
- `helper_functions.py` - The script contained required helper functions.
- `Vehicles.py` - The script contained a required Python class.
- `README.md` - The writeup explained the image process pipeline and the project.

## Data

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier. These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

---

## Histogram of Oriented Gradients (HOG)

### 1. Extracted HOG features from the training images.

The code for this step is contained in the 3rd code cell of the Jupyter notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

### 2. Settled on the final choice of HOG parameters.

I tried various combinations of parameters and tested them with a linear SVM classifier. Eventually I used all three channels from the `YCrCb` color space with HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. This combination usually yield the best test accuracy on the linear SVM model.

### 3. Trained a classifier using HOG features and color features.

This part of code is contained in the 4th and 5th code cells of the Jupyter notebook.

I trained a linear SVM using a combination of the spatial features, histogram features, and HOG features. First, the input image is converted to `YCrCb` color space. The spatial features are extracted by resizing the image to `16 x 16 pixels` and the flattening the 2D image to 1D vector. The histogram features are extracted by computing a normalized histogram with `32 bins` on each channel and connecting all three channel vectors together. The HOG features are extracted by using the technique explained in the previous section.

All feature values are calculated with a distribution in the range from 0 to 1 to avoid any feature becoming dominant. It should be paid close attention that different image format (PNG vs JPEG) and OpenCV's `cvtColor()` function may create different image data format with different value ranges.

Here is an example of final feature vector before scaling. It should always be checked before next step to make sure they distribute in the proper range.

![alt text][image3]

The final constructed feature vector length is 6156. The features are then scaled, shuffled, and split to training and testing sets. An off-the-shelf linear Support Vector Machine (SVM) model are used to trained the data. The test accuracy is 0.99.

## Sliding Window Search

### 1. Implemented a sliding window search.

This section of code in contained in the 6th code cell of the Jupyter notebook.

A HOG sub-sampling window search is used to find matched objects in the image. Different scales of sliding window were tried and eventually a combination of 64 x 64, 80 x 80, 96 x 96, and 112 x 112 pixels sliding windows are implemented with `scales = [1., 1.25, 1.5, 1.75]`. The combination of various sized windows makes sure generating enough number of bounding boxes for each detected object, therefore is beneficial for the next step of ruling out false positives.

Next step, a heatmap are created by combining overlapped boxes, and thresholded with a criterion to rule out false positives. Then a label function is applied to identify each detected object from the thresholded heatmap.

### 2. Some examples of test images to demonstrate the pipeline is working.

Here are some example images with sliding window scale of 1.5:

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

|`Scale`|`ystart`|`ystop`|
|:-----:|:------:|:-----:|
| 1.0   |  400   |  496  |
| 1.25  |  400   |  528  |
| 1.5   |  400   |  592  |
| 1.75  |  400   |  656  |

---

## Video Implementation

### 1. A link to your final video output.

Here's a [link to my video result](https://youtu.be/_S3sjjg78oc).

And a GIF:

![alt text][video1]

### 2. Implemented filters for false positives and methods for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
