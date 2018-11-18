# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

### Pipeline Steps:
* Read in and grayscale the image
* Define a kernel size and apply Guassian Smoothing
* Define our parameters for Canny and run it to get edges of the image
* Mask edges using cv2.fillPoly() to get the region of interest
* Define Hough transform parameters and run Hough Transform on masked edge-detected image
* Draw line segments
* Draw lines extrapolated from line segments
* combine line image with original image to see how accurate the line annotations are.

### I modified the draw_lines function by

* 1. sum over slope, x and y.
* 2. separate left and right line segments by checking positive and negative slope
* 3. calculate avg of slope, x and y for left and right line segments
* 4. use avg (slope, x, y) to extrapolate bottom and top points of the line
* 5. draw line for both left and right lines


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be that I hard coded many parameter such as kernel_size,
threshold and so on. This makes this pipeline less flexible. If the image pattern changes,
the result will be wrong.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be that open parameters setup as pipeline input and make this function as basic class.

We may use some child classes inheriting this basic class to detect different image patterns.