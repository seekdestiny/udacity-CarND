[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./out_images/dataset_visualization.png "Data Visualization"
[image1a]: ./out_images/train_set_bar.png "Train Bar Visualization"
[image1b]: ./out_images/test_set_bar.png "Test Bar Visualization"
[image1c]: ./out_images/valid_set_bar.png "Valid Bar Visualization"
[image2]: ./out_images/grayscale.png "Grayscale Visualization"
[image2a]: ./out_images/normalized_vs_orignal.png "Normalized Visualization"
[image3]: ./out_images/train_acc.png "Train Accuracy Visualization"
[image3a]: ./out_images/valid_acc.png "Validation Accuracy Visualization"
[image4]: ./out_images/new_signs.png " New Signs"
[image5]: ./out_images/new_image_prediction.png " New Signs Prediction"
[image6]: ./out_images/new_sign_bar.png " New Signs Bar"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

1. Files submitted. A HTML file, notebook, and write up file are included in repository.
 2. Dataset summary & visualization, the rubric is referring to explaining the size of the dataset and shape of the data therein.  There is some visual explorations as well.
 3. Design & test model: which includes preprocessing, model architecture, training, and solution. LeNet-5 is the starting point.
And then some multi-scale architecture is implemented base on [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
 4. Test model on new images, I found 6 images on the internet which were already classified out of the 43 classes.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/seekdestiny/udacity-CarND/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in `cell #2 - 3` of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The code for this step is contained in `cell #4 - 6` of the IPython notebook.  

Here is an exploratory visualization of the data set.  It pulls in all 43 images and labels them with the correct names in reference with the csv file to their respective id's.

![alt text][image1]

The code for this step is contained in `cell #7` of the IPython notebook.

After this point I also detail the dataset structure by plotting the occurrence of each image class to get an idea of how the data is distributed.  This can help understand where potential pitfalls could occur if the dataset isn't uniform in terms of a baseline occurrence.

![alt text][image1a]

![alt text][image1b]

![alt text][image1c]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The code for this step is contained in `cell #8 - 13` of the IPython notebook.

I convert the images to grayscale because extra color channel adds redundant info for model to fit.
The grayscale pics are shown below.

![alt text][image2]

After the graysscale I also normalized the image data because it speeds up gradient descent to achieve optimal solution.
The normalized pic compared to original one is shown. (only one example is picked)

![alt text][image2a]

I only want to try random_scale, random_brightness, random_warp and so on. Due to limited deadline, I will come back
later to implement it in the future.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in `cell #37` of the iPython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, inputs 28x28x6, outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, inputs 14x14x6, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  inputs 10x10x16, outputs 5x5x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, inputs 5x5x16, outputs 1x1x400    |
| RELU					|												|
| Flatten Layers		| conv2(5x5x16)->400 and conv3(1x1x400)->400  		|
| Concatenate		    | concat two 400 layers to one single 800 layer			|
| Dropout				| 50% keep        									|
| Fully connected		| input 800, output 43        									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I began by implementing the same architecture from the LeNet Lab, with no changes since my dataset is in grayscale. This model worked quite well to begin with (~89.3% validation accuracy), but I also implemented the Sermanet/LeCun model from their traffic sign classifier paper and saw an immediate improvement. Actually, the Sermanet architecture is a little similar to 
the idea used in the inception network which stacks the midstep result from different convolutional filter size.
This can let network adjust between size hyperparameters to improve overfitting.

I used the Adam optimizer (already implemented in the LeNet lab). The final settings used were:

batch size: 128

epochs: 20

learning rate: 0.001

mu: 0

sigma: 0.1

dropout keep probability: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in `cell# 41 - 43` cell of the Ipython notebook.

![alt text][image3]

![alt text][image3a]

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 95.4%
* test set accuracy of 93.8%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  *  I used LeNet-5 architecture used in the previous lab. I used it because it is a easy starting point and it got a good
train and valid accuracy already.
* What were some problems with the initial architecture?
  *  It has some overfitting issue because validation accuracy is much lower than train accuracy.
* How was the architecture adjusted and why was it adjusted?
  *  As mentioned previously, I used the architecture provided in Sermanet/LeCun paper because a feedforward
manner through two stage of convolutions and subsampling can improve overfitting. And dropout technique is also 
accquired.
* Which parameters were tuned? How were they adjusted and why?
  * Epoch, learning rate, batch size, and drop out probability were all parameters tuned. For Epoch the main reason I tuned this was after I started to get better accuracy early on I lowered the number once I had confidence I could reach my accuracy goals.  The batch size I increased only slightly since starting once I increased the dataset size.  The learning rate I think could of been left at .001 which is as I am told a normal starting point, but I just wanted to try something different so .00097 was used.  I think it mattered little.  The dropout probability mattered a lot early on, but after awhile I set it to 50% and just left it.
* What are some of the important design choices and why were they chosen? 
  * I think I could go over this project for another week and keep on learning.  I think this is a good question and I could still learn more about that.  I think the most important thing I learned was having a more uniform dataset along with enough convolutions to capture features will greatly improve speed of training and accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4]

The first image might be difficult to classify because it is white inside the cycle without any special shapes or words.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in `cell #54,#49` of the iPython notebook.

Here are the results of the prediction:
![alt text][image5]

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The answer to this question is above.

The softmax probailities bar char is shown here. As we see, only image 15 has second guess (33%).
![alt text][image6]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


