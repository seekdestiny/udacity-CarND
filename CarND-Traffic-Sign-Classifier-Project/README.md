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

[image1]: ./out_images/dataset_visualization.jpg "Data Visualization"
[image1a]: ./out_images/train_set_bar.jpg "Train Bar Visualization"
[image1b]: ./out_images/test_set_bar.jpg "Test Bar Visualization"
[image1c]: ./out_images/valid_set_bar.jpg "Valid Bar Visualization"

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

The code for this step is contained in `cell #39` of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The code for this step is contained in `cell #42-43` of the IPython notebook.  

Here is an exploratory visualization of the data set.  It pulls in all 43 images and labels them with the correct names in reference with the csv file to their respective id's.

![alt text][image1]

The code for this step is contained in `cell #44` of the IPython notebook.

After this point I also detail the dataset structure by plotting the occurrence of each image class to get an idea of how the data is distributed.  This can help understand where potential pitfalls could occur if the dataset isn't uniform in terms of a baseline occurrence.

![alt text][image1a]

![alt text][image1b]

![alt text][image1c]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The code for this step is contained in the fifth, sixth, seventh, eighth, ninth, and tenth code cell of the IPython notebook.

At first I tried to convert it to YUV as that was what the technical paper described that was authored by Pierre Sermanet and Yann LeCun.  I had difficulty getting this working at so I skipped over this in order to meet my time requirements.

The next step, I decided to convert the images to grayscale because in the technical paper it outlined several steps they used to achieve 99.7%.  I assume this works better because the excess information only adds extra confusion into the learning process.  After the grayscale I also normalized the image data because I've read it helps in speed of training and performance because of things like resources.  Also added additional images to the datasets through randomized modifications.

Here is an example of a traffic sign images that were randomly selected.  

![alt text][image2]

Here is a look at the normalized images. Which should look identical, but for some small random alterations such as opencv affine and rotation.  
![alt text][image2a]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eleventh cell of the iPython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 1x1	    | 2x2 stride, valid padding, outputs 1x1x412    |
| RELU					|												|
| Fully connected		| input 412, output 122        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 122, output 84        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 84, output 43        									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an LeNet for the most part that was given, but I did add an additional convolution without a max pooling layer after it like in the udacity lesson.  I used the AdamOptimizer with a learning rate of 0.00097.  The epochs used was 27 while the batch size was 156.  Other important parameters I learned were important was the number and distribution of additional data generated.  I played around with various different distributions of image class counts and it had a dramatic effect on the training set accuracy.  It didn't really have much of an effect on the test set accuracy, or real world image accuracy.  Even just using the default settings from the Udacity lesson leading up to this point I was able to get 94% accuracy with virtually no changes on the test set.  When I finally stopped testing I got 94-95.2% accuracy on the test set though so I think the extra data improved training accuracy, but not a huge help for test set accuracy.  Although this did help later on with the images from the internet.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 16th and 17th cell of the Ipython notebook.

![alt text][image3]

![alt text][image3a]

My final model results were:
* training set accuracy of 100.0%
* validation set accuracy of 99.3%
* test set accuracy of 94.2%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  *  I used a very similar architecture to the paper offered by the instructors.  I used it because they got such a good score the answer was given through it.
* What were some problems with the initial architecture?
  *  The first issue was lack of data for some images and the last was lack of knowledge of all the parameters.  After I fixed those issues the LeNet model given worked pretty well with the defaults.  I still couldn't break 98% very easily until I added another convolution.  After that it was much faster at reaching higher accuracy scores.
* How was the architecture adjusted and why was it adjusted?
  * Past what was said in the previous question, I didn't alter much past adding a couple dropouts with a 50% probability.
* Which parameters were tuned? How were they adjusted and why?
  * Epoch, learning rate, batch size, and drop out probability were all parameters tuned along with the number of random modifications to generate more image data was tuned.  For Epoch the main reason I tuned this was after I started to get better accuracy early on I lowered the number once I had confidence I could reach my accuracy goals.  The batch size I increased only slightly since starting once I increased the dataset size.  The learning rate I think could of been left at .001 which is as I am told a normal starting point, but I just wanted to try something different so .00097 was used.  I think it mattered little.  The dropout probability mattered a lot early on, but after awhile I set it to 50% and just left it.  The biggest thing that effected my accuracy was the data images generated with random modifications.  This would turn my accuracy from 1-10 epochs from 40% to 60% max to 70% to 90% within the first few evaluations. Increasing the dataset in the correct places really improved the max accuracy as well.
* What are some of the important design choices and why were they chosen? I think I could go over this project for another week and keep on learning.  I think this is a good question and I could still learn more about that.  I think the most important thing I learned was having a more uniform dataset along with enough convolutions to capture features will greatly improve speed of training and accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9]

I used semi-easy images to classify and even modified them slightly.  I made them all uniform in size and only had one partially cut off.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the last cell of the iPython notebook.

Here are the results of the prediction:
![alt text][image10a]
![alt text][image10b]
![alt text][image10c]
![alt text][image10d]
![alt text][image10e]
![alt text][image10f]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set although I did throw it a softball.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The answer to this question is above.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

