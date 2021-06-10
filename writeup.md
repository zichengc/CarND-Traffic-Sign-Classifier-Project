# **Traffic Sign Recognition** 



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_img/Sample_img.jpg "Visualization"
[image2]: ./output_img/label_dist.jpg "Bar Plot of Labels"
[image3]: ./output_img/Sample_img_gray.jpg "Grayscale Visualization"
[image4]: ./output_img/img_augmentation.jpg "Image Augmentation"
[image5]: ./output_img/original.jpg "Traffic Sign from the web"
[image6]: ./output_img/gray_resize.jpg "Traffic gray"
[image7]: ./output_img/toright.png "Road Narrows to the Right"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/zichengc/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the built-in functions of numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a plot of random selected images from the original dataset. The subplots are labeled with corresponding names of traffic signs. 

![alt text][image1]

Here is a bar plot that visulaizes the porportions of each label.
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color is not critical for the traffic sign classification task and the grayscale images are sufficient.

Here are the same traffic sign images as in figure 1 after grayscaling.
![alt text][image3]


I also applied normalization so that the entries in the input matrices range from 0 to 1. 

When trained with vanilla LeNet, I notices that the training accuracy reaches over 95% while the validation accuracy is 89%. The model is overfitting to the training dataset. To overcome the overfitting issue, a drooput layer is added to the neural network arthitecture. Some image augmentation processes are also applied
1. Rotational transformation with rotation angles between +/- $20^\circ$ 
2. Random zoom-in and zoom-out of the original image with ratios ranging from 0.8 to 1.2.

Here is an example of the image augmentation procedures applied to the original color images.
![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16		|
| Flattening		|        									|
|Dropout |  | rate: 0.2
| Fully connected				| outputs 120        									|
|RELU					|												|
|Fully connected					|	outputs 84											|
|Fully connected| outputs 43|
|Softmax| |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used adam as the optimizer with learning rate $0.001$. Givne the classification task, the model minimizes the crossentropy loss. The performance of the model is evaluated based on classification accuracy.

A batch size of 64 is chosen for the training dataset and the model is trained for 10 full epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9369
* validation set accuracy of 0.9426
* test set accuracy of 0.9261

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The vanilla LeNet is a well-known CNN artichecture for classification tasks and is used as the starting point for this project. The vannilla LeNet achieved over 95% accuracy on the training dataset and the issue is overfitting. First, a dropout layer is added and after some grid search for the rate, 0.2 was picked. Then, the hyperparameters for the image augmentation functions were tuned. Zooming in/ out helps dealing with differences between images in size and plotting different zooming ratios, 0.2 was picked. The classifer should also be invariant to rationtional transformation to some extent since the camera angle may vary or the traffic signs may get loose and the $20^\circ$ range gives best performance. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web using random selected sign names from the training dataset.
![alt text][image5]
and the images were transformed into grayscale images and downscaled to $32x32$.
![alt text][image6]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|Children crossing| Road narrows on the right    	|
| Priority road      			| Priority road  										|
| No vehicles  					| No vehicles  											|
| Dangerous curve to the right	      		| Dangerous curve to the right					 				|
| Bicycles crossing		| Bicycles crossing      							|
|Priority road |Priority road |

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%. 

The first image is misclassified as "Road narrows on the right" and the structure is similar to the traffic sign of "Children crossing".

![alt text][image7]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For all six images, the model predicted outputs with 1.0 probability even the misclassified one.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


