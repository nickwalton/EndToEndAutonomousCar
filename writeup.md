# Simulated Self Driving Car Using Behavioral Cloning


[//]: # (Image References)

[image2]: ./images/TrainingCenter.jpg "Center Line Training"
[image3]: ./images/RecoverRight.jpg "Recovery Image Right"
[image4]: ./images/RecoverLeft.jpg  "Recovery Image Left"
[image5]: ./images/Left.jpg  "Left Image"
[image6]: ./images/Right.jpg  "Right Image"
[image7]: ./images/Center.jpg  "Center Image"

### Included Files
* model.py containing the script to create and train a model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network for the normal track
* jungle_model.h5 containing a trained convolutional neural network for the jungle track
* writeup_report.md or writeup_report.pdf summarizing the results
* normal_track.mp4, a video of the car driving itself on the normal simulated track
* jungle_run.mp4, a video of the car driving itself on a more difficult winding road

#### 2. Running the Simulator using a Trained Model
Using the Udacity provided simulator from https://github.com/udacity/self-driving-car-sim
the car can drive itself by running "python drive.py model.h5" and entering the simulator in Autonomous mode


#### 3. Training a New Model
The model.py file contains the code for training and saving the convolution neural network. By creating data in the simulator driven by yourself you can train a new model that can drive the car autonomously. 

### Model Architecture and Training Strategy

#### 1. Architecture:

My model consists of a convolution neural network with a 4x4 filter and two 3x3 filters and depths between 16 and 36 (model.py lines 86-97) 

The model includes RELU layers (implemented with the convolutional layers) to introduce nonlinearity and the image is normalized using a lambda layer. 

#### 2. Attempts to reduce overfitting in the model

As opposed to classification problems it is a little harder to quantify overfitting as the loss in our loss function doesn't necessarily correspond exactly to the performance on the road. However I used a validation split in my modeling to see my validation performance to get an idea of when to stop the model and I used actual testing to see the model's performance. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 99)

#### 4. Appropriate training data

Training data consisted of two main types. Data for collected of the car staying in the middle of the road and data collected of the car returning to the middle of the road after having gotten off track (recording is started after it's off to track so that the car isn't taught to leave the middle of the road)

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I initially started my network with a convolutional network similar to LeNet. However after a point it was hard to improve it to the level of performance needed. After that I researched the Nvidia End to End Deep Learning architecture and took inspirtation from it. 

I didn't feel I needed to copy it outright as it is a more complex model for a more complex problem with higher resolution of data (I resized my data to 64x64) but I did add another convolutional layer as well as increasing the filter depths to higher levels for each convolutional layer. 


####2. Final Model Architecture

The final model architecture (model.py lines 84-95) consisted of a convolution neural network with the following layers and layer sizes:

-Normalization Layer
-Convolutional layer, filter size of 4x4, depth of 16.
-Relu Activation
-Max Pooling Layer
-Convolutional layer, filter size of 3x3, depth of 24.
-Relu Activation
-Max Pooling Layer
-Convolutional layer, filter size of 3x3, depth of 36.
-Relu Activation
-Max Pooling Layer
-Flatten layer
-Dense hidden layer of 124 neurons
-Dense hidden layer of 48 neurons
-Output layer of 1 steering angle

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it got too far to the left or right of the road. These images show what a recovery looks like starting from both the right and the left. 

![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped each image (and it's corresponding angle) and I used the images taken from the left and right of the car. To these I added either positive or negative 1.2 to the angle to simulate examples of when the car should be turning to get back in the center of the lane.

Here are two examples of the center image of the car with it's corresponding left and right images. 

![alt text][image5]
![alt text][image6]
![alt text][image7]

After the collection process, I had a little over 30,000 data points to train my neural network on. I resized these images in 64x64 images and cropped off the top 20 layers to remove unneeded scenery from these images (I also did the same in drive.py)

In the keras fit function I shuffled 20% of the data into a validation set. 


### Results

After augmenting the data, adding recovery sets of data and getting the final architecture just right I was amazed at how accurate my model was able to become with less than 10min of driving data. My car was able to stay completely on the road on the first track. 

Having been successful on the easier track I decided to try my model on the second track. I collected a similar amoung of data, two loops as well as some recovery data, and after testing and tweaking my model was able to work on the jungle track as well. I was surprised that with the necessary data it didn't take long to train a model that could perform to that level on such a difficult track. 


