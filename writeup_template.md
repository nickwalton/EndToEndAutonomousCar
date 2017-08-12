# Simulated Self Driving Car Using Behavioral Cloning


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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



####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
