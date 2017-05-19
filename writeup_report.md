# **Behavioral Cloning** 


**Project Goals**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./model.png "Model Visualization"
[hist1]: ./histogram_input_steering_angles.png "Default steering angles."
[hist2]: ./histogram_left_right_steering_angles.png "Left and right augmented steering angles."
[hist3]: ./histogram_final_steering_angles.png "Final steering angles with flipped input."

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy Overview

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64. It is adapted from the Nvidia Paper "End to End Learning for Self-Driving Cars" by Bojarski et al., 2016.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

The final model architecture (model.py lines 102-120) consisted of a convolution neural network with five convolutional layers and 5 fully connected layers.

Here is a visualization of the architecture including the layer sizes:
![Model Structure][model]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on the provided data set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used images from all three cameras.
For the left and right camera images, an artifical steering angle bias of +/-0.2 has been employed
to steer the vehicle back to the track.

For details about how I created the training data, see the next section. 

### Detailed Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to take the hints provided in the lesson
and go for the Nvidia End-to-End architechture.

My first step was to implement a convolution neural network model as described in the paper. I thought this model might be appropriate because it is proven in use by a real car.

In order to gauge how well the model was working,
I split my image and steering angle data into a training and validation set.
Both sets are fed through a generator, providing augmented data to the fit and validation procedure.
The generator takes a `batch_size` number of samples from the shuffled sample set,
and first generates three samples: the center image combined with the recorded steering angle, then left and right image with augmented steering angles where 0.2 rad are added or subtracted from the recorded steering angle.
Then these three samples are copied and flipped horizontally while inverting the steering angle.
Thus the bias of the recorded data is leviated.
At the end of the generator it is ensured that only a `batch_size` number of samples is returned at once.
The augmented data is again shuffled to not introduce any biases.

The final step was to run the simulator to see how well the car was driving around track one.
Having implemented all the augmentation up-front, before any testing, I was enlighted,
that the vehicle drove the whole track without any error. I was also able to increase the velocity
to 15 mph without leaving the track. Although, for this speed, it could be observed,
that the car drives straight parts of the track in form of a sine function, wiggling left to right and back.

To obtain the optimal training result I relied on
the Keras callback architecture and used
the ModelCheckpoint callback to save the model
every time the validation loss decreased below the
best loss observed. Using this I trained
for 20 epochs and ended up using the result from epoch 14.

#### 2. Analysis of the data and the augmented data

Analyzing the steering angle distribution of the sample training data
gives the following picture:
![Steering angle distribution of the sample training data][hist1]

We find, that the data has a lot of samples with a steering angle of zero.

After augmenting the data by using the left and right images together with
artifical steering angles generated by adding or subtracting a constant from the
recorded steering angle, we obtain an improved distribution:
![Augmented steering angle distribution][hist2]

Furthermore, I added flipped the images belonging to the above steering angle values
and inverted the steering angle accordingly.
The resulting distribution looks like:
![Final augmented steering angle distribution][hist3]