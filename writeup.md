# Deep Learning Project Writeup
[image1]: ./misc/SegNet_Badrinaray_2016.jpg
[image2]: ./misc/SegNet_Long_2014.jpg
[image3]: ./misc/BK_FCN_01.jpg
[image4]: ./misc/BK_FCN_04.jpg

## Project Introduction

In this project, the goal is to train a deep neural network to identify and track a target in simulation. The technique used for "follow me” applications could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

In particular, the deep learning technique called Fully Convolutional Network (FCN)  is applied to the images captured by the cameras mounted onto the drone.  with the help of Fully Convolutional Deep Neural Networks, we can train models that will not only be able to identify the contents of an image, it will also be able to figure out what part of the image the object is located at. 

This writeup was prepared for the benefit of the Udacity Robotics Nanodegree Deep Learning Project. The intent is to document and clarify the decisions and the reasons behind these that were made in the project. 

## Network Architecture
In this project, we are using a Fully Convolutional Neural Network to help us in image segmentation and object identification. The network architecture is composed of following layers
- encoder layers
- conv layer
- decoder layers

Below is an example of FCN architecture which shows an FCN network called SegNet taken from paper by Badrinarayanan, Kendall, and Cipolla.

![alt text][image1]

##### FCN:encoding layer

Each encoding layer performs a *depthwise separable convolution*. This requires less compute resources as opposed to using normal convolutions. It is able to accomplish this by significantly reducing the total number of parameters necessary for the computations. Please refered to Paul-Louis Pr?ve in his blog post entitled "[An Introduction to different Types of Convolutions in Deep Learning][1]" which provides a couple of examples illustrating this difference.

In our case, a kernel size of three (3) has been applied. Sride for each encoding layer is two (2), which basically halves the succeeding output's width and height for each encoding layer present. Results are also batch normalized before being returned as output.


##### FCN: 1X1 Convoluational layer

Once done with the encoder bloc Shallow Model 2 with 1X1 convolutionsks, output is then passed as input onto a convolutional 1x1 layer. A 1x1 convolution is essentially convolving with a set of filters of dimensions, including filter_size (HxWxD), stride and padding.

in TensorFlow, the output shape of a convolutional layer is a 4D tensor. However, when we wish to feed the output of a convolutional layer into a fully connected layer, we flatten it into a 2D tensor. This results in the loss of spatial information, because no information about the location of the pixels is preserved. This could be avoided by using 1X1 Fully Convolution layer.

##### FCN:Decoder

Compared to prviously discussed encoder block as well as the 1X1 convolution layer, the decoder bloack is fairly complex. The decoder block is composed of the following three subparts:

 - upsampling
 - skip connection
 - succeeding separable convolutions

Firstly, the upsampling part takes the input, and increases its width and height by a factor of two. This is done through what is called a *transposed convolution*; Secondly, The skip connection part concatenates the upsampled data with the corresponding encoder output with the same dimensions. This would allow us to retain information that the image has lost after going through multiple reductions of height and width; Last not least, two succeeding separable convolution layers with a stride of one and desired number of output or filters have been implemented  before returning the output.  


## Proposed Architectures & Implementation

As shown in the original FCN architecture by Shellhamer, Long, and Darrell (2014) below, FCN networks have many layers to performance segmentation on images. 

![alt text][image2]

However, The proposed models here are targeted on semantic segmentation of images taken from 3D simulated environment. It is unnecceary to duplicate this kind of architecture in order to get some reasonable performance. So smaller models are proposed for investigations:

- Shallow Model 1 with 1X1 convolutions
- Deep Model with 1X1 convolutions


#### Shallow Model with 1X1 convolutions

There are total of 5 layers consisted of this proposed model. The network architecture is shown as below:

![alt text][image3]

This model Loss values over time is shown as below, which demonstrate rapid learning. The evaluated intersection over union for this model was 0.427260086539.

![alt text][image5]


#### Deep Model with 1X convolution

This is a aliitle bit deeper model compared to shallow one above. 

![alt text][image4]

Below is the loss value over time during training period:

![alt text][image6]

 The evaluated intersection over union for this model was 0.430452304758.

## tuning Hyperparameters

Training the model requires specifying several hyperparameters.  This section of the writeup will attempt to discuss these parameters, indicate the values used, and clarify why such values were chosen

- **learning_rate**: The learning rate is a value that sets how quickly (or slowly) a neural network makes adjustments to what it has learned while it is being trained. The ideal learning rate would allow a neural network to reach the gradient descent and error minimization minimums at the least amount of time, without causing overfitting. In order to get a better learning rate, different valuse have been tested to see which one would be the most effective. In my case, 0.1, 0.5, 0.01,and 0.0005 have been tried. Eventually, 0.01 is determined.

- **batch_size**: number of training samples/images that get propagated through the network in a single pass.  A good batch size would be one that is large enough such that it can still be handled by available computing resources, but at the same time not too large to cause overfitting. It should also be small enough but not too small. If the batches are too small, the neural network may have difficulty forming its generalizations. In my case, I used 64 as batch_size for training.

- **num_epochs**: number of times the entire training dataset gets propagated through the network. The ideal number of epochs would be the lowest possible value that would still be able to fully minimize the error rates of the neural network being trained, and avoid overfitting at the same time. I started off at a low value of 4, the number was increased by 4 each step, and it reached to 16. By applying all of these tested values, the train model was still capable of pushing loss and error rates downward. Finally, 12 was used as num_epochs.

- **steps_per_epoch**: number of batches of training images that go through the network in 1 epoch. The value would be based on the chosen batch size. Normally the number of steps chosen would be such that all the available training images would be used for each epoch.One recommended value to try would be based on the total number of images in training dataset divided by the batch_size. The default number is 200. It works pretty well, so no need to change it. 

- **validation_steps**: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. The deafult number 50 is used.

- **workers**: maximum number of processes to spin up. This can affect the training speed and is dependent on your hardware. We have provided a recommended value to work with. 

## Discussion & Improvement

The training time heavily depends on number of layers and the filter size. By doubling the filter size from 32 to 64 resulted in 2~4 times the training time.  Also, keeping the same filter size (32) but making deeper convolution layers, caused the training time to be longer. Keeping all parameters the same but adding maxpooling2D layer to the encoder significantly increase the training time.

Training and validation data used for all runs were the default ones provided in the project, further improvements to the final score could be achieved by obtaining more training data.

Adding pooling layers could achieve higher score, but it will significantly increase the comupting time.

Lastly, using more epochs and maybe lower learning rates should improve the final score. 
