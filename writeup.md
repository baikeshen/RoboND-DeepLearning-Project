# Deep Learning Project Writeup

## Project Introduction

In this project, the goal is to train a deep neural network to identify and track a target in simulation. The technique used for "follow me” applications could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

In particular, the deep learning technique called Fully Convolutional Network (FCN)  is applied to the images captured by the cameras mounted onto the drone.  with the help of Fully Convolutional Deep Neural Networks, we can train models that will not only be able to identify the contents of an image, it will also be able to figure out what part of the image the object is located at. 

This writeup was prepared for the benefit of the Udacity Robotics Nanodegree Deep Learning Project. The intent is to document and clarify the decisions and the reasons behind these that were made in the project. 

## Network Architecture
In this project, we are using a Fully Convolutional Neural Network to help us in image segmentation and object identification. The network architecture is composed of following layers
- 2 encoder layers
- 1x1 conv layer
- 2 decoder layers

Below is an illustration of the final network architecture.
![network-architecture-overview](misc/fcb.png)

##### FCN:encoding layer

Each encoding layer performs a *depthwise separable convolution*. This requires less compute resources as opposed to using normal convolutions. It is able to accomplish this by significantly reducing the total number of parameters necessary for the computations. Please refered to Paul-Louis Pr?ve in his blog post entitled "[An Introduction to different Types of Convolutions in Deep Learning][1]" which provides a couple of examples illustrating this difference.

In our case, a kernel size of three (3) has been applied. Sride for each encoding layer is two (2), which basically halves the succeeding output's width and height for each encoding layer present. Results are also batch normalized before being returned as output.


##### FCN: 1X1 Convoluational layer

Once done with the encoder blocks, output is then passed as input onto a convolutional 1x1 layer. A 1x1 convolution is essentially convolving with a set of filters of dimensions:

1x1xfilter_size (HxWxD)： 128
stride = 1, and
zero (same) padding.

```
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_1x1 = conv2d_batchnorm(input_layer=enc_bloc2, filters=128, kernel_size=1, strides=1)
```


##### FCN:Decoder

Compared to prviously discussed encoder block as well as the 1X1 convolution layer, the decoder bloack is fairly complex. The decoder block is composed of the following three subparts:

 - upsampling
 - skip connection
 - two succeeding separable convolutions

Firstly, the upsampling part takes the input, and increases its width and height by a factor of two. This is done through what is called a *transposed convolution*; Secondly, The skip connection part concatenates the upsampled data with the corresponding encoder output with the same dimensions. This would allow us to retain information that the image has lost after going through multiple reductions of height and width; Last not least, two succeeding separable convolution layers with a stride of one and desired number of output or filters have been implemented  before returning the output.  


## tuning Hyperparameters

Training the model requires specifying several hyperparameters.  This section of the writeup will attempt to discuss these parameters, indicate the values used, and clarify why such values were chosen

- **learning_rate**: The learning rate is a value that sets how quickly (or slowly) a neural network makes adjustments to what it has learned while it is being trained. The ideal learning rate would allow a neural network to reach the gradient descent and error minimization minimums at the least amount of time, without causing overfitting. In order to get a better learning rate, different valuse have been tested to see which one would be the most effective. In my case, 0.1, 0.5, 0.01, 0.008 and 0.0005 have been tried. Eventually, 0.008 is determined.

- **batch_size**: number of training samples/images that get propagated through the network in a single pass.  A good batch size would be one that is large enough such that it can still be handled by available computing resources, but at the same time not too large to cause overfitting. It should also be small enough but not too small. If the batches are too small, the neural network may have difficulty forming its generalizations. In my case, I used 64 as batch_size for training.

- **num_epochs**: number of times the entire training dataset gets propagated through the network. The ideal number of epochs would be the lowest possible value that would still be able to fully minimize the error rates of the neural network being trained, and avoid overfitting at the same time. I started off at a low value of 4, the number was increased by 4 each step, and it reached to 16. By applying all of these tested values, the train model was still capable of pushing loss and error rates downward. Finally, 12 was used as num_epochs.

- **steps_per_epoch**: number of batches of training images that go through the network in 1 epoch. The value would be based on the chosen batch size. Normally the number of steps chosen would be such that all the available training images would be used for each epoch.One recommended value to try would be based on the total number of images in training dataset divided by the batch_size. The default number is 200. It works pretty well, so no need to change it. 

- **validation_steps**: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. The deafult number 50 is used.

- **workers**: maximum number of processes to spin up. This can affect the training speed and is dependent on your hardware. We have provided a recommended value to work with. 

## Results and Discussion

Multiple runs were initially performed using different numbers of layers and layer depths at lesser steps per epoch (40 steps per epoch instead of 200 steps), starting from 3 layer networks (1 encoder, 1x1 fully connected layer, 1 decoder) to 9 layer networks (4 encoders, 1x1 fully connected layer, 4 decoders).  The initial values for the depths were doubled for each encoder up to the 1x1 convolution layer, then halved for each decoder until the output.  For example, a 5 layer network was run with layer depths of (8, 16, 32, 16, output depth) up to layer depths of (32, 64, 128, 64, output depth). Initial runs showed that 9 layered networks produced the least number of results that returned 0 for the weights, IoU, and final score, and that the returned scores are generally higher than those found in networks with less layers. As such, we focused on training 9 layer networks with layer depths of (32, 64, 128, 256, 512, 256. 128, 64, output depth), at the normal number of steps (200 steps x 40 images per batch or step) per epoch. After performing 28 runs (of 20 epochs each) using the aforementioned parameters, 0.4490 was the highest final score obtained. Furthermore, typical final scores were underwhelming, and are mostly in the 0.20 - 0.30 range, and final scores of 0.0 were fairly common. Getting final scores that were above 0.35 was fairly rare.

Below table shows the accuracy score and training time on a local Ubuntu computer:

ecoders/decoders | filter_size | batch_size | num_epochs | workers | training_time | accuracy_score (%)
------------ | ------------ | ------------- | ------------- | ------------- | ------------- | -------------
3/3 | 32 | 50 | 30 | 3 | 08 hrs. (15 min/epoch) | 36.23
3/3 | 32 | 50 | 40 | 3 | 10 hrs. (15 min/epoch) | 39.78
3/3 | 64 | 48 | 48 | 8 | 28 hrs. (35 min/epoch) | 43.21
4/4 | 32 | 48 | 40 | 8 | 20 hrs. (30 min/epoch) | 38.58

The training time heavily depends on the filter size. By doubling the filter size from 32 to 64 resulted in three times the training time.  Also, keeping the same filter size (32) but making deeper convolution layers, caused the training time to be longer.  However, the deeper model (4 encoders and 4 decoders) with smaller filter depth (32) seems to take lesser training time per epoch than the larger filter depth. Keeping all parameters the same but adding additional convolution layer caused the model training time to be doubles while it did not help improve the accuracy of prediction (compare row2 and row 4 in the above table). However it seems that increasing the filter depth from 32 to 64 and increasing the number of epochs helped improve the prediction accuracy. The final training loss for model trained with encoder size 32 (left, for second row from the table) and 64 (right, for third row from the table) are shown below.




However, some experimentation in quadrupling depth sizes with each encoder instead of doubling them resulted in more consistent and generally better results. In our case, utilizing a 5 layer network, with depth sizes of (16, 64, 256, 64, output depth), at 20 epochs for each run, returned final scores that were consistently better than the previous results.  No final score returned for this configuration was beneath the value of 0.20. Furthermore, scores above 0.40 were fairly common, and constituted about half of the results. And after 19 runs, the highest score of 0.4756 was obtained.

Further experimentation however was done upon the author's realization that it was not necessary to use "output_depth" (which had a value of 3) as the last depth size for the final decoder. As such multiple runs were performed once again, replacing the value of 3 with 16 for the depth of the final decoder, such that the final set of depth sizes was (16, 64, 256, 64, 16) instead of (16, 64, 256, 64, output depth(3)). For the initial set of 24 runs at 20 epochs per each run, the lowest final score obtained was 0.3555 and the highest score obtained was 0.4796. Only five (5) out of the 24 runs (20.83%) had a final score below 0.30. Six (6) out of the 24 runs (25%) were above 0.45, and the rest (13 out of 24 runs or 54%) had scores from a range of 0.40 to 0.45. These figures represent a significant improvement over previous results, however, the increase in the highest score obtained, though exceeding the previous record, only posted a 0.0040 improvement over the previous high. The network architecture for these runs is the final form used in this project, and presented in this writeup.


## Limitations and Potential Improvements
Training and validation data used for all runs were the default ones provided in the project, with the single change of making flipped and mirrored copies of the images to effectively double the amount of training data, from around 4000 images, to more than 8000 training images. But despite this, further improvements to the final score could be achieved by obtaining more training data.

Regarding the use of the same model and data to detect and follow other people or objects, this would not be possible for the current model since it has been trained for a particular image or likeness of a person. However, we could use the same network architecture to train and produce separate models for other objects we would want to be able to detect.  The primary requirement for us to achieve this would be to collect training data that is particular for the person/object of interest, and train a model using this data on the presented network architecture.

Lastly, attempting to perform the training using more epochs and maybe lower learning rates may eke out additional gains to the final score.  Preliminary runs with increased epochs (20 + 5) with the same learning rates did generally improve the scores, however with no scores that exceeded the record high. The lowest score however plummeted (0.18), probably due to overfitting. Trying out other values for the 

[1]: https://medium.com/towards-data-science/types-of-convolutions-in-deep-learning-717013397f4d "An Introduction to different Types of Convolutions in Deep Learning"
