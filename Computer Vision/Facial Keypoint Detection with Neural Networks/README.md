Project 4: Facial Keypoint Detection with Neural Networks
=============================================================

In this project, we use convolutional neural networks to  automatically detect facial keypoints!

### Part 1: Nose Tip Detection

I began by loading in images from the IMM Face Database to use for training our toy models. For preprocessing, I rescale images to be of size 60x40 pixels, transform the images to greyscale, and normalize the pixel values between -0.5 and 0.5. Below are sampled image from my custom dataloader visualized with ground-truth keypoints.

<table>
<col width="33%" />
<col width="33%" />
<col width="33%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/nose_keypoint2.jpg" alt="sample nose 1" /></p>
<p>Sample Image 1</p></td>
<td align="left"><p><img src="outputs/nose_keypoint36.jpg" alt="sample nose 1" /></p>
<p>Sample Image 2</p></td>
<td align="left"><p><img src="outputs/nose_keypoint47.jpg" alt="sample nose 1" /></p>
<p>Sample Image 3</p></td>
</tr>
</tbody>
</table>

My neural network is as follows:

NoseNet(
 (conv1): Conv2d(1, 12, kernel\_size=(7, 7), stride=(1, 1))
 (conv2): Conv2d(12, 16, kernel\_size=(5, 5), stride=(1, 1))
 (conv3): Conv2d(16, 32, kernel\_size=(3, 3), stride=(1, 1))
 (pool): MaxPool2d(kernel\_size=3, stride=2, padding=0, dilation=1, ceil\_mode=False)
 (fc1): Linear(in\_features=576, out\_features=200, bias=True)
 (fc2): Linear(in\_features=200, out\_features=2, bias=True)
)

I used mean squared error loss as the prediction loss and trained the network using the Adam optimizer with a learning rate of 0.001.
Training extended over 25 epochs, and the results displayed promising accuracy, though some challenges were encountered due to the limited size of the training sample.

<table>
<col width="50%" />
<col width="50%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/training_loss_nose.jpg" alt="Training accuracy" /></p>
<p>Training Accuracy</p></td>
<td align="left"><p><img src="outputs/validation_loss_nose.jpg" alt="Training accuracy" /></p>
<p>Validation Accuracy</p></td>
</tr>
</tbody>
</table>

Here are my results:

<table>
<col width="50%" />
<col width="50%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/nose_predictions/nose_41.jpg" alt="Nose Correctly Detected" /></p>
<p>Nose Correctly Detected</p></td>
<td align="left"><p><img src="outputs/nose_predictions/nose_30.jpg" alt="Nose Correctly Detected" /></p>
<p>Nose Correctly Detected</p></td>
</tr>
</tbody>
</table>

Looking at these results, I think this initial model preforms quite well on most photos.

<table>
<col width="50%" />
<col width="50%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/nose_predictions/nose_12.jpg" alt="Nose Correctly Detected" /></p>
<p>Nose Not Correctly Detected</p></td>
<td align="left"><p><img src="outputs/nose_predictions/nose_21.jpg" alt="Nose Correctly Detected" /></p>
<p>Nose Not Correctly Detected</p></td>
</tr>
</tbody>
</table>

Other factors that may have contributed to low accuracy it was the constrast, facial oritentation, or saturation of the photo.

### Part 2: Full Facial Keypoint Detection

I followed a similar structure for this portion of the project as well. To prevent the trained model from overfitting, I augmented the data by rotating the image by random angle, between -10 and 10 degrees, as well as by randomly changing the images' brightness, hue, saturation, and contrast. Here are sampled image from my dataloader visualized with ground-truth keypoints.

<table>
<col width="33%" />
<col width="33%" />
<col width="33%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/face_keypoint2.jpg" alt="sample nose 1" /></p>
<p>Sample Image 1</p></td>
<td align="left"><p><img src="outputs/face_keypoint36.jpg" alt="sample nose 1" /></p>
<p>Sample Image 2</p></td>
<td align="left"><p><img src="outputs/face_keypoint47.jpg" alt="sample nose 1" /></p>
<p>Sample Image 3</p></td>
</tr>
</tbody>
</table>

My neural network was a follows:

FaceNet(
 (conv1): Conv2d(1, 4, kernel\_size=(7, 7), stride=(1, 1))
 (conv2): Conv2d(4, 8, kernel\_size=(5, 5), stride=(1, 1))
 (conv3): Conv2d(8, 16, kernel\_size=(3, 3), stride=(1, 1))
 (conv4): Conv2d(16, 32, kernel\_size=(3, 3), stride=(1, 1))
 (conv5): Conv2d(32, 56, kernel\_size=(3, 3), stride=(1, 1))
 (pool): MaxPool2d(kernel\_size=2, stride=2, padding=0, dilation=1, ceil\_mode=False)
 (fc1): Linear(in\_features=840, out\_features=420, bias=True)
 (fc2): Linear(in\_features=420, out\_features=112, bias=True)
)

As for hyperparamters, I used a batch size of 1 and learning of 0.0001. Again, I used the Adam Optimizer and MSE for loss.

Here is my training and validation accuracy during the training process, over 30 epochs:

<table>
<col width="50%" />
<col width="50%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/training_loss_face.jpg" alt="Training accuracy" /></p>
<p>Training Accuracy</p></td>
<td align="left"><p><img src="outputs/validation_loss_face.jpg" alt="Training accuracy" /></p>
<p>Validation Accuracy</p></td>
</tr>
</tbody>
</table>

Here are my results:

<table>
<col width="50%" />
<col width="50%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/face_predictions/face_33.jpg" alt="Nose Correctly Detected" /></p>
<p>Face Correctly Detected</p></td>
<td align="left"><p><img src="outputs/face_predictions/face_34.jpg" alt="Nose Correctly Detected" /></p>
<p>Face Correctly Detected</p></td>
</tr>
</tbody>
</table>

<table>
<col width="50%" />
<col width="50%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/face_predictions/face_22.jpg" alt="Nose Correctly Detected" /></p>
<p>Face Not Correctly Detected</p></td>
<td align="left"><p><img src="outputs/face_predictions/face_7.jpg" alt="Nose Correctly Detected" /></p>
<p>Face Not Correctly Detected</p></td>
</tr>
</tbody>
</table>


Again, I think this dataset was rather small to train which can contribute the wrong detections. The missed detections could result from exposure, or some other image attribute, being too dramatic or lack there of, cause the image to be an outlier and thus making the neural network less keen to it.

Here is a visualiztion of my learned filters:

<table>
<col width="33%" />
<col width="33%" />
<col width="33%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/face_conv1.jpg" alt="sample nose 1" /></p>
<p>1st Convolutional Layer</p></td>
<td align="left"><p><img src="outputs/face_conv2.jpg" alt="sample nose 1" /></p>
<p>2nd Convolutional Layer</p></td>
<td align="left"><p><img src="outputs/face_conv3.jpg" alt="sample nose 1" /></p>
<p>3rd Convolutional Layer</p></td>
</tr>
</tbody>
</table>

### Part 3: Large Dataset

For this part, I utilized google colab in addition to a GPU. I sampled images from the 'ibug face in the wild dataset'. I augmented the data for this section as well, again rotating the image and randomly changing image attributes. Below are sampled image from my custom dataloader visualized with ground-truth keypoints.

<table>
<col width="33%" />
<col width="33%" />
<col width="33%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/part3/big_face_keypoint126.jpg" alt="sample nose 1" /></p>
<p>Sample Image 1</p></td>
<td align="left"><p><img src="outputs/part3/big_face_keypoint789.jpg" alt="sample nose 1" /></p>
<p>Sample Image 2</p></td>
<td align="left"><p><img src="outputs/part3/big_face_keypoint5678.jpg" alt="sample nose 1" /></p>
<p>Sample Image 3</p></td>
</tr>
</tbody>
</table>


My neural network is the ResNet18 which is a predefined PyTorch model. I made two modifications including making the first layer input channel to 1  as the inputs are grayscale images. I also made last layer's output channel number be 68 * 2 = 136, since there are 68 landmarks for each image. I choose learning rate to be lr = 0.001, and used a batch size of 1. I used the same optimizer and loss function as peviously described.

Here is my training and validation accuracy during the training process:

<table>
<col width="50%" />
<col width="50%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/part3/training_loss.jpg" alt="Training accuracy" /></p>
<p>Training Accuracy</p></td>
<td align="left"><p><img src="outputs/part3/validation_loss.jpg" alt="Training accuracy" /></p>
<p>Validation Accuracy</p></td>
</tr>
</tbody>
</table>

After training the data set, I found the the mean squared error on the whole testing data set to be 15.96319.

Here are my results on the testing dataset:

<table>
<col width="33%" />
<col width="33%" />
<col width="33%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/part3/big_test20.jpg" alt="Nose Correctly Detected" /></p>
<p>Face Detection</p></td>
<td align="left"><p><img src="outputs/part3/big_test7.jpg" alt="Nose Correctly Detected" /></p>
<p>Face Detection</p></td>
<td align="left"><p><img src="outputs/part3/big_test13.jpg" alt="Nose Correctly Detected" /></p>
<p>Face Not Detection</p></td>
</tr>
</tbody>
</table>


I then tested my model on 3 personal images. I got them mostly right, the offset is most likely due to either a not perfect model, or variations in my cropping when passing the images through the neural net.

<table>
<col width="33%" />
<col width="33%" />
<col width="33%" />
<tbody>
<tr class="odd">
<td align="left"><p><img src="outputs/part3/morgan1result.jpg" alt="Nose Correctly Detected" /></p>
<p>Face Detection</p></td>
<td align="left"><p><img src="outputs/part3/testresult.jpg" alt="Nose Correctly Detected" /></p>
<p>Face Detection</p></td>
<td align="left"><p><img src="outputs/part3/face2result.jpg" alt="Nose Correctly Detected" /></p>
<p>Face Detection</p></td>
</tr>
</tbody>
</table>


