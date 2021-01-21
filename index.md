# Tutorial on Unsupervised Image Segmentation for Electron Microscopy

*Leena Vyas, James P. Horwath, and Eric A. Stach*

In situ transmission electron microscopy (TEM) allows scientists to observe dynamic processes in real time.  By combining high resolution imaging which is characteristic of TEM with rapid image capture capabilities enabled by modern direct detection cameras, time-dependent processes can be observed and studied with unprecedented levels of detail.  However, experimental advances do not come without challenges.  The amount of data generated in experiments which record up to thousands of images per second prohibits traditional manual analysis.  Without advances in image processing to extract physical data from TEM images, full experimental capabilities cannot be realized.
Building on advances in the fields of computer vision and machine learning, the goal of this website is to demonstrate how deep learning can be applied to the processing of scientific images.  Readers will be introduced to important preprocessing steps to remove experimental artifacts, and to fundamental design aspects for building a convolutional neural network from scratch.  Interactive jupyter notebooks are used to illustrate individual steps in image preparation, building and training a convolutional neural network to perform semantic segmentation of TEM images, and evaluating the results on real images.  

Our goal is to provide a foundation in practical machine learning for microscopists with minimal background in machine learning or computer programming.  The principles discussed here follow recent work which may be found [here](https://www.nature.com/articles/s41524-020-00363-x)[1].


## Introduction to Machine Learning and Unsupervised Segmentation
In this section, we will familiarize readers with concepts relevant to image processing and machine learning, but this is not meant to be a thorough introduction to the topic - many sources are available for in-depth [instruction](https://www.deeplearningbook.org/)[2].

Machine learning can be considered a case multidimensional regression, where optimization principles are applied to reveal relationships between many, potentially disparate, features of  a data set.  As more data is collected, these learned relationships can be applied to make predictions and inferences about the behavior of the system in question.

There are two main types of machine learning approaches: supervised and unsupervised.
In supervised learning, a ground truth label is available for the training data so that the parameters of the machine learning model can be optimized with respect to known outcomes.  This approach is common for classification or regression tasks where unique labels are known, or the value of interest can be observed experimentally.  One application of this type of learning is classifying images of handwritten numbers - the well known MNIST dataset provides thousands of images of handwritten numbers along with digital labels of what number each image [corresponds](http://yann.lecun.com/exdb/mnist/)[3] to.  A model is used to encode information from each image and output one of the ground-truth labels which best fits the data in question.
In unsupervised learning, features of the data are often reduced to a simplified form such that the training data can be grouped, or clustered, based on the similarities between training examples. A well known method of unsupervised learning is principal component analysis, where the dimensionality (a proxy for complexity) of the data is reduced based on statistical similarity between available observations to provide new compound features which produce a simplified, yet accurate, representation of the data. For example, features of an email, such as the address of the sender, frequency, length of messages, and number of times a user accesses the email, can be used as data to group emails in terms of importance.  In this way, spam can be weeded out without a person having to manually label spam messages. 

<img width="743" alt="Screen Shot 2021-01-21 at 2 16 36 PM" src="https://user-images.githubusercontent.com/76077037/105400773-62efdf00-5bf3-11eb-9f06-3816a3199827.png">

**Figure 1.**  Diagram showing the processes for supervised and unsupervised learning using a convolutional neural network.  In supervised learning (red line), the network’s output is compared to a ground-truth segmentation, and layer weights are updated against the difference between the model output and the true pixel labels.  In unsupervised segmentation (blue line), weight are updated so that the model output reproduces the input image.

The most important information in images is often stored in the spatial relationships between objects. Therefore, a method for encoding image data which includes feature arrangement is imperative.  This is achieved using convolutional filters, which operate on regions of images as a rolling window.  Processing the information obtained by these convolutional windows through several layers of abstract, non-linear connections produces a convolutional neural network (CNN) - one of the most common machine learning methodologies applied to image processing.  By deliberately sampling different areas of an image and applying convolutional filters, spatial information can be encoded and used to classify individual pixels of an image.  For supervised learning, ground truth images, where each pixel is assigned to a given class, are available for each training image, and the CNN works to reproduce the classification with the correct label.  In unsupervised learning, the same CNN can be used to reproduce the input image rather than a known ground-truth (which is unavailable in many cases).  Here, the number of tunable parameters is severely constrained and the CNN cannot fully reproduce the input image but can output a simplified version that can be easily segmented with minimal further processing steps.

Because creating a labeled training set can often be difficult and time consuming, unsupervised learning provides clear benefits. Therefore, unsupervised segmentation of TEM images will be the focus of this tutorial.

## Processing Steps

### Drift Correction

In the first notebook, images are cropped to remove black borders caused by drift during the experiment. The most prominent reason for drift is high temperature but it can also be caused by vibrations in the room/microscope. Without proper drift correction, particles cannot be tracked and no information can be gained from the datasets. 

<img width="581" alt="Screen Shot 2021-01-11 at 5 55 19 PM" src="https://user-images.githubusercontent.com/76077037/104248606-d34e7180-5437-11eb-878e-0356c5657ff9.png">

**Figure 2.** The contours show particle outlines as a function of time. Blue contours and the image show the initial position of particles, while the red show particle positions after 250s at 900C. Particles are seen to move towards the lower left of the frame. Drift is seen as a consistent motion through a series of images. This means that the whole image, and hence, all the particles, move in the same direction. Small, random jumps in the microscope stage can cause discontinuities in the drift. These jumps are corrected using cross-correlation.

[Drift Correction Jupyter Notebook](https://colab.research.google.com/drive/13LBYQRLUFVsE8hW9RRT9hFBgD9oI4ruV?usp=sharing)

### Dataset Prep

The second notebook prepares the images to go through the machine learning model. It uses data augmentation, including cropping and rotation, to increase the number of unique images that can be used for training. This method saves resources when collecting images by requiring fewer initial raw images. 

<img width="581" alt="Screen Shot 2021-01-11 at 5 16 43 PM" src="https://user-images.githubusercontent.com/76077037/104244651-e0b42d80-5430-11eb-92e8-a6301354a4c7.png">

**Figure 3.** Individual images can be broken into segments for data augmentation, each of which can be used as a unique example in the training set. Care must be taken to ensure that each sub-image still closely resembles the larger image so that the trained model is not biased by artifacts from data augmentation.

[Dataset Prep Jupyter Notebook](https://colab.research.google.com/drive/1xVKG8MZAmSDpgJZZkyJlQBlwUmFXwR_n?usp=sharing)

### Training

The third notebook is used for training. Raw training images from the previous notebook are fed through the model to produce images close to ground truth. Practice datasets will be included, but the code can be easily adjusted to work on any images. The images are fed through the model in batches adjusting layer weights based on the average performance. 

[Training Jupyter Notebook](https://colab.research.google.com/drive/1RJFv8rTUo77LzvbZ63_MuMUSVp2oDGx6?usp=sharing)

### Inference

In the fourth notebook, the remaining images are fed through the trained machine learning model. Chosen weights from training are used for segmentation of the images. The output allows for further analysis of the images including number of particles per image, size of particles, interactions between particles to determine coarsening mechanisms, etc. 

[Inference Jupyter Notebook](https://colab.research.google.com/drive/1RYvPYfCiz35TggzieT-r3Q9Uosfmmvde?usp=sharing)

## Outcomes

To work with new data, a few things must be edited. There are green lines in the above notebooks starting with ACTION showing where these are. Critical elements that need to be changed are the loading of the dataset, cropping dimensions for drift correction, and the number of images in the dataset. 

## Application:  Quantification of Nanoparticle Evolution using in situ Transmission Electron Microscopy

Nanoscale microstructures are present in many functional materials; due to their high surface to volume ratio, the relatively high surface energy of nanomaterials can be utilized for purposes ranging from the prohibition of crack propagation to enhancing electronic properties of materials.  In a particularly important case, supported metal particles are commonly used as industrial catalysts.  While their exceptional activity stems from the high surface energy and under coordinated surface structures, the same surface energy drives nanoparticles to grow in order to become more stable.  Several mechanisms and theories have been suggested to describe nanoparticle growth and interaction under a variety of conditions, but the difficulty of real-time observation and accurate characterization of these systems at the nanoscale have made verification of these mechanisms impossible.  However, using quantitative electron microscopy to understand the evolution of supported nanoparticles at high temperatures attempts to address this issue.

### Experimental Details:
The included sample images show gold nanoparticles supported on silicon nitride supports compatible with in situ heating TEM experiments.  Samples were heated in the TEM column to temperatures ranging from 700 - 900C and annealed over the course of an hour.  Images of each sample were collected every five seconds to track changes through time.

The following tutorial details the preparation of images, segmentation and measurement of particles using unsupervised convolutional autoencoders, and extraction of physically important information from image data.

<img width="581" alt="Screen Shot 2021-01-10 at 6 33 12 PM" src="https://user-images.githubusercontent.com/76077037/104138483-53120880-5372-11eb-9844-534d476ddd12.png">

**Figure 4.** Demonstration of the type of data that can be extracted from in situ TEM experiments with the help of machine learning.  A.) the first image of the time series. B.) the outline of each particle (contained in the box in A) at each point in time.  Changes in particle morphology can indicate the presence of a specific coarsening mechanism. C.) collects particle size distributions for the system at each time step in the experiment.  This type of characterization is common in traditional studies of nanoparticle growth, while the characterization of individual particles, captured in D), is one example of the augmented information enabled by augmented image segmentation.

References:

[1]. Horwath, Zakharov, Megret, and Stach, “Understanding important features of deep learning models for segmentation of high-resolution transmission electron microscopy images,” npj Computational Materials 6 (2020).

[2]. Goodfellow, Bengio, Courville, Deep Learning, MIT Press (2016).

[3].  LeCun, Cortes, and Burges, The MNIST database of handwritten digits.
