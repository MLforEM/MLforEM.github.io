## Machine Learning for Electron Microscopy

This page can be used as a practical guide for unsupervised machine learning for image segmentation with electron microscopy. 

### Here is a way to display a title

```markdown
Then i can put stuff in the cells

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

```

### Add more headers

Add more text

### Header again

More text.

## Jupyter Notebooks

### Drift Correction

During all electron microscopy experiments the sample stage drifts. The most prominent reason for drift is high temperature but it can also be due to vibrations in the room/microscope, or other reasons. Most of the time this is not a problem, however sample drift becomes a big issue when exposure times are long and the sample is held at a high temperature. Correcting for drift is vital when looking at images of the same size with a focus on the position of objects in the images. Drift can be seen as consistent motion through a series of images. This means that the whole image, and hence, all the particles, move in the same direction. However, the direction of motion can change throughout the experiment. We know it is the images and not the particles moving because the particles always move the same amount in the same direction. 

Drift correction is extremely important because if it is not done correctly it can cause problems in all future notebooks. Without proper drift correction we are not able to track particles and no information can be gained from the datasets. At the end of this notebook we check the drift correction by making a movie of the cropped images. We do this by looking for black borders on each frame. 

[Drift Correction Jupyter Notebook](https://colab.research.google.com/drive/13LBYQRLUFVsE8hW9RRT9hFBgD9oI4ruV?usp=sharing)

### Dataset Prep

Now that drift correction is complete, we can prepare the image to feed through the machine learning model.  This notebook will demonstrate cropping images, rotating images, and making new arrays of images. The goal here is to take a small number of full images and produce a large number of unique training images.

In many cases, deep learning is not possible or produces inaccurate results due to a lack of available training data. Data availability can often be limited by the time and expense required to manually process and extract important information from images. On the other hand, in image analysis cases where ample data is provided, manually providing ground truth data can be very tedious and expensive. In situ Transmission Electron Microscopy provides a prime example of this, where modern direct detection cameras can collect hundreds or thousands of images per second yet quantifying the information in these images is extremely difficult. Convolutional neural networks have been shown to be extremely effective for segmenting electron microscopy images, but the amount of data required for training can often be prohibitive. By preparing a training set with built-in data augmentation, we can reduce the initial cost of developing a machine learning model, and significantly decrease the time required to move from raw data to fully segmented images.

[Dataset Prep Jupyter Notebook](https://colab.research.google.com/drive/1xVKG8MZAmSDpgJZZkyJlQBlwUmFXwR_n?usp=sharing)

### Training

This notebook takes the dataset we created and trains the machine learning model. 

[Training Jupyter Notebook](https://colab.research.google.com/drive/1RJFv8rTUo77LzvbZ63_MuMUSVp2oDGx6?usp=sharing)

### Inference

Once an ML model is trained, we can use it for segmentation of the original images in the dataset.  This process, of applying a trained model to new data, is called inference.

[Inference Jupyter Notebook](https://colab.research.google.com/drive/1RYvPYfCiz35TggzieT-r3Q9Uosfmmvde?usp=sharing)



