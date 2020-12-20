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

### Drift Correction

During all electron microscopy experiments the sample stage drifts. The most prominent reason for drift is high temperature but it can also be due to vibrations in the room/microscope, or other reasons. Most of the time this is not a problem, however sample drift becomes a big issue when exposure times are long and the sample is held at a high temperature. Correcting for drift is vital when looking at images of the same size with a focus on the position of objects in the images. Drift can be seen as consistent motion through a series of images. This means that the whole image, and hence, all the particles, move in the same direction. However, the direction of motion can change throughout the experiment. We know it is the images and not the particles moving because the particles always move the same amount in the same direction. Drift correction is extremely important because if it is not done correctly it can cause problems in all future notebooks. Without proper drift correction we are not able to track particles and no information can be gained from the datasets. At the end of this notebook we check the drift correction by making a movie of the cropped images. We do this by looking for black borders on each frame. 

https://colab.research.google.com/drive/13LBYQRLUFVsE8hW9RRT9hFBgD9oI4ruV?usp=sharing
