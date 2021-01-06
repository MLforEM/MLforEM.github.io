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

## Processing Steps

### Drift Correction

In the first notebook, raw images are read from the proprietary format of the microscope camera and formatted as numpy arrays. This is necessary so that images can easily be edited and operated on using common python packages. These images are cropped to remove black borders caused by shifts during the experiment. 

[Drift Correction Jupyter Notebook](https://colab.research.google.com/drive/13LBYQRLUFVsE8hW9RRT9hFBgD9oI4ruV?usp=sharing)

### Dataset Prep

In the first notebook, raw images are read from the proprietary format of the microscope camera and formatted as numpy arrays. This is necessary so that images can easily be edited and operated on using common python packages. These images are cropped to remove black borders caused by shifts during the experiment. 

[Dataset Prep Jupyter Notebook](https://colab.research.google.com/drive/1xVKG8MZAmSDpgJZZkyJlQBlwUmFXwR_n?usp=sharing)

### Training

The third notebook is used for training. Raw images are fed through the model with the goal of producing images close to ground truth. Practice datasets will be included, but the code can be easily adjusted to work on any images. The images are fed through the model in batches adjusting layer weights based on the average performance. 

[Training Jupyter Notebook](https://colab.research.google.com/drive/1RJFv8rTUo77LzvbZ63_MuMUSVp2oDGx6?usp=sharing)

### Inference

In the fourth notebook,  the remaining images are fed through the trained machine learning model. Chosen weights from training are used for segmentation of the images. The output allows for further analysis of the images including number of particles per image, size of particles, interactions between particles to determine coarsening mechanisms etc. 

[Inference Jupyter Notebook](https://colab.research.google.com/drive/1RYvPYfCiz35TggzieT-r3Q9Uosfmmvde?usp=sharing)

## Outcomes

Make diagram.

In order to work with new data, a few things must be edited. There are green lines in the above notebooks starting with ACTION telling you where these are. Critical elements that need to be changed are the loading of the dataset, cropping dimensions for drift correction, and possibly the training model. 
