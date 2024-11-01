# Image Detection

## A fully functional image dectection program using torchvision datasets, and LeNet machine learning to seperate and classify images.

### Disclaimer 
This application is for informational purposes only.

Welcome to the Image Detection program. The program will begin with a torchvision dataset that was given by the user. 
Then, the program will condensed the the size of the iamge into a 32x32 pixel image, allowing for quicker processing and recognition. 
After, the program uses supervised machine learning, which makes it improve on previous, wrong classifications, and improve. 
Last, the training model is saved, and now the user can give the program new images, which are classified correctly.

## For Developers: Comments are included in the code explaining functions and chunks of code. These will help explain how the program works.

## For Users and Developers: To run the program, many packages are required. Packages include: 


import torch

import torch.nn.functional as F

from torchvision import datasets,transforms

from torch import nn

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

After importing the latest versions of the packages, you should be able to run the code without errors. 




## If you find a bug or a possible improvement to this project, please submit an issue in the issues tab above. Thank you!
