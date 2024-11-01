#
#  Load our pre-trained model that we create with the previous code "Image Recongition Trainer.py"
#
#  Load some images from the internet that were not part of the CIFAR training or testing set.
#
#  Then transform them to the same 32x32 format as the CIFAR images and try to identify them
#
#  Note: this code is heavily modified from the example code located at:
#  https://www.kaggle.com/code/vikasbhadoria/cifar10-high-accuracy-model-build-on-pytorch/notebook
#
#


import torch
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# use nvidia graphics card for processing if available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
# We need to convert the images to numpy arrays as tensors are not compatible with matplotlib.
def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy() # This process will happen in normal cpu.
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image

# Different classes from CIPHAR 10 dataset.
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class LeNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1) # input is color image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting, stride is 1 , padding is 1 extract all edge features.
      self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1) # We double the feature maps for every conv layer as in pratice it is really good.
      self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
      self.fc1 = nn.Linear(4*4*64, 2000) # I/p image size is 32*32, after 3 MaxPooling layers it reduces to 4*4 and 64 because our last conv layer has 64 outputs. Output nodes is 500
      self.dropout1 = nn.Dropout(0.5)
      self.fc2 = nn.Linear(2000, 10) # output nodes are 10 because our dataset have 10 different categories
    def forward(self, x):
      x = F.relu(self.conv1(x)) #Apply relu to each output of conv layer.
      x = F.max_pool2d(x, 2, 2) # Max pooling layer with kernal of 2 and stride of 2
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv3(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*64) # flatten our images to 1D to input it to the fully connected layers
      x = F.relu(self.fc1(x))
      x = self.dropout1(x) # Applying dropout b/t layers which exchange highest parameters. This is a good practice
      x = self.fc2(x)
      return x





# Load previously trained model
PATH = './cifar_net_LeNet_WIP.pth'
net = LeNet()
net.load_state_dict(torch.load(PATH, weights_only=True))

images_to_test = ["grey_car.jpg", "red_car.jpg", "white_horse.jpg", "brown_horse.jpg", "black_cat.jpg", "orange_cat.jpg"]

filepath = r"data\my_images\\"

for test_image in images_to_test:

    filename = filepath + test_image

    img = Image.open(filename)
    plt.imshow(img)
    plt.show()

    img = transform(img)  # applying the transformations on new image as our model has been trained on these transformations
    plt.imshow(im_convert(img)) # convert to numpy array for plt
    plt.show()

    image = img.to(device).unsqueeze(0) # put inputs in device as our model is running there
    output = net(image)  # was model(images)
    _, pred = torch.max(output, 1)
    print(f"The image {test_image} has been identified as a: {classes[pred.item()]}")
