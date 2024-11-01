###############################
#  Learning how to train and use use a neural network
#  Start with some example code and modify it for our purposes
#  This code trains up an image recognition model.
#  It uses the CIFAR 10 image set for training and testing
#  It then saves the trained model for later use.
#
#  Note: this code is modified from the example code located at:
#  https://www.kaggle.com/code/vikasbhadoria/cifar10-high-accuracy-model-build-on-pytorch/notebook
#
###############################


import torch
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# Use Nvidia cuda for training processing if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set some neural network parameters
channel_width_1 = 3  # 3 color planes
channel_width_2 = 16  # 16 wide initially
kernel_num = 3
batch_size_num = 100

# process the images to prepare them for the training : resize, flip, set rotation, colors and then convert to tensor and normalize the image
transform_train = transforms.Compose(
    [transforms.Resize((32, 32)),  # rResize images to 32x32 for training
     transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(10),
     transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
training_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                    transform=transform_train)
validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size_num,
                                              shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size_num, shuffle=False)


# We need to convert the images to numpy arrays as tensors are not compatible with matplotlib.
def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy() # This process will happen in normal cpu.
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image

# 10 classes of the images in the CIFAR dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 
dataiter = iter(training_loader) # converting our train_dataloader to iterable so that we can iter through it.
#images, labels = dataiter.next() #going from 1st batch of 100 images to the next batch
images, labels = next(dataiter)
fig = plt.figure(figsize=(25, 4))

# We plot 20 images from our train_dataset
for idx in np.arange(20):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[idx])) #converting to numpy array as plt needs it.
  ax.set_title(classes[labels[idx].item()])



class LeNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(channel_width_1, channel_width_2, kernel_num, 1, padding=1) # input is color image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting, stride is 1 , padding is 1 extract all edge features.
      self.conv2 = nn.Conv2d(channel_width_2, channel_width_2*2, kernel_num, 1, padding=1) # We double the feature maps for every conv layer as in pratice it is really good.
      self.conv3 = nn.Conv2d(channel_width_2*2, channel_width_2*4, kernel_num, 1, padding=1)
      # below 4x4x64, original out features for next line and 3rd line was 500
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
      #print(x.shape)
      x = x.view(-1, 4*4*64) # flatten our images to 1D to input it to the fully connected layers was 4*4*64
      x = F.relu(self.fc1(x))
      x = self.dropout1(x) # Applying dropout b/t layers which exchange highest parameters. This is a good practice
      x = self.fc2(x)
      return x

model = LeNet().to(device) # run our model on cuda GPU for faster results
model

criterion = nn.CrossEntropyLoss() # same as categorical_crossentropy loss used in Keras models which runs on Tensorflow
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # fine tuned the lr

net = LeNet()

epochs = 25   # Experimentation has shown that model accuracy doesn't improve much past 25
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs):  # train for the given number of epochs specified above

    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0

    for inputs, labels in training_loader:
        inputs = inputs.to(device)  # input to device as our model is running in mentioned device.
        labels = labels.to(device)
        outputs = model(inputs)  # every batch of 100 images are put as an input.
        loss = criterion(outputs, labels)  # Calc loss after each batch i/p by comparing it to actual labels.

        optimizer.zero_grad()  # setting the initial gradient to 0
        loss.backward()  # backpropagating the loss
        optimizer.step()  # updating the weights and bias values for every single step.

        _, preds = torch.max(outputs, 1)  # taking the highest value of prediction.
        running_loss += loss.item()
        running_corrects += torch.sum(
            preds == labels.data)  # calculating te accuracy by taking the sum of all the correct predictions in a batch.

    else:
        with torch.no_grad():  # we do not need gradient for validation.
            for val_inputs, val_labels in validation_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)

        epoch_loss = running_loss / len(training_loader)  # loss per epoch
        epoch_acc = running_corrects.float() / len(training_loader)  # accuracy per epoch
        running_loss_history.append(epoch_loss)  # appending for displaying
        running_corrects_history.append(epoch_acc)

        val_epoch_loss = val_running_loss / len(validation_loader)
        val_epoch_acc = val_running_corrects.float() / len(validation_loader)
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc)
        # show how well our model is training at each epoch
        print('epoch :', (e + 1))
        print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
        print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))


# now that we've spent all that time training our model, lets save it so we can use it later without going through all the retraining
PATH = './cifar_net_LeNet_Model.pth'
torch.save(model.state_dict(), PATH)


print ("Trained model file saved!")

