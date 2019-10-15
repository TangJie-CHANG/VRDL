#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim  
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import math
import torchvision.models as models


# In[2]:


TRAINING_PATH = 'dataset/dataset/train/'
TESTING_PATH = 'dataset/dataset/test/'


# In[3]:


def load_split_train_test(datadir, valid_size=.2):
    train_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_data = torchvision.datasets.ImageFolder(datadir, transform=train_transforms)
    valid_data = torchvision.datasets.ImageFolder(datadir, transform=valid_transforms)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    
    from torch.utils.data.sampler import SubsetRandomSampler
    
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=16)
    validloader = torch.utils.data.DataLoader(valid_data,
                   sampler=valid_sampler, batch_size=16)
    return trainloader, validloader


trainloader, validloader = load_split_train_test(TRAINING_PATH, .2)
print(trainloader.dataset.classes)
print(validloader.dataset.classes)


# In[4]:


def show_batch(imgs):
    grid = utils.make_grid(imgs,nrow=5)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')


for i, (batch_x, batch_y) in enumerate(trainloader):
    if i < 16:
        print(i, batch_x.size(), batch_y.size())

        show_batch(batch_x)
        plt.axis('off')
        plt.show()


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnext50_32x4d(pretrained=True)
model = model.cuda()
print(model)


# In[20]:


optimizer = optim.SGD(model.parameters(), lr=0.000001, momentum=0.9)
criterion = nn.CrossEntropyLoss()


# In[24]:


epochs = 10
steps = 0
running_loss = 0
print_every = 200
train_losses, valid_losses = [], []


for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        #logps = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model(inputs)
                    #logps = model.forward(inputs)
                    batch_loss = criterion(output, labels)
                    valid_loss += batch_loss.item()
                    
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))                    
            print(f"Epoch {epoch+1}/{epochs}. "
                  f"Train loss: {running_loss/print_every:.3f}. "
                  f"valid loss: {valid_loss/len(validloader):.3f}. "
                  f"valid accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
            
#torch.save(model, 'aerialmodel.pth')


# In[31]:


# torch.save(model, 'model_93942')


# In[21]:


# torch.save(model, 'model_93750')


# In[37]:


# torch.save(model, 'model_94711')


# In[49]:


# torch.save(model, 'model_94903')


# In[17]:


# torch.save(model, 'model_95192')


# In[26]:


# torch.save(model, 'model_95480')


# In[10]:


def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')
    image = loader(image).float()
    #image = torch.tensor(image, requires_grad=True)
    image.clone().detach().requires_grad_(True)
    image = image.unsqueeze(0)
    return image

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[11]:


category = {
 0:'bedroom',
 1:'coast',
 2:'forest',
 3:'highway',
 4:'insidecity',
 5:'kitchen',
 6:'livingroom',
 7:'mountain',
 8:'office',
 9:'opencountry',
 10:'street',
 11:'suburb',
 12:'tallbuilding'}


# In[25]:


model.eval()

predict_df = pd.read_csv('sameple_submission.csv')


for i in range(1040):
    print('{:04d}'.format(i))
    img = image_loader(data_transforms, TESTING_PATH + '\image_'+'{:04d}'.format(i)+'.jpg')
    img = img.to(device)
    #var_image = Variable(img)

    output = model(img)
    prediction = int(torch.max(output.data, 1)[1])#.numpy())
    #prediction = int(torch.max(F.softmax(output).cpu(), 1)[1].numpy())
    
    print(prediction)
    
    predict_df.loc[i,'label'] = category[prediction]
    
predict_df.to_csv('out_resnext50_batch16.csv', index=False)


# In[ ]:




