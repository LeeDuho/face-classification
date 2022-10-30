#classify the face using the trained model

import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

#device= torch.device("cuda:0" if torch.cuda.is_available() else"cpu")


tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
])

  #'cat': 0, 'dog': 1, 'horse': 2, 'monkey': 3
train_data = torchvision.datasets.ImageFolder('./processed_data', transform=tf)
test_data = torchvision.datasets.ImageFolder('./processed_test_data', transform=tf)

print(train_data.class_to_idx)
#print intformation and target of first data in train_data 
print(train_data.imgs[0])
print(train_data.targets[0])
print(train_data.classes[0])
print("---------------------------------")
print(train_data.imgs[-1])
print(train_data.targets[-1])
print(train_data.classes[-1])


loader = DataLoader(train_data, batch_size=8, shuffle=True)
tdloader = DataLoader(test_data, batch_size=8, shuffle=False)

# VGG모델 불러오기, 사용
model = torchvision.models.vgg16(pretrained=True)
#model = torchvision.models.vgg16(pretrained=True).to(device)
print(model)

model.classifier[3] = nn.Linear(4096,1024)  #vgg16의 마지막 fully connected layer의 output을 4개로 바꿔줌
model.classifier[6] = nn.Linear(1024,4)

print(model)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_func = nn.CrossEntropyLoss()


for epoch in range(2): 
    print('epoch: ', epoch)

    model.train()
    pbar = tqdm(enumerate(loader), total=len(loader))
    for i, (imgs, labels) in pbar:
        
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        pbar.set_description("Epoch: {}, Loss: {:.4f}".format(epoch, loss.item()))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for imgs, labels in tdloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

#save model
torch.save(model.state_dict(), './model.pth')

model.eval()
with torch.no_grad():
    for imgs, labels in tdloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        #print each image's name and predicted class
        for i in range(len(outputs)):
            print("-------------------------------------")
            print('name: ', test_data.imgs[i][0], 'class: ', outputs[i].argmax().item())
            #outputs[i]의 각각의 확률 출력
            print('cat: ', outputs[i][0].item(), 'dog: ', outputs[i][1].item(), 'horse: ', outputs[i][2].item(), 'monkey: ', outputs[i][3].item())
            print("-------------------------------------")


# Path: face_classification.py

# print('finished')
# print('save model')
# torch.save(model, './model.pth')
# print('saved')
# print(model.state_dict().keys())  #모델의 weight들을 확인
# print(model.parameters()) #모델의 weight들을 확인

# # model.eval()
# print('load model')
# model = torch.load('./model.pth')

