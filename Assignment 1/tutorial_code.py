import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def hello():
    a=np.random.randn(2,3)
    print("Helloo world")
    print(a)
    print(np.multiply(a,a))
if __name__ ==  '__main__':
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
    testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
    testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)
    classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    def imshow(img):
        img=img/2+0.5
        npimg=img.numpy()
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.show()
    dataiter=iter(trainloader)
    images,labels=dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            #self.conv1 = nn.Conv2d(3, 6, 5)
            #self.pool1 = nn.MaxPool2d(2, 2)
            #self.conv2 = nn.Conv2d(6, 16, 5)
            #self.pool2 = nn.MaxPool2d(2, 2)
            #self.conv3 = nn.Conv2d(16, 64, 5)
            #self.pool3 = nn.MaxPool2d(5, 1)
            #self.conv4 = nn.Conv2d(64, 128, 5)
            #self.pool4 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32*32*3, 120)
            self.fc2 = nn.Linear(120,84)
            self.fc3 = nn.Linear(84, 10)
            #self.fc4 = nn.Linear(32,10)
    
        def forward(self, x):
            #x = self.pool1(F.relu(self.conv1(x)))
            #x = self.pool2(F.relu(self.conv2(x)))
            #x = self.pool3(F.relu(self.conv3(x)))
            #x = self.pool4(F.relu(self.conv4(x)))
            x = x.view(-1, 32*32*3)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            #x = self.fc4(x)
            return x
    
    
    net = Net()
    import torch.optim as optim
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    list_x2=[]
    list_y2=[]
    for epoch in range(10):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                x_value=epoch+((i+1)/12500)
                list_x2.append(x_value)
                list_y2.append(running_loss/2000)
                #print("lis_x is"+str(list_x))
                #print("list_y is"+str(list_y))
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    print('Finished Training')   
    
    #training over
    
    
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))