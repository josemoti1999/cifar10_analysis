if __name__ ==  '__main__':
    
    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    import torch, os
    import torchvision
    import torchvision.transforms as transforms
    from tqdm import tqdm
    transform = transforms.Compose([transforms.ToTensor(),])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    ########################################################################
    # Define a Convolution Neural Network
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Copy the neural network from the Neural Networks section before and modify it to
    # take 3-channel images (instead of 1-channel images as it was defined).
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    # <<<<<<<<<<<<<<<<<<<<< EDIT THE MODEL DEFINITION >>>>>>>>>>>>>>>>>>>>>>>>>>
    # Try experimenting by changing the following:
    # 1. number of feature maps in conv layer
    # 2. Number of conv layers
    # 3. Kernel size
    # etc etc.,
    
    num_epochs = 15
    learning_rate = 0.001
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
            self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
            self.conv3=nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3)
            self.pool3=nn.MaxPool2d(kernel_size=2,stride=2)
            #self.conv4=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5)
            #self.pool4=nn.MaxPool2d(kernel_size=2,stride=2)
            self.fc1 = nn.Linear(in_features=5*5*256, out_features=1000)
            self.fc2 = nn.Linear(in_features=1000, out_features=400)
            self.fc3 = nn.Linear(in_features=400, out_features=64)
            self.fc4 = nn.Linear(in_features=64, out_features=10)
    
        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            #x = self.pool4(F.relu(self.conv4(x)))
            x = x.view(-1, 5*5*256)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    ################### DO NOT EDIT THE BELOW CODE!!! #######################
    
    net = Net()
    
    # transfer the model to GPU
    if torch.cuda.is_available():
        net = net.cuda()
    
    ########################################################################
    # Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.
    
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
    ########################################################################
    # Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    
    def train(epoch, trainloader, optimizer, criterion):
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader), 0):
            # get the inputs
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
        print('epoch %d training loss: %.3f' %
                (epoch + 1, running_loss / (len(trainloader))))
        y=running_loss/len(trainloader)
        return y
        
    ########################################################################
    # Let us look at how the network performs on the test dataset.
    
    def test(testloader, model):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()        
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                                        100 * correct / total))
        a=100*correct/total
        return a
    
    
    def classwise_test(testloader, model):
    ########################################################################
    # class-wise accuracy
    
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()        
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
    
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
    
    
    print('Start Training')
    os.makedirs('./models', exist_ok=True)
    list7_y1=[]
    list7_a1=[]
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('epoch ', epoch + 1)
        y=train(epoch, trainloader, optimizer, criterion)
        list7_y1.append(y)
        a=test(testloader, net)
        list7_a1.append(a)
        classwise_test(testloader, net)
        torch.save(net.state_dict(), './models/model-'+str(epoch)+'.pth')
        print('loss'+str(list7_y1))
        print('accuracy_test'+str(list7_a1))
    print('Finished Training')
