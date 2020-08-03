if __name__ ==  '__main__':

    import torch, os
    import torchvision
    import torchvision.transforms as transforms
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    
    
    transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])



    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    
   
    
    def train_new(epoch, trainloader, optimizer, criterion):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print('epoch %d training loss: %.3f' %
                (epoch + 1, running_loss / (len(trainloader))))
        print('Accuracy of the network on the 50000 train images: %d %%' % (
                                        100 * correct / total))
        y=running_loss/len(trainloader)
        ya=100*correct/total
        
        return y,ya
    
    
    def test(testloader, model, criterion):
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()        
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        
        print('etest loss: %.3f' %
                (running_loss / (len(testloader))))
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                                        100 * correct / total))
        a=100*correct/total
        al=running_loss/len(testloader)
        
        
        
        return a,al