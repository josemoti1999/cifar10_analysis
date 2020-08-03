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
        transforms.Resize((224,224), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224,224), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    


    indices=list(range(0, 5000))
    
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, sampler=torch.utils.data.SubsetRandomSampler(indices), shuffle=False, num_workers=2)

    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(len(trainloader))
    print(len(testloader))
