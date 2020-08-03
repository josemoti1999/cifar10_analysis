    
if __name__ == '__main__':
    from torchvision import datasets, transforms, models
    from torch.nn import Linear, Conv2d, MaxPool2d, AvgPool2d, ReLU
    

    
    
    model = models.resnet18(pretrained=False)
    
    model.conv1=Conv2d(3,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
    model.maxpool=MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)
    
        
    model.fc = Linear(512, 10)
    
    if torch.cuda.is_available():
      model.cuda()
      
      
    