if __name__=='__main__':
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
            out=x
            x = x.view(-1, 5*5*256)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x,out
    
    
    def test_4(testloader, model):
        correct = 0
        total = 0
        find_change=[]
        find_pred=[]
        with torch.no_grad():
            mv=[]
            mvi=0
            mv_index=[]
            for data in tqdm(testloader):
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()        
                (outputs,intermediate) = net(images)
                _, predicted = torch.max(outputs.data, 1)
                
                for i in range(labels.size(0)):
                    max_value=torch.sum(intermediate[i])
                    max_value=max_value.cpu()
                    
                    if max_value>3800:
                        mv.append(max_value)
                        mv_index.append(mvi)
                    
                    mvi+=1
                    
                    
                    if predicted[i]==labels[i]:
                        find_change.append(1)
                        find_pred.append(predicted[i])
                    else:
                        find_change.append(0)
                        find_pred.append(predicted[i])
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                                        100 * correct / total))
        a=100*correct/total
        return a, find_change, find_pred, mv , mv_index
    
    
    
    
    def test_3(testloader, model):
        correct = 0
        total = 0
        find_images=[]
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()        
                (outputs, intermediate) = net(images)
                _, predicted = torch.max(outputs.data, 1)
                
                for i in range(labels.size(0)):    
                    find_images.append(images[i])
                    
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                                        100 * correct / total))
        a=100*correct/total
        return a,find_images
    
    
    
    
    with torch.no_grad():
        net = Net()
        epoch=14
        net.load_state_dict(torch.load('./models/model-'+str(epoch)+'.pth'))

        net.eval().cuda()
        #a=test_check(testloader,net)
        (a1,find_change_1, find_pred_1, mv, mv_index)=test_4(testloader,net)
        print(mv_index)
        print("\n\n"+str(len(mv_index)))
        print(mv)
        
        
        #(a,find_images)=test_3(testloader,net)
        #print(len(find_images))
        
        
        #classwise_test(testloader, net)
        print("Final accuracy is"+str(a1))
        
        
        
    
    
    
