    
if __name__ == '__main__':
    for epoch in range(30):  # loop over the dataset multiple times
        print('epoch ', epoch + 1)
        (y,ya)=train_new(epoch, trainloader, optimizer, criterion)
        list_y10.append(y)
        list_ya10.append(ya)
        (a,al)=test(testloader, model, criterion)
        list_a10.append(a)
        list_al10.append(al)
        
        print('loss_train='+str(list_y10))
        print('accuracy_train='+str(list_ya10))
        print('accuracy_test='+str(list_a10))
        print('loss_test='+str(list_al10))
        
    print('Finished Training')
    torch.save(model.state_dict(), './models_transfer/5modelnew30.pth')