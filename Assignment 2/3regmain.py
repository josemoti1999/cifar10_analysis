if __name__=='__main__':
    
    num_epochs = 15
    learning_rate = 0.001 
    
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate,weight_decay=0.0005,momentum=0.9)
    optimizer = optim.Adagrad(net.parameters(), lr=learning_rate,weight_decay=0.009)
    
      
    print('Start Training')
    os.makedirs('./models_reg10', exist_ok=True)
    listreg1_y10=[]
    listreg1_ya10=[]
    listreg1_a10=[]
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('epoch ', epoch + 1)
        (y,ya)=train_new(epoch, trainloader, optimizer, criterion)
        listreg1_y10.append(y)
        listreg1_ya10.append(ya)
        a=test(testloader, net)
        listreg1_a10.append(a)
        torch.save(net.state_dict(), './models_reg10/model-'+str(epoch)+'.pth')
        print('loss'+str(listreg1_y10))
        print('accuracy_train'+str(listreg1_ya10))
        print('accuracy_test'+str(listreg1_a10))
    print('Finished Training')