if __name__=='__main__':
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    
    (a,al)=test(testloader, model, criterion)
    print(a)
    print(al)

    



   
    