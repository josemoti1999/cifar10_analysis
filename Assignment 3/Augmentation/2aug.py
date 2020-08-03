if __name__ ==  '__main__':    


    def imshow(inp, title):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
    
    
    # Get a batch of training data
    # inputs contains 4 images because batch_size=4 for the dataloaders
    a=b=c=d=e=f=g=h=j=k=0
    n=0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        for i in range(len(labels)):
            n=n+1
            if labels[i]==0:
                    a=a+1            
            if labels[i]==1:
                    b=b+1
            if labels[i]==2:
                    c=c+1
            if labels[i]==3:
                    d=d+1
            if labels[i]==4:
                    e=e+1
            if labels[i]==5:
                    f=f+1
            if labels[i]==6:
                    g=g+1
            if labels[i]==7:
                    h=h+1
            if labels[i]==8:
                    j=j+1
            if labels[i]==9:
                    k=k+1

    print(a)
    print(b)
    print(c)
    print(d)        
    print(e) 
    print(f)
    print(g)
    print(h)
    print(j)
    print(k)
    print(n)
        
    #for i in range(128):
        #print(classes[labels[i]])
    
    # Make a grid from batch
    #out = torchvision.utils.make_grid(inputs)
    
    #imshow(out, title=[classes[x] for x in labels])