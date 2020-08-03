# -*- coding: utf-8 -*-
"""
all occlusion datasets from previous file do not load it again temp.py
"""
if __name__=='__main__':
    import copy
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    net = Net()
    epoch=14
    net.load_state_dict(torch.load('./models/model-'+str(epoch)+'.pth'))
    net.eval().cuda()
    
    
    
    def imshow_occlusion(img):
        img=img/2+0.5
        npimg=img
        plt.figure(figsize=(3,3))
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.title("Image with occlusion")
        #plt.savefig('Occlusion_dataset.jpg')
        plt.show()
        
        
        
        
    def test_occlusion(image,model,m):
        with torch.no_grad():
            image = torch.from_numpy(image)
            image=image.unsqueeze(0)
            if torch.cuda.is_available():
                    image= image.cuda()
            print(image.shape)
            outputs = net(image)
            output=F.softmax(outputs)
            output=output[0,m]
        return output
    
    a=[]
    max_confidence=[]
    for i in range(20):
        a.append(dataset[i].cpu().numpy())
        imshow_occlusion(a[i])
        confidence=test_occlusion(a[i],net,int(i/2))
        max_confidence.append(confidence.cpu().numpy())
    print(max_confidence)
        

    confidence_matrix=np.random.randn(32,32)
    
    
    
    
    
    #for 3*3 window
    for m in range(20):
        for i in range(32):
            for j in range(32):
                b=copy.deepcopy(a)
                b[m][0:3,max((i-1),0):min((i+2),31),max((j-1),0):min((j+2),31)]=0
                confidence=test_occlusion(b[m],net,int(m/2))
                imshow_occlusion(b[m])
                print("Confidence is "+str(confidence.cpu().numpy()))
                confidence_matrix[i,j]=confidence.cpu().numpy()
        plt.figure(figsize=(2,2))
        plt.imshow(confidence_matrix,cmap='autumn')
        plt.title("Occlusion 3*3 image"+str(m))
        plt.legend(loc='best') 
        plt.savefig("Occlusion_3_3_graph "+str(m)+".jpg")
        plt.show()
        imshow_occlusion(a[m])
        
        


    
    #for 5&5 window
    for m in range(20):
        for i in range(32):
            for j in range(32):
                b=copy.deepcopy(a)
                b[m][0:3,max((i-2),0):min((i+3),31),max((j-2),0):min((j+3),31)]=0
                confidence=test_occlusion(b[m],net,int(m/2))
                imshow_occlusion(b[m])
                print("Confidence is "+str(confidence.cpu().numpy()))
                confidence_matrix[i,j]=confidence.cpu().numpy()
        plt.figure(figsize=(2,2))
        plt.imshow(confidence_matrix,cmap='autumn')
        plt.title("Occlusion 5*5 image"+str(m))
        plt.legend(loc='best') 
        plt.savefig("Occlusion_5_5_graph "+str(m)+".jpg")
        plt.show()
        imshow_occlusion(a[m])
        
        

    
    
   
    #for 7*7 window
    for m in range(20):
        for i in range(32):
            for j in range(32):
                b=copy.deepcopy(a)
                b[m][0:3,max((i-3),0):min((i+4),31),max((j-3),0):min((j+4),31)]=0
                confidence=test_occlusion(b[m],net,int(m/2))
                imshow_occlusion(b[m])
                print("Confidence is "+str(confidence.cpu().numpy()))
                confidence_matrix[i,j]=confidence.cpu().numpy()
        plt.figure(figsize=(2,2))
        plt.imshow(confidence_matrix,cmap='autumn')
        plt.title("Occlusion 7*7 image"+str(m))
        plt.legend(loc='best') 
        plt.savefig("Occlusion_7_7_graph "+str(m)+".jpg")
        plt.show()
        imshow_occlusion(a[m])
        
        

    
    #for 9*9 window
    for m in range(20):
        for i in range(32):
            for j in range(32):
                b=copy.deepcopy(a)
                b[m][0:3,max((i-4),0):min((i+5),31),max((j-4),0):min((j+5),31)]=0
                confidence=test_occlusion(b[m],net,int(m/2))
                imshow_occlusion(b[m])
                print("Confidence is "+str(confidence.cpu().numpy()))
                confidence_matrix[i,j]=confidence.cpu().numpy()
        plt.figure(figsize=(2,2))
        plt.imshow(confidence_matrix,cmap='autumn')
        plt.title("Occlusion 9*9 image"+str(m))
        plt.legend(loc='best') 
        plt.savefig("Occlusion_9_9_graph "+str(m)+".jpg")
        plt.show()
        imshow_occlusion(a[m])


    
    
