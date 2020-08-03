if __name__=='__main__':
    net = Net()
    epoch=14
    net.load_state_dict(torch.load('./models/model-'+str(epoch)+'.pth'))
    net.eval().cuda()
    classwise_test(testloader,net)
    a=test(testloader,net)
    #print("Accuracy="+str(a))
    #dataset,outputs,predicted,labels=occlusion_test(testloader,net)
    