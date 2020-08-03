if __name__=='__main__':
    def test_final(image,model):
        with torch.no_grad():
            if torch.cuda.is_available():
                image= image.cuda()        
                (outputs, intermediate) = net(image)
                    
        return intermediate
    for i in range(10):
    
        plt.figure(figsize=(2,2))
        img=np.transpose(find_images[mv_index_2[i]].cpu().numpy(),(1,2,0))
        plt.imshow(img)
        plt.title("Image"+str(i))
        plt.savefig('Image'+str(i)+'.jpg')
        plt.show()
        intermediate_1=test_final(find_images[mv_index_2[i]].unsqueeze(0),net)
        intermediate_1=intermediate_1.cpu().numpy()
        plt.figure(figsize=(2,2))
        plt.imshow(intermediate_1[0,0],cmap='autumn')
        plt.title("Image patch L1F3-"+str(i))
        plt.savefig('Image patch L1F3-'+str(i)+'.jpg')
        plt.show()
        
    