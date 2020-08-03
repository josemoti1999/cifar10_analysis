def imshow_filter(img,l,p1,p2,i):
    img=img/2+0.5
    npimg=img.cpu().numpy()
    p1=classes[p1.cpu().numpy()]
    p2=classes[p2.cpu().numpy()]
    l=classes[l.cpu().numpy()]
    plt.figure(figsize=(2,2))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title("Label="+str(l))
    plt.savefig('Filter_dataset'+str(i)+'.jpg')
    plt.show()
    
    
for i in range(10):
    imshow_filter(find_images[mv_index_2[i]],find_labels_2[mv_index_2[i]],
                  find_pred_2[mv_index_2[i]],find_pred_2[mv_index_2[i]],i)
