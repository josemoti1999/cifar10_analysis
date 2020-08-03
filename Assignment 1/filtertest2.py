def imshow_filter(img,p1,p2,i):
    img=img/2+0.5
    npimg=img.cpu().numpy()
    p1=classes[p1.cpu().numpy()]
    p2=classes[p2.cpu().numpy()]
    plt.figure(figsize=(2,2))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title("p1="+str(p1)+", p2="+str(p2))
    plt.savefig('Filter_dataset'+str(i)+'.jpg')
    plt.show()
            
count=[]
plane=[]
car=[]
bird=[]
cat=[]
deer=[]
dog=[]
frog=[]
horse=[]
ship=[]
truck=[]
#print(find_change_1)
#print(find_change_2)

#for table values onlu



for i in range(10000):
    #if find_change_2[i]==1 and find_change_1[i]==0:
    if find_change_2[i]==0 and find_change_1[i]==1:
    #if find_change_1[i]==1:
    #if find_change_2[i]==1:
        count.append(i)
        if find_pred_2[i]==0:
            plane.append(i)
        if find_pred_2[i]==1:
            car.append(i)
        if find_pred_2[i]==2:
            bird.append(i)
        if find_pred_2[i]==3:
            cat.append(i)
        if find_pred_2[i]==4:
            deer.append(i)
        if find_pred_2[i]==5:
            dog.append(i)
        if find_pred_2[i]==6:
            frog.append(i)
        if find_pred_2[i]==7:
            horse.append(i)
        if find_pred_2[i]==8:
            ship.append(i)
        if find_pred_2[i]==9:
            truck.append(i)
            
            
            
        
        
print(len(count))
print(len(plane))
print(len(car))
print(len(bird))
print(len(cat))
print(len(deer))
print(len(dog))
print(len(frog))
print(len(horse))
print(len(ship))
print(len(truck))



#for i in range(50):
    #imshow_filter(find_images[count[i]],find_pred_1[count[i]],find_pred_2[count[i]],i)
