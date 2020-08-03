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


for i in range(100):
    
    if find_labels_2[mv_index_2[i]]==0:
        plane.append(i)
    if find_labels_2[mv_index_2[i]]==1:
        car.append(i)
    if find_labels_2[mv_index_2[i]]==2:
        bird.append(i)
    if find_labels_2[mv_index_2[i]]==3:
        cat.append(i)
    if find_labels_2[mv_index_2[i]]==4:
        deer.append(i)
    if find_labels_2[mv_index_2[i]]==5:
        dog.append(i)
    if find_labels_2[mv_index_2[i]]==6:
        frog.append(i)
    if find_labels_2[mv_index_2[i]]==7:
        horse.append(i)
    if find_labels_2[mv_index_2[i]]==8:
        ship.append(i)
    if find_labels_2[mv_index_2[i]]==9:
        truck.append(i)



print("100 top images")
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
            
            