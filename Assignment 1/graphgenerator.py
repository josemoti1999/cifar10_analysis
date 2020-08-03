import matplotlib.pyplot as plt
list_x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
list3_y2=list2_y4
list3_a2=list2_a4
plt.figure(1)

plt.plot(list_x,list_y1,'-g',label="stride=1")
plt.plot(list_x,list6_y2,'-b',label="stride=2")
plt.plot(list_x,list6_y3,'-r',label="stride=4")
plt.plot(list_x,list6_y4,'-y',label="stride=6")
plt.plot(list_x,list2_y4,'-k',label="4 fc layers")
plt.title('Train set Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.savefig('plotconvlayers.jpg')
plt.show()
plt.figure(2)

plt.plot(list_x,list_a1,'-g',label="stride=1")
plt.plot(list_x,list6_a2,'-b',label="stride=2")
plt.plot(list_x,list6_a3,'-r',label="stride=4")
plt.plot(list_x,list6_a4,'-y',label="stride=6")
plt.plot(list_x,list2_a4,'-k',label="4 fc layers")
plt.title('Test set Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.savefig('plotconvlayersaccuracy.jpg')
plt.show()
