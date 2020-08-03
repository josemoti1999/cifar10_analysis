import matplotlib.pyplot as plt
list_x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]


plt.figure(1)
plt.plot(list_x,listreg1_y1,'-g',label="decay=0")
plt.plot(list_x,listreg1_y5,'-k',label="decy=0.0005")
plt.plot(list_x,listreg1_y2,'-b',label="decay=0.001")
plt.plot(list_x,listreg1_y4,'-y',label="decay=0.005")
plt.plot(list_x,listreg1_y3,'-r',label="decay=0.01")


plt.title('Train set Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.savefig('trainloss.jpg')
plt.show()





plt.figure(2)
plt.plot(list_x,listreg1_a1,'-g',label="decay=0")
plt.plot(list_x,listreg1_a5,'-k',label="decay=0.0005")
plt.plot(list_x,listreg1_a2,'-b',label="decay=0.001")
plt.plot(list_x,listreg1_a4,'-y',label="decay=0.005")
plt.plot(list_x,listreg1_a3,'-r',label="decay=0.01")

plt.title('Test set Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.savefig('testaccuracy.jpg')
plt.show()




plt.figure(3)
plt.plot(list_x,listreg1_ya1,'-g',label="decay=0")
plt.plot(list_x,listreg1_ya5,'-k',label="decay=0.0005")
plt.plot(list_x,listreg1_ya2,'-b',label="decay=0.001")
plt.plot(list_x,listreg1_ya4,'-y',label="decay=0.005")
plt.plot(list_x,listreg1_ya3,'-r',label="decay=0.01")


plt.title('Train set Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy_train')
plt.legend(loc='best')
plt.savefig('trainaccuracy.jpg')
plt.show()
