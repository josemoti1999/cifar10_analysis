import matplotlib.pyplot as plt
list_x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]


plt.figure(1)

plt.plot(list_x,listreg1_y6,'-c',label="Adagrad_decay=0.001")
plt.plot(list_x,listreg1_y7,'-m',label="Adagrad_decay=0.01")
plt.plot(list_x,listreg1_y9,'-y',label="Adagrad_decay=0.02")
plt.plot(list_x,listreg1_y8,'-r',label="Adagrad_decay=0.05")


plt.title('Train set Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.savefig('trainloss2.jpg')
plt.show()





plt.figure(2)

plt.plot(list_x,listreg1_a6,'-c',label="Adagrad_decay=0.001")
plt.plot(list_x,listreg1_a7,'-m',label="Adagrad_decay=0.01")
plt.plot(list_x,listreg1_a9,'-y',label="Adagrad_decay=0.02")
plt.plot(list_x,listreg1_a8,'-r',label="Adagrad_decay=0.05")

plt.title('Test set Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.savefig('testaccuracy2.jpg')
plt.show()




plt.figure(3)
plt.plot(list_x,listreg1_ya6,'-c',label="Adagrad_decay=0.001")
plt.plot(list_x,listreg1_ya7,'-m',label="Adagrad_decay=0.01")
plt.plot(list_x,listreg1_ya9,'-y',label="Adagrad_decay=0.02")
plt.plot(list_x,listreg1_ya8,'-r',label="Adagrad_decay=0.05")

plt.title('Train set Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy_train')
plt.legend(loc='best')
plt.savefig('trainaccuracy2.jpg')
plt.show()
