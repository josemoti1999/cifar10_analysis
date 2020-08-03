
import matplotlib.pyplot as plt
list_x=[]
for i in range (90):
  list_x.append(i+1)
  
  
loss_train=[7.011791468886158, 6.0872937419530375, 5.9287518863482855, 6.398459786954134, 5.746684042084247, 6.195538692157287, 6.1135745426577985, 6.382612374737439, 5.910499467874122, 6.168980515521506, 6.120649472526882, 5.979708051437612, 6.1463887014657335, 6.359533228227854, 5.873333876090282, 5.720704753685485, 5.827263959533418, 6.229377327672661, 6.114821671220043, 6.170203413804779, 5.910633063377322, 6.18108877074688, 6.06956627423806, 6.288884823279613, 6.183855706773451, 6.057008734749406, 6.053220070841367, 6.008975186311376, 6.177531463105965, 6.278481521264976, 2.803386669024787, 2.027786918308424, 1.8284006451096986, 1.7199913537715708, 1.6917318114844124, 1.6660373360299698, 1.6537915962126555, 1.6606004713746287, 1.65291906958041, 1.6519823775571936, 1.644545271268586, 1.6582052945480932, 1.654889434804697, 1.6571366369266949, 1.6591514748380618, 1.644946340404813, 1.6653198043403723, 1.6538646958977974, 1.6513688878330124, 1.6561056595019368, 1.6516128528453504, 1.6556355498940742, 1.6619444792837743, 1.6498826293994093, 1.654468206493446, 1.6496262324740514, 1.6415569068830642, 1.6527198117102504, 1.6603922578684813, 1.6600676168261281]
accuracy_train=[32.046, 33.836, 33.914, 33.382, 34.222, 33.318, 33.852, 33.746, 33.664, 33.862, 33.95, 33.756, 33.502, 33.444, 33.998, 34.29, 33.562, 33.72, 33.996, 33.754, 34.13, 33.748, 33.79, 33.366, 33.97, 34.138, 34.112, 33.532, 34.24, 33.884, 43.252, 44.196, 44.354, 44.674, 44.864, 44.966, 44.99, 44.484, 44.596, 44.782, 44.544, 44.736, 44.608, 44.586, 44.858, 44.726, 44.306, 44.658, 44.81, 44.556, 44.734, 44.366, 44.524, 44.508, 44.534, 44.718, 44.95, 44.606, 44.612, 44.342]
accuracy_test=[33.57, 34.88, 32.86, 31.64, 33.31, 30.68, 35.25, 34.92, 32.34, 32.4, 33.97, 33.56, 29.69, 32.56, 32.69, 30.99, 30.86, 34.51, 32.45, 31.86, 32.43, 32.8, 32.78, 35.46, 35.99, 31.26, 32.31, 30.69, 29.73, 32.85, 42.48, 42.68, 41.79, 42.46, 43.06, 44.13, 42.54, 44.1, 43.46, 43.76, 43.55, 42.95, 43.86, 43.16, 43.41, 42.66, 42.67, 42.93, 44.33, 43.14, 39.79, 44.46, 43.56, 42.03, 42.12, 43.27, 43.37, 43.05, 42.49, 41.7]
loss_test=[7.099485130310058, 5.628449530601501, 6.43655599117279, 6.67317991733551, 6.2114070749282835, 7.8445034599304195, 6.547454566955566, 6.0279730749130245, 5.988296146392822, 6.487298855781555, 6.075066537857055, 6.126173810958862, 7.226069669723511, 6.346400957107544, 5.8152416849136355, 7.004642038345337, 6.9219964456558225, 6.0406836318969725, 7.139968090057373, 6.618490591049194, 6.386580457687378, 6.761363158226013, 5.876645274162293, 5.610819549560547, 5.302061719894409, 6.450514636039734, 6.767355880737305, 7.649774074554443, 7.558789930343628, 6.287306823730469, 2.3279005908966064, 1.9799817705154419, 1.8656383538246155, 1.8027275025844574, 1.7301000297069549, 1.681739741563797, 1.7189399600028992, 1.6975924348831177, 1.6894110596179963, 1.6853671205043792, 1.6829921674728394, 1.7412980794906616, 1.6924275398254394, 1.716379166841507, 1.7082803547382355, 1.7319376540184022, 1.7392021894454956, 1.7193155932426452, 1.671981475353241, 1.721218786239624, 1.823811057806015, 1.6651534974575042, 1.6679883813858032, 1.7586032676696777, 1.714518278837204, 1.7275404381752013, 1.7008287000656128, 1.6789879930019378, 1.7684259390830994, 1.7823975825309752]


loss_train=loss_train+[1.4965438025686748, 1.4662693763328025, 1.4619478962915329, 1.4600807407018168, 1.457653630115187, 1.456841047157717, 1.4582738915977576, 1.456286903842331, 1.4563446706518188, 1.4563751315216884, 1.4566719721040458, 1.4565122886691861, 1.4567890118455034, 1.4574373875127729, 1.4560373015415944, 1.4555840522736845, 1.4549893177378819, 1.456840114520334, 1.4572046446373395, 1.4559400252369055, 1.4563996014387712, 1.4565926717065485, 1.4567627080566132, 1.4564456079926942, 1.4557898770207944, 1.4558877252861666, 1.4561849911804394, 1.4560500227886697, 1.4554649362783603, 1.456052756370486]
accuracy_train=accuracy_train+[48.754, 49.398, 49.58, 49.53, 49.698, 49.74, 49.57, 49.662, 49.58, 49.734, 49.618, 49.646, 49.66, 49.724, 49.736, 49.674, 49.726, 49.662, 49.624, 49.734, 49.654, 49.74, 49.73, 49.622, 49.626, 49.624, 49.69, 49.632, 49.606, 49.688]
accuracy_test=accuracy_test+[46.55, 46.55, 47.13, 46.94, 46.75, 46.45, 46.96, 46.9, 46.54, 46.91, 46.9, 46.26, 46.57, 46.43, 46.33, 46.78, 46.06, 46.85, 46.78, 46.75, 46.89, 46.91, 46.75, 46.41, 46.17, 46.55, 46.69, 46.85, 46.42, 46.66]
loss_test=loss_test+[1.578548104763031, 1.571926966905594, 1.561918261051178, 1.5603084194660186, 1.5613840067386626, 1.5636560881137849, 1.558588353395462, 1.5587475049495696, 1.5607672226428986, 1.5594867157936096, 1.5560881650447846, 1.5652550554275513, 1.5556936657428742, 1.5600204861164093, 1.5616307997703551, 1.5570446133613587, 1.564684317111969, 1.5573553621768952, 1.5543155014514922, 1.5602844369411468, 1.5541653990745545, 1.5528114593029023, 1.5563659358024597, 1.5561083447933197, 1.5607488393783568, 1.5592074728012084, 1.5565634310245513, 1.55528648853302, 1.5553760409355164, 1.5566393232345581]





plt.figure(1)
plt.plot(list_x,loss_train,'-g')
plt.title('Test 1')
plt.xlabel('epochs')
plt.ylabel('Train set loss')
plt.legend(loc='best')
plt.savefig('test_1_train_loss')

plt.figure(2)
plt.ylim(0,100)
plt.plot(list_x,accuracy_train,'-g')
plt.title('Test 1')
plt.xlabel('epochs')
plt.ylabel('Train set Accuracy')
plt.legend(loc='best')
plt.savefig('test_1_train_accuracy')


plt.figure(3)
plt.ylim(0,100)
plt.plot(list_x,accuracy_test,'-g')
plt.title('Test 1')
plt.xlabel('epochs')
plt.ylabel('Test set Accuracy')
plt.legend(loc='best')
plt.savefig('test_1_test_accuracy')




plt.figure(4)
plt.plot(list_x,loss_test,'-g')
plt.title('Test 1')
plt.xlabel('epochs')
plt.ylabel('Test set Loss')
plt.legend(loc='best')
plt.savefig('test_1_test_loss')