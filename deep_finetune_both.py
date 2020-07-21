import csv,warnings,os,cv2,numpy as np
warnings.filterwarnings('ignore')

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.utils import shuffle
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.densenet import DenseNet169
from tensorflow.python.keras.applications.resnet import ResNet152
from sklearn.model_selection import train_test_split

def gather_paths_all(jpg_path,num_classes=16):
  if(num_classes==16):
    label_map=['retroflex-rectum', 'out-of-patient', 'ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'blurry-nothing', 'retroflex-stomach', 'instruments', 'dyed-resection-margins', 'stool-plenty', 'esophagitis', 'normal-pylorus', 'polyps', 'stool-inclusions', 'colon-clear']
  elif(num_classes==8):
    label_map=['ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-pylorus', 'polyps']
  count=sum([len(os.listdir(jpg_path+f)) for f in os.listdir(jpg_path)])
  counta=count
  folder=os.listdir(jpg_path)
  ima=['' for x in range(count)]
  labels=np.zeros((count,len(folder)),dtype=float)
  label=[0 for x in range(count)]
  for fldr in folder:
      inner=1
      for f in os.listdir(jpg_path+fldr+"/"):
          im=jpg_path+fldr+"/"+f
          count-=1
          ima[count]=im
          label[count]=label_map.index(fldr)+1
          inner+=1
      if(count<=0):
          break
  for i in range(counta):
      labels[i][label[i]-1]=1
  return ima,label,labels

def gather_images_from_paths(jpg_path,start,count,img_rows=224,img_cols=224):
  print('Stats of Images Start:',start,' To:',(start+count),'All Images:',len(jpg_path))
  ima=np.zeros((count,img_rows,img_cols,3))
  for i in range(count):
      img=cv2.imread(jpg_path[start+i])
      im = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
      im[:,:,0] -= 103.939
      im[:,:,1] -= 116.779
      im[:,:,2] -= 123.68
      ima[i]=im
  return ima


def alter_last_layer(base_model=None,output_layer='avg_pool',num_classes=16):
  x = base_model.get_layer(output_layer).output
  x = Dense(num_classes, activation='softmax', name="output")(x)
  model = Model(base_model.input, outputs=x)
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='mean_squared_error', optimizer=sgd)
  return model
def finetune(base_model=None,output_layer='avg_pool',weights='imagenet',include_top=True,image_paths=None, labels=None,count_train=5293,weights_new=None,nb_epoch=1,i=0,num_classes=16):
  print(num_classes,"\t Classes")
  model=alter_last_layer(base_model,output_layer,num_classes)
  X_train1, X_test1, Y_train, Y_test = train_test_split(image_paths, labels, test_size=0.33, random_state=5)
  X_train=gather_images_from_paths(X_train1,start=0,count=len(X_train1))
  X_test=gather_images_from_paths(X_test1,start=0,count=len(X_test1))
  print(X_train.shape,X_test.shape,len(Y_train),len(Y_test))
  model.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(X_test, Y_test),)
  weights=weights_new+".h5"
  model.save(weights)

def re_finetune(base_model=None,output_layer='avg_pool',weights='imagenet',include_top=True,image_paths=None, labels=None,count_train=5293,weights_new=None,nb_epoch=1,i=0,num_classes=16):
  print(num_classes,"\t Classes")
  model=alter_last_layer(base_model,output_layer,num_classes)
  model.load_weights(weights_new)
  X_train1, X_test1, Y_train, Y_test = train_test_split(image_paths, labels, test_size=0.33, random_state=5)
  X_train=gather_images_from_paths(X_train1,start=0,count=len(X_train1))
  X_test=gather_images_from_paths(X_test1,start=0,count=len(X_test1))
  print(X_train.shape,X_test.shape,len(Y_train),len(Y_test))
  #print(model.summary())
  # Start Fine-tuning
  model.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(X_test, Y_test),)
  model.save(weights_new)



data_path='path/to/data/'
features_path='/path/to/features.csv'
weight_path_resnet="resnet152_weights_tf_400.h5"
weight_path_densenet="densenet169_weights_tf_400.h5"


img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3

num_classes = 16
image_paths,label,labels=gather_paths_all(data_path,num_classes=num_classes)

#base_model=ResNet152(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
base_model=DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

count_train=len(label)
batch_size = 25

nb_epoch = 50

step_size=4000
finetune(base_model=base_model,output_layer='avg_pool',weights='imagenet',include_top=True,image_paths=image_paths, labels=labels,count_train=count_train,weights_new=weight_path_resnetfine,nb_epoch=nb_epoch)
#re_finetune(base_model=base_model,output_layer='avg_pool',weights='imagenet',include_top=True,image_paths=image_paths, labels=labels,count_train=count_train,weights_new=weight_path_densenet,nb_epoch=nb_epoch)