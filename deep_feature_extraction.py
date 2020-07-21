# -*- coding: utf-8 -*-
"""deep_feature_extraction

zeshan khan

"""

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


def gather_paths_all(jpg_path,num_classes=16):
  if(num_classes==16):
    label_map=['retroflex-rectum', 'out-of-patient', 'ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'blurry-nothing', 'retroflex-stomach', 'instruments', 'dyed-resection-margins', 'stool-plenty', 'esophagitis', 'normal-pylorus', 'polyps', 'stool-inclusions', 'colon-clear']
  elif(num_classes==8):
    label_map=['ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-pylorus', 'polyps']
  count=sum([len(os.listdir(jpg_path+f)) for f in os.listdir(jpg_path)])
  counta=count
  folder=os.listdir(jpg_path)
  ima=['' for x in range(count)]
  labels=np.zeros((count,num_classes),dtype=float)
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
      #print(jpg_path[start+i])
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

def extract_class_prob(data_path=None,features_path=None,weight_path='imagenet',output_layer='avg_pool',base_model=None,step_size=500,num_classes=16):
  num_classes=len(os.listdir(data_path))
  count_train=np.sum([len(os.listdir(data_path+f)) for f in os.listdir(data_path)])
  image_paths,label,labels=gather_paths_all(data_path,num_classes=num_classes)
  X_train1 = image_paths[:]
  Y_train1=label[:]
  new_model=alter_last_layer(base_model=base_model,output_layer="avg_pool",num_classes=16)
  if(not weight_path=='imagenet'):
    new_model.load_weights(weight_path)
  return extract_features(model1=new_model,layer=None,X_train1=X_train1,Y_train1=Y_train1,features_path=features_path,step_size=step_size,count=count_train)

def extract_features(model1=None,layer=None,X_train1=None,Y_train1=None,features_path=None,step_size=500,count=5293):
  warnings.filterwarnings('ignore')
  steps=int(count/step_size)+1
  for i in range(steps):
    st=i*step_size
    end=(i+1)*step_size
    if(i==steps-1):
      end=count
    img_names=X_train1[st:end]
    print(st,end)
    Y=Y_train1[st:end]
    images=gather_images_from_paths(X_train1,st,end-st,img_rows=224,img_cols=224)
    output=model1.predict(images)
    ch='a'
    if(i==0):
      ch='w'
    with open(features_path, ch, newline='') as csvfile:
      spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
      for i in range(len(output)):
        spamwriter.writerow(np.concatenate((output[i],[Y[i]],[img_names[i].split("/")[-2]],[img_names[i].split("/")[-1].split(".")[0]])))
  return



data_path='path/to/data/'
features_path='/path/to/features.csv'
weight_path_resnet="resnet152_weights_tf_400.h5"
weight_path_densenet="densenet169_weights_tf_400.h5"




img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3

num_classes = 16

base_model=DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
extract_class_prob(data_path=data_path,features_path=features_path,weight_path=weight_path_densenet,output_layer="avg_pool",base_model=base_model,step_size=1000,num_classes=num_classes)

base_model=ResNet152(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
extract_class_prob(data_path=data_path,features_path=features_path,weight_path=weight_path_densenet,output_layer="avg_pool",base_model=base_model,step_size=1000,num_classes=num_classes)
