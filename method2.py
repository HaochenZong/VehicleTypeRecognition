## Competition step2(training networks)

#%% import libraries 
import numpy as np
import matplotlib.pyplot as plt
import os 
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten,Dropout
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, applications
import efficientnet.keras as efn
#%%  Create an index of class names
# you can change the route 
class_names = sorted(os.listdir(r"C:/Users/95189/Downloads/ML/project/vehicle/train/train/"))
num_classes = len(class_names)
#%%  Load all images 
# Find all image files in the data directory.
data_image = []  
classes = [] 
for root, dirs, files in os.walk(r"C:/Users/95189/Downloads/ML/project/vehicle/train/train/"): 
    for name in files: 
        if name.endswith(".jpg"):
            img = plt.imread(root + os.sep + name)
            img = cv2.resize(img, (224,224))
            img = img.astype(np.float32) 
            img -= 128
            data_image.append(img)
            label = root.split("/")[-1] 
            classes.append(class_names.index(label))
    print("finished class",root.split("/")[-1])

# cast lists to array
data_image = np.array(data_image) 
classes = np.array(classes)
#%% split training data and testing data
from sklearn.model_selection import train_test_split
(image_train, image_val, class_train, class_val) = train_test_split(data_image, classes, test_size=0.2)

# convert class vectors to binary class matrices
class_train = tf.keras.utils.to_categorical(class_train, num_classes)
class_val = tf.keras.utils.to_categorical(class_val, num_classes)

#%% Mobilenet
mobile = MobileNet(include_top = False, alpha = 0.25,
                   input_shape = (224,224,3))
out = mobile.output
out = Flatten()(out)
out = Dense(100,activation='relu')(out) 
out = Dense(17,activation='softmax')(out) 
model = Model(inputs = mobile.input,outputs = out)

model.compile(loss = tf.keras.losses.categorical_crossentropy,
              optimizer = tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()
#%% training Mobilenet
model.fit(image_train, class_train,
          batch_size= 20,
          epochs= 12,
          verbose=1,
          validation_data=(image_val, class_val))
score = model.evaluate(image_val, class_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%% MobilenetV2
from tensorflow.keras.applications import MobileNetV2
mobilenetV2 = MobileNetV2(include_top=False,input_shape=(224,224,3))
model_V2 = Sequential()
model_V2.add(mobilenetV2)
model_V2.add(layers.GlobalAveragePooling2D())
model_V2.add(layers.Dropout(0.5))
model_V2.add(layers.Dense(100,activation='relu'))
model_V2.add(layers.Dense(17, activation='softmax'))
model_V2.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model_V2.summary()
#%% training MobilenetV2
model_V2.fit(image_train, class_train,
          batch_size= 20,
          epochs= 12,
          verbose=1,
          validation_data=(image_val, class_val))
score_V2 = model_V2.evaluate(image_val, class_val, verbose=0)
print('Test loss:', score_V2[0])
print('Test accuracy:', score_V2[1])

#%% InceptionV3
from tensorflow.keras.applications.inception_v3 import InceptionV3
inceptionV3 = InceptionV3(include_top=False,input_shape=(224,224,3))
model_in3 = Sequential()
model_in3.add(inceptionV3)
 # add a global spatial average pooling layer
model_in3.add(layers.GlobalAveragePooling2D())
model_in3.add(layers.Dropout(0.5))
model_in3.add(layers.Dense(100,activation='relu'))
model_in3.add(layers.Dense(17, activation='softmax'))
model_in3.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model_in3.summary()

#%% training InceptionV3
model_in3.fit(image_train, class_train,
          batch_size= 20,
          epochs= 12,
          verbose=1,
          validation_data=(image_val, class_val))
score_in3 = model_in3.evaluate(image_val, class_val, verbose=0)
print('Test loss:', score_in3[0])
print('Test accuracy:', score_in3[1])

#%% add processing of data augmentation
import imgaug.augmenters as iaa

# OneOf: choose only one processing to generate new image
seq = iaa.OneOf([
    iaa.Flipud(), # vertical flips
    iaa.Flip(), # horizontal fiips
    iaa.Affine(rotate=90), # roatation
    iaa.Multiply((1.2, 1.5)), #random brightness
    iaa.GaussianBlur(sigma = [1,4])]) # random blurring image
    

# data augmentation in image_train
# Get a numpy array of all the indices of the input data

indices = np.arange(len(image_train))
np.random.shuffle(indices)

img_aug = []
img_aug_class = []
# in order to speed up, I only increase len(image_train)/4 images
for i in indices:
    image = image_train[i]
    new_image = seq(images = image)
    img_aug.append(new_image)
    img_aug_class.append(class_train[i])
#%% convert list to array
img_aug = np.array(img_aug)
img_aug_class = np.array(img_aug_class)

#%% combine img_agu and image_train
aug_train_img = np.vstack((image_train,img_aug))
aug_train_class = np.vstack((class_train,img_aug_class))
#%% training InceptionV3 model again
# train network
model_in3.fit(aug_train_img, aug_train_class,
          batch_size= 20,
          epochs= 12,
          verbose=1,
          validation_data=(image_val, class_val))
score_in3 = model_in3.evaluate(image_val, class_val, verbose=0)
print('Test loss:', score_in3[0])
print('Test accuracy:', score_in3[1])

#%%  EfficientNetB7 model
model_eff = efn.EfficientNetB7((224,224,3),classes = 17, weights='imagenet')
model_eff.layers.append(layers.Dropout(0.5))
model_eff.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6),loss='categorical_crossentropy',metrics=['accuracy'])
model_eff.summary()
#%% 
model_eff.fit(image_train, class_train,
          batch_size= 30,
          epochs= 12,
          verbose=1,
          validation_data=(image_val, class_val))
score_eff = model_eff.evaluate(image_val, class_val, verbose=0)
print('Test loss:', score_eff[0])
print('Test accuracy:', score_eff[1])


#%% Write in test file
from tensorflow.keras.preprocessing import image
with open("C:/Users/95189/Downloads/ML/project/vehicle/submission.csv", "w") as fp:
    fp.write("Id,Category\n")
    for root, dirs, images in os.walk(r"C:/Users/95189/Downloads/ML/project/vehicle/test/testset/"):
        for i in range(len(images)):
            img = plt.imread(root + os.sep + images[i])
            img = cv2.resize(img, (224,224))
            img = img.astype(np.float32) 
            img -= 128
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            class_index = model_in3.predict_classes(x)[0] 
            label = class_names[class_index]
            fp.write("%d,%s\n" % (i, label))
            
        

        


            


 


