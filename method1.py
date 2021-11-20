## Competition step2(training networks)

#%% import libraries 
import numpy as np
import matplotlib.pyplot as plt
import os 
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.mobilenet import MobileNet
#%%  Create an index of class names
# you can change the route 
class_names = sorted(os.listdir(r"C:/Users/95189/Downloads/ML/project/vehicle/train/train/"))

#%% Prepare a pretrained CNN for feature extraction
base_model = tf.keras.applications.mobilenet.MobileNet( 
        input_shape = (224,224,3), include_top = False)

in_tensor = base_model.inputs[0] # Grab the input of base model 
out_tensor = base_model.outputs[0] # Grab the output of base model
# Add an average pooling layer (averaging each of the 1024 channels): 
out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)
# Define the full model by the endpoints. 
model = tf.keras.models.Model(inputs = [in_tensor], outputs = [out_tensor])
# Compile the model for execution. Losses and optimizers # can be anything here, since we don’t train the model. 
model.compile(loss = "categorical_crossentropy", optimizer = 'sgd')

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
#%%  Load all images and apply the network to each
# Find all image files in the data directory.
features = [] # Feature vectors will go here. 
classes = [] # Class ids will go here.
for root, dirs, files in os.walk(r"C:/Users/95189/Downloads/ML/project/vehicle/train/train/"): 
    for name in files: 
        if name.endswith(".jpg"):
            img = plt.imread(root + os.sep + name)
            img = cv2.resize(img, (224,224))
            img = img.astype(np.float32) 
            img -= 128
            x = model.predict(img[np.newaxis, ...])[0]
            features.append(x)
            label = root.split("/")[-1] 
            classes.append(class_names.index(label))
    print("finished class",root.split("/")[-1])

# cast lists to array
features = np.array(features) 
classes = np.array(classes)

#%% Try different models
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


x_train,x_test, y_train,  y_test = model_selection.train_test_split(
                                   features, classes, test_size = 0.2)



models = []
models.append(('RandomForest', RandomForestClassifier(n_estimators=100)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('SVM_linear', svm.SVC(kernel='linear')))
models.append(('SVM_rbf', svm.SVC(kernel='rbf')))
models.append(('LR', LogisticRegression()))


for name, model in models:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    msg = "Accuracy of model %s: %f" % (name, score)
    print(msg)
#%% Select the best model
# according to above result, the best model is SVM with rbf kernel

name, select_model = models[3]
#%% Write in test file

with open("C:/Users/95189/Downloads/ML/project/vehicle/submission.csv", "w") as fp:
    fp.write("Id,Category\n")
    for root, dirs, images in os.walk(r"C:/Users/95189/Downloads/ML/project/vehicle/test/testset/"):
        for i in range(len(images)):
            img = plt.imread(root + os.sep + images[i])
            img = cv2.resize(img, (224,224))
            img = img.astype(np.float32) 
            img -= 128
            x = model.predict(img[np.newaxis, ...])[0]
            x = np.array(x) 
            class_index = select_model.predict([x])[0]
            label = class_names[class_index]
            fp.write("%d,%s\n" % (i, label))
            
        

        


            


 


