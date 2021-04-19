#!/usr/bin/env python
# coding: utf-8

# In[19]:


#importing important libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight

#ignoring the warnings
import warnings
warnings.filterwarnings('ignore')


# In[20]:


#importing test and train data
train = pd.read_csv("C:/Users/manal/Desktop/Data/train.csv", header = None)
test = pd.read_csv("C:/Users/manal/Desktop/Data/test.csv", header = None)


# In[21]:


#Resampling 20000 record from normal beat(as its the biggest percent) set in order to balance the percentage of other cases
from sklearn.utils import resample

#resampled_arrayssequence of array-like of shape (n_samples,) or (n_samples, n_outputs)
#Sequence of resampled copies of the collections. The original arrays are not impacted.

#collecting classes from last coloumn and putting it in variables
d1=train[train[187]==1]
d2=train[train[187]==2]
d3=train[train[187]==3]
d4=train[train[187]==4]

#specifying 20000 recordfrom normal beat cases
d0=(train[train[187]==0]).sample(n=20000,random_state=42)

#Random_case Determines random number generation for shuffling the data.
#Pass an int for reproducible results across multiple function calls.

d1_upsampled=resample(d1,replace=True,n_samples=20000,random_state=123)
d2_upsampled=resample(d2,replace=True,n_samples=20000,random_state=124)
d3_upsampled=resample(d3,replace=True,n_samples=20000,random_state=125)
d4_upsampled=resample(d4,replace=True,n_samples=20000,random_state=126)

train=pd.concat([d0,d1_upsampled,d2_upsampled,d3_upsampled,d4_upsampled])


# In[22]:


#testing the resampling by plotting the data
# per_case = train[187].value_counts()
# plt.figure(figsize=(20,10))
# circle=plt.Circle( (0,0), 0.8, color='white')
# cmap = plt.get_cmap("tab20c")
# plt.pie(per_case, labels=['normal beat','unknown Beats','Ventricular ectopic beats','Supraventricular ectopic beats','Fusion Beats'], colors=cmap(np.arange(5)*4),autopct='%1.1f%%')
# p=plt.gcf()
# p.gca().add_artist(circle)
# plt.show()


# In[24]:


classes=train.groupby(187,group_keys=False).apply(lambda train : train.sample(1))


# In[25]:


#adding gaussian  noise to the signals to make it more realistic
def add_gaussian_noise(signal):
    noise=np.random.normal(0,0.05,186)
    return (signal+noise)


# In[27]:


tempo=classes.iloc[0,:186]
bruiter=add_gaussian_noise(tempo)
#testing if the noise is applied
# plt.subplot(2,1,1)
# plt.plot(classes.iloc[0,:186])

# plt.subplot(2,1,2)
# plt.plot(bruiter)

# plt.show()


# In[52]:


target_train=train[187]
target_test=test[187]
y_train=to_categorical(target_train)
y_test=to_categorical(target_test)
x_train=train.iloc[:,:186].values
x_test=test.iloc[:,:186].values
for i in range(len(X_train)):
    x_train[i,:186]= add_gaussian_noise(X_train[i,:186])
x_train = x_train.reshape(len(x_train), x_train.shape[1],1)
x_test = x_test.reshape(len(x_test), x_test.shape[1],1)


# In[53]:


#building the network
def network(x_train,y_train,x_test,y_test):
    im_shape=(x_train.shape[1],1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
    conv1_1=Convolution1D(64, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
    conv1_1=BatchNormalization()(conv1_1)
    pool1=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
    conv2_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool1)
    conv2_1=BatchNormalization()(conv2_1)
    pool2=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv2_1)
    conv3_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    pool3=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv3_1)
    flatten=Flatten()(pool3)
    dense_end1 = Dense(64, activation='relu')(flatten)
    dense_end2 = Dense(32, activation='relu')(dense_end1)
    main_output = Dense(5, activation='softmax', name='main_output')(dense_end2)
    
    
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    history=model.fit(x_train, y_train,epochs=5,callbacks=callbacks, batch_size=32,validation_data=(x_test,y_test))
    model.load_weights('best_model.h5')
    return(model,history)
     


# In[54]:


def evaluate_model(history,x_test,y_test,model):
    scores = model.evaluate((x_test),y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    target_names=['0','1','2','3','4']
    
    y_true=[]
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba=model.predict(X_test)
    prediction=np.argmax(prediction_proba,axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)


# In[55]:


from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

model,history=network(x_train,y_train,x_test,y_test)


# In[49]:


#comparing model loss and accuracy
# evaluate_model(history,x_test,y_test,model)
# y_pred=model.predict(x_test)


# In[ ]:




