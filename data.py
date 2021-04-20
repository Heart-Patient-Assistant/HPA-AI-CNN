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


#importing test and train data
train = pd.read_csv("C:/Users/manal/Desktop/Data/train.csv", header = None)
test = pd.read_csv("C:/Users/manal/Desktop/Data/test.csv", header = None)

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

#testing the resampling by plotting the data
# per_case = train[187].value_counts()
# plt.figure(figsize=(20,10))
# circle=plt.Circle( (0,0), 0.8, color='white')
# cmap = plt.get_cmap("tab20c")
# plt.pie(per_case, labels=['normal beat','unknown Beats','Ventricular ectopic beats','Supraventricular ectopic beats','Fusion Beats'], colors=cmap(np.arange(5)*4),autopct='%1.1f%%')
# p=plt.gcf()
# p.gca().add_artist(circle)
# plt.show()

classes=train.groupby(187,group_keys=False).apply(lambda train : train.sample(1))

#adding gaussian  noise to the signals to make it more realistic
def add_gaussian_noise(signal):
    noise=np.random.normal(0,0.05,186)
    return (signal+noise)

    tempo=classes.iloc[0,:186]
bruiter=add_gaussian_noise(tempo)
#testing if the noise is applied
# plt.subplot(2,1,1)
# plt.plot(classes.iloc[0,:186])

# plt.subplot(2,1,2)
# plt.plot(bruiter)

# plt.show()

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