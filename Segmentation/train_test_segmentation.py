
import tensorflow as tf
import numpy as np
from tfrbm import GBRBM  # Specifc implementation with NRELU provided in 
from matplotlib import pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Loading Training and Testing pataches, extracted from images and flattened using extract_patches.py
f=open('data_all_normalised.pickle','rb')   
data=pickle.load(f)
train_array=np.array(data['train_x'])
test_array=np.array(data['test_x'])
y_train=np.array(data['train_y'])
y_test=np.array(data['test_y'])

# Training Restricted BoltzMann Machine
rbm = GBRBM(11*11, 60, learning_rate=0.001, momentum=0, err_function='mse', use_tqdm=True, sample_visible=False, sigma=1)
errs = rbm.fit(train_array,n_epoches=100, batch_size=32, shuffle=True, verbose=True)
#Extracting features using trained RBM
features_train=rbm.transform(train_array)
features_test=rbm.transform(test_array)

#Training and generating the results from Random Forest Classifier.
clf = RandomForestClassifier(n_estimators=200, criterion='entropy')
clf.fit(features_train,y_train)
y_pred_test=clf.predict(features_test)
y_pred_train=clf.predict(features_train)
accuracy_test=accuracy_score(y_test,y_pred_test)
accuracy_train=accuracy_score(y_train,y_pred_train)

print('Training Accuracy',accuracy_train)
print('Test Accuracy',accuracy_test)

# Save the trained model to use for segmentation of other imagess.
res={}
res['RF']=clf
res['rbm']=rbm.get_weights()
f=open('model.pickle','wb')
pickle.dump(res,f)
