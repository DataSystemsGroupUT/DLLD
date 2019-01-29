from sklearn import svm
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load training Data
f=open('training_data.pickle','rb')
train=pickle.load(f)

#Load testing data
f=open('testing_data.pickle','rb')
test=pickle.load(f)
X_train=train['X']
print(len(X_train))
y_train=train['y']
print(len(y_train))

X_test=test['X']
y_test=test['y']

print(len(X_test))

# Traing the SVM
model=svm.SVC(gamma='scale',class_weight={0:1,1:5})
model.fit(X_train,y_train)
# Check the svm performance on test data
preds=model.predict(X_test)
count=0
t_count=0
for ind,i in enumerate(y_test):
	if(i==1):
		count+=1
		if(preds[ind]==1):
			t_count+=1


print('sensitivity',float(t_count)/float(count))

print(confusion_matrix(y_test,preds))
print(accuracy_score(y_test,preds))

# save the trained classifier
f=open('classifier.pickle','wb')
pickle.dump(model,f)


