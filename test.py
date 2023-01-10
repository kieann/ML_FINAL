import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

KFOLD = 5

# reading data
train_df = pd.read_csv('./kaggle/train.csv')
test_df = pd.read_csv('./kaggle/test.csv')
feature_names = train_df.columns.values

x_data = train_df.drop(labels=["failure"], axis="columns")
x_data = x_data.values
y_data = train_df['failure'].values

test_data = test_df.values



# for finding the attribute feature of test and train
'''
attribute_list = np.arange(1, 5)
print(f"The different attribute values of train data")
for i in attribute_list:
    s = set(x_data[:, i])
    print(feature_names[i+2], s)

print("")
print(f"The different attribute values of test data")
for i in attribute_list:
    s = set(test_data[:, i])
    print(feature_names[i+2], s)
'''

# for finding the distribution of the feature of the data
for i in range(len(x_data)):
    plt.plot(x_data[:,i], x_data[:,0],'.',color = 'blue')
    plt.plot(test_data[:,i], test_data[:,0],'.',color = 'red')
    plt.show()



# Getting data that does not include unwanted features
x_data = train_df.drop(labels=["failure", "product_code", "id", "attribute_1", "attribute_2", "attribute_3",], axis="columns")
x_data = x_data.values
y_data = train_df['failure'].values

# kfold data
X = np.arange(len(x_data))
kf = KFold(shuffle=True)
kfold_data = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    kfold_data.append([train_index, test_index])


# Data training and testing 

# ================= Decision tree ====================
best_acc = 0
for k in range(KFOLD):
    x_train = x_data[kfold_data[k][0]]
    y_train = y_data[kfold_data[k][0]]

    x_test = x_data[kfold_data[k][1]]
    y_test = y_data[kfold_data[k][1]]

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_pred, y_test)
    if best_acc < acc:
        best_acc = acc
print("best accuracy using DecisionTree : {0:.2f}".format(best_acc*100))

# ================== SVD ===================
best_acc = 0
clf = SVC(C=1, kernel='rbf', gamma=7.5e-05)
for k in range(KFOLD):

    x_train = x_data[kfold_data[k][0]]
    y_train = y_data[kfold_data[k][0]]

    x_test = x_data[kfold_data[k][1]]
    y_test = y_data[kfold_data[k][1]]

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_pred, y_test)
    if best_acc < acc:
        best_acc = acc

print("best accuracy using SVM : {0:.2f}".format(best_acc*100))


# ================ XGB ======================

clf = XGBClassifier(n_estimators=20, learning_rate=0.1)

for k in range(KFOLD):
    x_train = x_data[kfold_data[k][0]]
    y_train = y_data[kfold_data[k][0]]

    x_test = x_data[kfold_data[k][1]]
    y_test = y_data[kfold_data[k][1]]

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    if best_acc < acc:
        best_acc = acc

print("best accuracy using XGB : {0:.2f}".format(best_acc*100))
