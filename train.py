import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


train_df = pd.read_csv('./kaggle/train.csv')
numerics_col_list = list(train_df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns)
cat_col_list = list(set(train_df.columns)-set(numerics_col_list))
train_df = train_df.dropna()

num_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
cat_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                            (('odi', (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan))))])
transformer = ColumnTransformer(transformers=[("num", num_trans, numerics_col_list),
                                              ("cat", cat_trans, cat_col_list)])
train_df = pd.DataFrame(transformer.fit_transform(train_df), columns=(numerics_col_list+cat_col_list))
train_df = train_df.drop(labels = ['id','product_code', 'attribute_1', 'attribute_2', 'attribute_3'], axis=1)

x_data = train_df.drop(['failure'], axis=1).values
y_data = train_df['failure'].values


# === Load train data & Data pre-processing ===
'''
train_df = pd.read_csv('./kaggle/train.csv')
train_df = train_df.dropna()
x_data = train_df.drop(labels=["failure", "product_code", "id", "attribute_1", "attribute_2", "attribute_3",], axis="columns")
x_data = x_data.values
y_data = train_df['failure'].values

for i in range(len(x_data)):
    if (x_data[i][1] == "material_5"):
        x_data[i][1] = 0
    else:
        x_data[i][1] = 1

null = []
for col in range(len(x_data[0])):
    for i in range(len(x_data)):
        if np.isnan(x_data[i][col]):
            null.append(i)
            x_data[i][col] = 0
    avg = np.mean(x_data[:, col])
    x_data[null, col] = avg
'''

# ============== kfold data =================
KFOLD = 5
clf = XGBClassifier(n_estimators=20, learning_rate=0.1)
X = np.arange(len(x_data))
kf = KFold(shuffle=True)
kfold_data = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    kfold_data.append([train_index, test_index])


# === Finding best hyperparameter based on validation set ===
estimator = [20, 40, 60, 80, 100]
learning_rate = [0.05, 0.1, 0.5]

acc = 0
best_acc = 0
best_parameters = []

for n in estimator:
    for lr in learning_rate:
        clf = XGBClassifier(n_estimators=n, learning_rate=lr)
        k_acc = 0
        for k in range(KFOLD):
            x_train = x_data[kfold_data[k][0]]
            y_train = y_data[kfold_data[k][0]]

            x_val = x_data[kfold_data[k][1]]
            y_val = y_data[kfold_data[k][1]]

            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_val)
            k_acc = k_acc + accuracy_score(y_pred, y_val)
        acc = k_acc / KFOLD

        print(
            "acc using parameters n_estimaters={0:}, lr={1:} : {2:.2f}".format(n, lr, acc))

        if best_acc < acc:
            best_acc = acc
            best_parameters = [n, lr]

print("best accuracy : {0:} using parameters n_estimaters={1:}, lr={2}".format(
    best_acc, best_parameters[0], best_parameters[1]))

# === Train the entire model base on the entire training data ===
clf = XGBClassifier(
    n_estimators=best_parameters[0], learning_rate=best_parameters[1])

clf.fit(x_data, y_data)

# ===== Output the pre-trained model ========
clf.save_model("model.bin")
