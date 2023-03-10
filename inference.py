import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

# ==== Load and pre-process data =====

test_df = pd.read_csv('./kaggle/test.csv')
test_idx = test_df['id'].values
numerics_col_list = list(test_df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns)
cat_col_list = list(set(test_df.columns)-set(numerics_col_list))


num_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
cat_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                            (('odi', (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan))))])
transformer = ColumnTransformer(transformers=[("num", num_trans, numerics_col_list),
                                              ("cat", cat_trans, cat_col_list)])
test_df = pd.DataFrame(transformer.fit_transform(test_df), columns=(numerics_col_list+cat_col_list))


test_data = test_df.drop(labels = ['product_code','id', 'attribute_1', 'attribute_2', 'attribute_3'], axis=1).values



# ==== Load pre-trained model ==== 
clf = XGBClassifier()
clf.load_model("model.bin")
pred = clf.predict(test_data)


# ==== Write prediction to submission file ====
df = pd.DataFrame({'id':test_idx, 'failure': pred})
df.to_csv("submission.csv", index=False)