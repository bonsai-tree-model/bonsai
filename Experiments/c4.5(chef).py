import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from chefboost import Chefboost as chef
from sklearn.metrics import accuracy_score, f1_score



# # Zoo
# df_pq = pd.read_csv('dataset/zoo/zoo_train.csv')
# df_pq_test = pd.read_csv('dataset/zoo/zoo_test.csv')
# categorical_cols =[]
# target_col = 'class'

# # Wholesale
# df_pq = pd.read_csv('dataset/Wholesale/Wholesale_train.csv')
# df_pq_test = pd.read_csv('dataset/Wholesale/Wholesale_test.csv')
# categorical_cols = ['Channel']
# target_col = 'class'

# # wine
# df_pq = pd.read_csv('dataset/wine/wine_train.csv')
# df_pq_test = pd.read_csv('dataset/wine/wine_test.csv')
# categorical_cols = []
# target_col = 'class'

# # heart
# df_pq = pd.read_csv('dataset/heart/heart_train.csv')
# df_pq_test = pd.read_csv('dataset/heart/heart_test.csv')
# categorical_cols=['anaemia','diabetes','high_blood_pressure','sex','smoking']
# target_col = 'DEATH_EVENT'

# # # contraceptive
# df_pq = pd.read_csv('dataset/contraceptive/contraceptive_train.csv')
# df_pq_test = pd.read_csv('dataset/contraceptive/contraceptive_test.csv')
# categorical_cols = ['wife_edu','husband_edu','wife_religion','wife_working','husband_occupation','standard_of_living_index','media_exposure']
# target_col = 'class'

# # credit_approval
# df_pq = pd.read_csv('dataset/credit_approval/credit_approval_train.csv')
# df_pq = df_pq.dropna().reset_index(drop=True)
# df_pq_test = pd.read_csv('dataset/credit_approval/credit_approval_test.csv')
# df_pq_test = df_pq_test.dropna().reset_index(drop=True)
# categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
# target_col = 'class'

# # hepatitis
# df_pq = pd.read_csv('dataset/hepatitis/hepatitis_train.csv')
# df_pq = df_pq.dropna().reset_index(drop=True)
# df_pq_test = pd.read_csv('dataset/hepatitis/hepatitis_test.csv')
# df_pq_test = df_pq_test.dropna().reset_index(drop=True)
# categorical_cols = ['Sex','Steroid','Antivirals','Fatigue','Malaise','Anorexia','Liver_Big','Liver_Firm','Spleen_Palpable','Spiders','Ascites','Varices','Histology']
# target_col = 'class'

# # ILPD
# df_pq = pd.read_csv('dataset/ILPD/ILPD_train.csv')
# df_pq_test = pd.read_csv('dataset/ILPD/ILPD_test.csv')
# categorical_cols =['Gender']
# target_col = 'class'

# # # blood_transfusion
# df_pq = pd.read_csv('dataset/blood_transfusion/blood_transfusion_train.csv')
# df_pq_test = pd.read_csv('dataset/blood_transfusion/blood_transfusion_test.csv')
# categorical_cols =[]
# target_col = 'class'

# # penguins
# df_pq = pd.read_csv('dataset/penguin/penguin_train.csv')
# df_pq_test = pd.read_csv('dataset/penguin/penguin_test.csv')
# categorical_cols =['island','sex']
# target_col = 'class'

# iris
df_pq = pd.read_csv('dataset/iris/iris_train.csv')
df_pq_test = pd.read_csv('dataset/iris/iris_test.csv')
categorical_cols=[]
target_col = 'class'

# # rice
# df_pq = pd.read_csv('dataset/Rice/Rice_train.csv')
# df_pq_test = pd.read_csv('dataset/Rice/Rice_test.csv')
# categorical_cols = []
# target_col = 'class'

# # TODO diabetes
# df_pq = pd.read_csv('dataset/diabetes/diabetes_train.csv')
# df_pq_test = pd.read_csv('dataset/diabetes/diabetes_test.csv')
# categorical_cols = []
# target_col = 'class'
#
# # TODO breast_cancer_wisconsin_prognostic
# df_pq = pd.read_csv('dataset/breast_cancer_wisconsin_prognostic/breast_cancer_wisconsin_prognostic_train.csv')
# df_pq_test = pd.read_csv('dataset/breast_cancer_wisconsin_prognostic/breast_cancer_wisconsin_prognostic_test.csv')
# categorical_cols = []
# target_col = 'class'

# # TODO parkinsons
# df_pq = pd.read_csv('dataset/parkinsons/parkinsons_train.csv')
# df_pq_test = pd.read_csv('dataset/parkinsons/parkinsons_test.csv')
# categorical_cols = []
# target_col = 'class'

# # TODO car_evaluation
# df_pq = pd.read_csv('dataset/car_evaluation/car_evaluation_train.csv')
# df_pq_test = pd.read_csv('dataset/car_evaluation/car_evaluation_test.csv')
# categorical_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
# target_col = 'class'

# # TODO tic_tac_toe
# df_pq = pd.read_csv('dataset/tic_tac_toe/tic_tac_toe_train.csv')
# df_pq_test = pd.read_csv('dataset/tic_tac_toe/tic_tac_toe_test.csv')
# categorical_cols = [
#     'top-left-square', 'top-middle-square', 'top-right-square',
#     'middle-left-square', 'middle-middle-square', 'middle-right-square',
#     'bottom-left-square', 'bottom-middle-square', 'bottom-right-square'
# ]
# target_col = 'class'

# # banknote_authentication
# df_pq = pd.read_csv('dataset/banknote_authentication/banknote_authentication_train.csv')
# df_pq_test = pd.read_csv('dataset/banknote_authentication/banknote_authentication_test.csv')
# categorical_cols = []
# target_col = 'class'


# # sensor_readings_24
# df_pq = pd.read_csv('dataset/sensor_readings_24/sensor_readings_24_train.csv')
# df_pq_test = pd.read_csv('dataset/sensor_readings_24/sensor_readings_24_test.csv')
# categorical_cols = []
# target_col = 'class'

# # image_segmentation
# df_pq = pd.read_csv('dataset/image_segmentation/image_segmentation_train.csv')
# df_pq_test = pd.read_csv('dataset/image_segmentation/image_segmentation_test.csv')
# categorical_cols = []
# target_col = 'class'

# # seismic_bumps
# df_pq = pd.read_csv('dataset/seismic_bumps/seismic_bumps_train.csv')
# df_pq_test = pd.read_csv('dataset/seismic_bumps/seismic_bumps_test.csv')
# categorical_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard']
# target_col = 'class'
######################################### data ##############################################

# if len(categorical_cols) != 0:
#     encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#     X_train_cat = encoder.fit_transform(df_pq[categorical_cols])
#     X_test_cat = encoder.transform(df_pq_test[categorical_cols])
#     # 변환된 데이터프레임으로 변환
#     X_train_cat_df = pd.DataFrame(X_train_cat, columns=encoder.get_feature_names_out(categorical_cols))
#     X_test_cat_df = pd.DataFrame(X_test_cat, columns=encoder.get_feature_names_out(categorical_cols))
#     # 원본 데이터에서 숫자형 데이터만 선택
#     X_train_num = df_pq.drop(columns=categorical_cols)
#     X_test_num = df_pq_test.drop(columns=categorical_cols)
#     # 숫자형 데이터와 원핫 인코딩 데이터를 합치기
#     df_pq = pd.concat([X_train_num.reset_index(drop=True), X_train_cat_df], axis=1)
#     df_pq_test = pd.concat([X_test_num.reset_index(drop=True), X_test_cat_df], axis=1)
#
df_pq = df_pq.rename(columns={target_col: 'target'})
df_pq_test = df_pq_test.rename(columns={target_col: 'target'})

# Chefboost는 class가 문자열이어야 잘 작동함
df_pq['target'] = df_pq['target'].astype(str)
df_pq_test['target'] = df_pq_test['target'].astype(str)

train_df = df_pq
test_df = df_pq_test

config = {
    "algorithm": "C4.5",
    "enableParallelism": False,
    "enableTuning": False,
    "maxDepth": 3
}
model = chef.fit(train_df, config=config, target_label='target')

# 공통 feature 추출
feature_cols = df_pq.columns.drop('target').tolist()

# ===== Train 평가 =====
X_train = train_df[feature_cols].values.tolist()
y_train_true = train_df['target'].values
y_train_pred = [chef.predict(model, row) for row in X_train]
train_acc = accuracy_score(y_train_true, y_train_pred)
train_f1 = f1_score(y_train_true, y_train_pred, average='macro')

# ===== Test 평가 =====
X_test = test_df[feature_cols].values.tolist()
y_test_true = test_df['target'].values
y_test_pred = [chef.predict(model, row) for row in X_test]
test_acc = accuracy_score(y_test_true, y_test_pred)
test_f1 = f1_score(y_test_true, y_test_pred, average='macro')

print(f"Train Accuracy: {train_acc:.4f} | Train F1: {train_f1:.4f}")
print(f"Test  Accuracy: {test_acc:.4f} | Test  F1: {test_f1:.4f}")

