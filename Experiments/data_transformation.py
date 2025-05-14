import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np

file_name = 'Zoo'
df_pq = pd.read_csv('dataset/zoo/zoo_train.csv')
df_pq_test = pd.read_csv('dataset/zoo/zoo_test.csv')
# categorical_cols =['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic',
#  'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins',
#  'legs', 'tail', 'domestic', 'catsize']
categorical_cols =[]
target_col = 'class'

# file_name = 'Wholesale'
# df_pq = pd.read_csv('dataset/Wholesale/Wholesale_train.csv')
# df_pq_test = pd.read_csv('dataset/Wholesale/Wholesale_test.csv')
# categorical_cols = ['Channel']
# target_col = 'class'

# file_name = 'wine'
# df_pq = pd.read_csv('dataset/wine/wine_train.csv')
# df_pq_test = pd.read_csv('dataset/wine/wine_test.csv')
# categorical_cols = []
# target_col = 'class'

# file_name = 'heart'
# df_pq = pd.read_csv('dataset/heart/heart_train.csv')
# df_pq_test = pd.read_csv('dataset/heart/heart_test.csv')
# categorical_cols=['anaemia','diabetes','high_blood_pressure','sex','smoking']
# target_col = 'class'

# file_name = 'contraceptive'
# df_pq = pd.read_csv('dataset/contraceptive/contraceptive_train.csv')
# df_pq_test = pd.read_csv('dataset/contraceptive/contraceptive_test.csv')
# categorical_cols = ['wife_edu','husband_edu','wife_religion','wife_working','husband_occupation','standard_of_living_index','media_exposure']
# target_col = 'class'

# file_name = 'credit_approval'
# df_pq = pd.read_csv('dataset/credit_approval/credit_approval_train.csv')
# df_pq = df_pq.dropna().reset_index(drop=True)
# df_pq_test = pd.read_csv('dataset/credit_approval/credit_approval_test.csv')
# df_pq_test = df_pq_test.dropna().reset_index(drop=True)
# categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
# target_col = 'class'

# file_name = 'hepatitis'
# df_pq = pd.read_csv('dataset/hepatitis/hepatitis_train.csv')
# df_pq = df_pq.dropna().reset_index(drop=True)
# df_pq_test = pd.read_csv('dataset/hepatitis/hepatitis_test.csv')
# df_pq_test = df_pq_test.dropna().reset_index(drop=True)
# categorical_cols = ['Sex','Steroid','Antivirals','Fatigue','Malaise','Anorexia','Liver_Big','Liver_Firm','Spleen_Palpable','Spiders','Ascites','Varices','Histology']
# target_col = 'class'

# file_name = 'ILPD'
# df_pq = pd.read_csv('dataset/ILPD/ILPD_train.csv')
# df_pq_test = pd.read_csv('dataset/ILPD/ILPD_test.csv')
# categorical_cols =['Gender']
# target_col = 'class'

# file_name = 'blood_transfusion'
# df_pq = pd.read_csv('dataset/blood_transfusion/blood_transfusion_train.csv')
# df_pq_test = pd.read_csv('dataset/blood_transfusion/blood_transfusion_test.csv')
# categorical_cols =[]
# target_col = 'class'

# file_name = 'penguin'
# df_pq = pd.read_csv('dataset/penguin/penguin_train.csv')
# df_pq_test = pd.read_csv('dataset/penguin/penguin_test.csv')
# categorical_cols =['island','sex']
# target_col = 'class'

# file_name = 'iris'
# df_pq = pd.read_csv('dataset/iris/iris_train.csv')
# df_pq_test = pd.read_csv('dataset/iris/iris_test.csv')
# categorical_cols=[]
# target_col = 'class'

# file_name = 'rice'
# df_pq = pd.read_csv('dataset/Rice/Rice_train.csv')
# df_pq_test = pd.read_csv('dataset/Rice/Rice_test.csv')
# categorical_cols = []
# target_col = 'class'


# # TODO diabetes
# df_pq = pd.read_csv('dataset/diabetes/diabetes_train.csv')
# df_pq_test = pd.read_csv('dataset/diabetes/diabetes_test.csv')
# categorical_cols = []
# target_col = 'class'
# file_name = 'diabetes'

# # TODO breast_cancer_wisconsin_prognostic
# df_pq = pd.read_csv('dataset/breast_cancer_wisconsin_prognostic/breast_cancer_wisconsin_prognostic_train.csv')
# df_pq_test = pd.read_csv('dataset/breast_cancer_wisconsin_prognostic/breast_cancer_wisconsin_prognostic_test.csv')
# categorical_cols = []
# target_col = 'class'
# file_name = 'breast_cancer_wisconsin_prognostic'

# # TODO parkinsons
# df_pq = pd.read_csv('dataset/parkinsons/parkinsons_train.csv')
# df_pq_test = pd.read_csv('dataset/parkinsons/parkinsons_test.csv')
# categorical_cols = []
# target_col = 'class'
# file_name = 'parkinsons'

# # TODO car_evaluation
# df_pq = pd.read_csv('dataset/car_evaluation/car_evaluation_train.csv')
# df_pq_test = pd.read_csv('dataset/car_evaluation/car_evaluation_test.csv')
# categorical_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
# target_col = 'class'
# file_name = 'car_evaluation'

# # TODO tic_tac_toe
# df_pq = pd.read_csv('dataset/tic_tac_toe/tic_tac_toe_train.csv')
# df_pq_test = pd.read_csv('dataset/tic_tac_toe/tic_tac_toe_test.csv')
# categorical_cols = [
#     'top-left-square', 'top-middle-square', 'top-right-square',
#     'middle-left-square', 'middle-middle-square', 'middle-right-square',
#     'bottom-left-square', 'bottom-middle-square', 'bottom-right-square'
# ]
# target_col = 'class'
# file_name = 'tic_tac_toe'

# # banknote_authentication
# df_pq = pd.read_csv('dataset/banknote_authentication/banknote_authentication_train.csv')
# df_pq_test = pd.read_csv('dataset/banknote_authentication/banknote_authentication_test.csv')
# categorical_cols = []
# target_col = 'class'
# file_name = 'banknote_authentication'

# # sensor_readings_24
# df_pq = pd.read_csv('dataset/sensor_readings_24/sensor_readings_24_train.csv')
# df_pq_test = pd.read_csv('dataset/sensor_readings_24/sensor_readings_24_test.csv')
# categorical_cols = []
# target_col = 'class'
# file_name = 'sensor_readings_24'

# # image_segmentation
# df_pq = pd.read_csv('dataset/image_segmentation/image_segmentation_train.csv')
# df_pq_test = pd.read_csv('dataset/image_segmentation/image_segmentation_test.csv')
# categorical_cols = []
# target_col = 'class'
# file_name = 'image_segmentation'

# # seismic_bumps
# df_pq = pd.read_csv('dataset/seismic_bumps/seismic_bumps_train.csv')
# df_pq_test = pd.read_csv('dataset/seismic_bumps/seismic_bumps_test.csv')
# categorical_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard']
# target_col = 'class'
# file_name = 'seismic_bumps'

# # statlog
# file_name = 'statlog'
# df_pq = pd.read_csv('dataset/statlog/statlog_train.csv')
# df_pq_test = pd.read_csv('dataset/statlog/statlog_test.csv')
# categorical_cols = [
#     'Attribute1', 'Attribute3', 'Attribute4', 'Attribute6', 'Attribute7',
#     'Attribute9', 'Attribute10', 'Attribute12', 'Attribute14', 'Attribute15',
#     'Attribute17', 'Attribute19', 'Attribute20'
# ]
# target_col = 'class'


# # census_income
# file_name = 'census_income'
# df_pq = pd.read_csv('dataset/census_income/census_income_train.csv')
# df_pq_test = pd.read_csv('dataset/census_income/census_income_test.csv')
# categorical_cols = [
#     'workclass', 'education', 'marital-status', 'occupation',
#     'relationship', 'race', 'sex', 'native-country'
# ]
# target_col = 'class'


# # Bank_Marketing
# file_name = 'Bank_Marketing'
# df_pq = pd.read_csv('dataset/Bank_Marketing/bank_marketing_train.csv')
# df_pq_test = pd.read_csv('dataset/Bank_Marketing/bank_marketing_test.csv')
# categorical_cols = [
#     'job', 'marital', 'education', 'default', 'housing', 'loan',
#     'contact', 'month', 'poutcome'
# ]
# target_col = 'class'

######################################### data ##############################################


if len(categorical_cols) != 0:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = encoder.fit_transform(df_pq[categorical_cols])
    X_test_cat = encoder.transform(df_pq_test[categorical_cols])
    # 변환된 데이터프레임으로 변환
    X_train_cat_df = pd.DataFrame(X_train_cat, columns=encoder.get_feature_names_out(categorical_cols))
    X_test_cat_df = pd.DataFrame(X_test_cat, columns=encoder.get_feature_names_out(categorical_cols))
    # 원본 데이터에서 숫자형 데이터만 선택
    X_train_num = df_pq.drop(columns=categorical_cols)
    X_test_num = df_pq_test.drop(columns=categorical_cols)
    # 숫자형 데이터와 원핫 인코딩 데이터를 합치기
    df_pq = pd.concat([X_train_num.reset_index(drop=True), X_train_cat_df], axis=1)
    df_pq_test = pd.concat([X_test_num.reset_index(drop=True), X_test_cat_df], axis=1)

le = LabelEncoder()
X_train = df_pq.drop(columns=[target_col])
X_train.astype(float)
y_train = df_pq[target_col]
y_train = le.fit_transform(y_train)


X_test = df_pq_test.drop(columns=[target_col])
X_test.astype(float)
y_test = df_pq_test[target_col]
y_test = le.transform(y_test)


# Determine number of features and classes
F = X_train.shape[1]
C = len(np.unique(y_train))  # number of unique classes

# Check if classification (integer labels) or regression (continuous)
treetype = "C"
class_names = le.classes_.tolist()

xmax = pd.concat([X_train], axis=0).max(axis=0).tolist()
xmin = pd.concat([X_train], axis=0).min(axis=0).tolist()
# Construct JSON structure
data_json = {
    "F": F,
    "C": int(C),
    "treetype": treetype,
    "Xtrain": X_train.values.tolist(),
    "Xtest": X_test.values.tolist(),
    "Ytrain": y_train.tolist(),
    "Ytest": y_test.tolist(),
    "Class": class_names,
    "Xmax": xmax,
    "Xmin": xmin
}

# Save as JSON file
output_path = f"{file_name}.json"
with open(output_path, "w") as f:
    json.dump(data_json, f, indent=4)
