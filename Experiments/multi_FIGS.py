from sklearn.multiclass import OneVsRestClassifier   # or OneVsOneClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imodels import FIGSClassifier, get_clean_dataset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# # Zoo
# df_pq = pd.read_csv('dataset/zoo/zoo_train.csv')
# df_pq_test = pd.read_csv('dataset/zoo/zoo_test.csv')
# # categorical_cols =['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic',
# #  'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins',
# #  'legs', 'tail', 'domestic', 'catsize']
# categorical_cols = []
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
# target_col = 'class'

# # contraceptive
df_pq = pd.read_csv('dataset/contraceptive/contraceptive_train.csv')
df_pq_test = pd.read_csv('dataset/contraceptive/contraceptive_test.csv')
categorical_cols = ['wife_edu','husband_edu','wife_religion','wife_working','husband_occupation','standard_of_living_index','media_exposure']
target_col = 'class'

# # credit_approval
# df_pq = pd.read_csv('dataset/credit_approval/credit_approval_train.csv')
# df_pq = df_pq.dropna().reset_index(drop=True)
# df_pq_test = pd.read_csv('dataset/credit_approval/credit_approval_test.csv')
# df_pq_test = df_pq_test.dropna().reset_index(drop=True)
# categorical_cols =['A1','A4','A5','A6','A7','A9','A10','A12','A13']
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
# df_pq = df_pq.dropna().reset_index(drop=True)
# df_pq_test = pd.read_csv('dataset/ILPD/ILPD_test.csv')
# df_pq_test  = df_pq_test .dropna().reset_index(drop=True)
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

# # iris
# df_pq = pd.read_csv('dataset/iris/iris_train.csv')
# df_pq_test = pd.read_csv('dataset/iris//iris_test.csv')
# categorical_cols=[]
# target_col = 'class'

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

# # TODO breast_cancer_wisconsin_prognostic
# df_pq = pd.read_csv('dataset/breast_cancer_wisconsin_prognostic/breast_cancer_wisconsin_prognostic_train.csv')
# df_pq = df_pq.dropna().reset_index(drop=True)
# df_pq_test = pd.read_csv('dataset/breast_cancer_wisconsin_prognostic/breast_cancer_wisconsin_prognostic_test.csv')
# df_pq_test = df_pq_test.dropna().reset_index(drop=True)
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

# # statlog
# df_pq = pd.read_csv('dataset/statlog/statlog_train.csv')
# df_pq_test = pd.read_csv('dataset/statlog/statlog_test.csv')
# categorical_cols = [
#     'Attribute1', 'Attribute3', 'Attribute4', 'Attribute6', 'Attribute7',
#     'Attribute9', 'Attribute10', 'Attribute12', 'Attribute14', 'Attribute15',
#     'Attribute17', 'Attribute19', 'Attribute20'
# ]
# target_col = 'class'


# # census_income
# df_pq = pd.read_csv('dataset/census_income/census_income_train.csv')
# df_pq_test = pd.read_csv('dataset/census_income/census_income_test.csv')
# categorical_cols = [
#     'workclass', 'education', 'marital-status', 'occupation',
#     'relationship', 'race', 'sex', 'native-country'
# ]
# target_col = 'class'


# # Bank_Marketing
# df_pq = pd.read_csv('dataset/Bank_Marketing/bank_marketing_train.csv')
# df_pq_test = pd.read_csv('dataset/Bank_Marketing/bank_marketing_test.csv')
# categorical_cols = [
#     'job', 'marital', 'education', 'default', 'housing', 'loan',
#     'contact', 'month', 'poutcome'
# ]
# target_col = 'class'

####################### data ##################
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


X_train = df_pq.drop(columns=[target_col])
y_train = df_pq[target_col]
X_test = df_pq_test.drop(columns=[target_col])
y_test = df_pq_test[target_col]

# 3. wrap the binary learner
base_figs = FIGSClassifier(max_rules=10, max_trees=3, random_state=42)
clf = OneVsRestClassifier(base_figs)   # OvR is usually simplest

# 4. fit & predict
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
preds_proba = clf.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

y_pred_train = clf.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
f1_train = f1_score(y_train, y_pred_train, average='macro')

print(clf.classes_)
print("\n📘 Train Set Evaluation")
print(f"train_Accuracy: {accuracy_train:.4f}")
print(f"test_Accuracy: {accuracy:.4f}")
print(f"test_Macro F1 Score: {f1:.4f}")
for i, figs_model in enumerate(clf.estimators_):
    figs_model.plot(feature_names=X_train.columns, filename=f'out{i}.svg', dpi=300,impurity=True)