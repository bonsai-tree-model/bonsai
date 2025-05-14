from collections import Counter
import pandas as pd




def compute_multiclass_imbalance_ratio(y):
    counter = Counter(y)
    majority = max(counter.values())
    minority = min(counter.values())
    ratio = majority / minority
    print(f"Class distribution: {dict(counter)}")
    print(f"Imbalance Ratio (max/min): {ratio:.4f}")
    return ratio


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
# target_col = 'class'

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

# # iris
# df_pq = pd.read_csv('dataset/iris/iris_train.csv')
# df_pq_test = pd.read_csv('dataset/iris/iris_test.csv')
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


# Bank_Marketing
df_pq = pd.read_csv('dataset/Bank_Marketing/bank_marketing_train.csv')
df_pq_test = pd.read_csv('dataset/Bank_Marketing/bank_marketing_test.csv')
categorical_cols = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'poutcome'
]
target_col = 'class'







####################################### data #############################################################3

df_all = pd.concat([df_pq, df_pq_test], axis=0, ignore_index=True)
y = df_all[target_col]
compute_multiclass_imbalance_ratio(y)

