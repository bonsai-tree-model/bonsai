import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from chefboost import Chefboost as Chef
import pandas as pd
from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier

# # Rice
# df_pq = pd.read_csv('dataset/Rice/Rice_train.csv')
# df_pq_test = pd.read_csv('dataset/Rice/Rice_test.csv')
# categorical_cols = []
# attributes_map = {
#     'Area': 'continuous',
#     'Perimeter': 'continuous',
#     'Major_Axis_Length': 'continuous',
#     'Minor_Axis_Length': 'continuous',
#     'Eccentricity': 'continuous',
#     'Convex_Area': 'continuous',
#     'Extent': 'continuous'
# }
# target_col = "class"


# # penguns
# df_pq = pd.read_csv('dataset/penguin/penguin_train.csv')
# df_pq_test = pd.read_csv('dataset/penguin/penguin_test.csv')
# categorical_cols = ['island', 'sex']
# attributes_map = {
#     'culmen_length_mm': 'continuous',
#     'culmen_depth_mm': 'continuous',
#     'flipper_length_mm': 'continuous',
#     'body_mass_g': 'continuous',
#     'culmen_ratio': 'continuous',
#     'island': 'categorical',
#     'sex': 'categorical'
# }
# target_col = "class"


# #blood_transfusion
# df_pq = pd.read_csv('dataset/blood_transfusion/blood_transfusion_train.csv')
# df_pq_test = pd.read_csv('dataset/blood_transfusion/blood_transfusion_test.csv')
# categorical_cols = []
# attributes_map = {
#     'Recency': 'continuous',
#     'Frequency': 'continuous',
#     'Monetary': 'continuous',
#     'Time': 'continuous'
# }
# target_col = "class"

# # heart
# df_pq = pd.read_csv('dataset/heart/heart_train.csv')
# df_pq_test = pd.read_csv('dataset/heart/heart_test.csv')
# categorical_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
# attributes_map = {
#     'age': 'continuous',
#     'anaemia': 'categorical',
#     'creatinine_phosphokinase': 'continuous',
#     'diabetes': 'categorical',
#     'ejection_fraction': 'continuous',
#     'high_blood_pressure': 'categorical',
#     'platelets': 'continuous',
#     'serum_creatinine': 'continuous',
#     'serum_sodium': 'continuous',
#     'sex': 'categorical',
#     'smoking': 'categorical',
#     'time': 'continuous'
# }
# target_col = "class"

# # zoo
# df_pq = pd.read_csv('dataset/zoo/zoo_train.csv')
# df_pq_test = pd.read_csv('dataset/zoo/zoo_test.csv')
# categorical_cols = []
# attributes_map = {
#     'hair': 'continuous',
#     'feathers': 'continuous',
#     'eggs': 'continuous',
#     'milk': 'continuous',
#     'airborne': 'continuous',
#     'aquatic': 'continuous',
#     'predator': 'continuous',
#     'toothed': 'continuous',
#     'backbone': 'continuous',
#     'breathes': 'continuous',
#     'venomous': 'continuous',
#     'fins': 'continuous',
#     'legs': 'continuous',
#     'tail': 'continuous',
#     'domestic': 'continuous',
#     'catsize': 'continuous'
# }
# target_col = "class"


# # Wholesale
# df_pq = pd.read_csv('dataset/Wholesale/Wholesale_train.csv')
# df_pq_test = pd.read_csv('dataset/Wholesale/Wholesale_test.csv')
# categorical_cols = ['Channel']
# attributes_map = {
#     'Channel': 'categorical',
#     'Fresh': 'continuous',
#     'Milk': 'continuous',
#     'Grocery': 'continuous',
#     'Frozen': 'continuous',
#     'Detergents_Paper': 'continuous',
#     'Delicassen': 'continuous'
# }
# target_col = "class"


# # wine
# df_pq = pd.read_csv('dataset/wine/wine_train.csv')
# df_pq_test = pd.read_csv('dataset/wine/wine_test.csv')
# categorical_cols = []
# attributes_map = {
#     'Proline': 'continuous',
#     'Alcohol': 'continuous',
#     'Malicacid': 'continuous',
#     'Ash': 'continuous',
#     'Alcalinity_of_ash': 'continuous',
#     'Magnesium': 'continuous',
#     'Total_phenols': 'continuous',
#     'Flavanoids': 'continuous',
#     'Nonflavanoid_phenols': 'continuous',
#     'Proanthocyanins': 'continuous',
#     'Color_intensity': 'continuous',
#     'Hue': 'continuous',
#     '0D280_0D315_of_diluted_wines': 'continuous'
# }
# target_col = "class"



# # contraceptive
# df_pq = pd.read_csv('dataset/contraceptive/contraceptive_train.csv')
# df_pq_test = pd.read_csv('dataset/contraceptive/contraceptive_test.csv')
# categorical_cols = ['wife_edu', 'husband_edu', 'wife_religion', 'wife_working', 'husband_occupation', 'standard_of_living_index', 'media_exposure']
# attributes_map = {
#     'wife_age': 'continuous',
#     'wife_edu': 'categorical',
#     'husband_edu': 'categorical',
#     'num_children': 'continuous',
#     'wife_religion': 'categorical',
#     'wife_working': 'categorical',
#     'husband_occupation': 'categorical',
#     'standard_of_living_index': 'categorical',
#     'media_exposure': 'categorical'
# }
# target_col = "class"


# # credit_approval
# df_pq = pd.read_csv('dataset/credit_approval/credit_approval_train.csv')
# df_pq = df_pq.dropna().reset_index(drop=True)
# df_pq_test = pd.read_csv('dataset/credit_approval/credit_approval_test.csv')
# df_pq_test = df_pq_test.dropna().reset_index(drop=True)
# attributes_map = {
#     "A1": "categorical",
#     "A2": "continuous",
#     "A3": "continuous",
#     "A4": "categorical",
#     "A5": "categorical",
#     "A6": "categorical",
#     "A7": "categorical",
#     "A8": "continuous",
#     "A9": "categorical",
#     "A10": "categorical",
#     "A11": "continuous",
#     "A12": "categorical",
#     "A13": "categorical",
#     "A14": "continuous",
#     "A15": "continuous"
# }
# target_col = "class"

# # hepatitis
# df_pq = pd.read_csv('dataset/hepatitis/hepatitis_train.csv')
# df_pq = df_pq.dropna().reset_index(drop=True)
# df_pq_test = pd.read_csv('dataset/hepatitis/hepatitis_test.csv')
# df_pq_test = df_pq_test.dropna().reset_index(drop=True)
# categorical_cols = ['Sex','Steroid','Antivirals','Fatigue','Malaise','Anorexia','Liver_Big','Liver_Firm','Spleen_Palpable','Spiders','Ascites','Varices','Histology']
# attributes_map = {
#     "Age": "continuous",
#     "Sex": "categorical",
#     "Steroid": "categorical",
#     "Antivirals": "categorical",
#     "Fatigue": "categorical",
#     "Malaise": "categorical",
#     "Anorexia": "categorical",
#     "Liver_Big": "categorical",
#     "Liver_Firm": "categorical",
#     "Spleen_Palpable": "categorical",
#     "Spiders": "categorical",
#     "Ascites": "categorical",
#     "Varices": "categorical",
#     "Bilirubin": "continuous",
#     "Alk_Phosphate": "continuous",
#     "Sgot": "continuous",
#     "Albumin": "continuous",
#     "Protime": "continuous",
#     "Histology": "categorical"
# }
# target_col = "class"

# # ILPD
# df_pq = pd.read_csv('dataset/ILPD/ILPD_train.csv')
# df_pq_test = pd.read_csv('dataset/ILPD/ILPD_test.csv')
# attributes_map = {
#     "Age": "continuous",
#     "Gender": "categorical",
#     "TB": "continuous",
#     "DB": "continuous",
#     "Alkphos": "continuous",
#     "Sgpt": "continuous",
#     "Sgot": "continuous",
#     "TP": "continuous",
#     "ALB": "continuous",
#     "A_G_Ratio": "continuous"
# }
# target_col = "class"


# # iris
# df_pq = pd.read_csv('dataset/Iris/iris_train.csv')
# df_pq_test = pd.read_csv('dataset/Iris/iris_test.csv')
# categorical_cols = []
# attributes_map = {
#     'sepal_length': 'continuous',
#     'sepal_width': 'continuous',
#     'petal_length': 'continuous',
#     'petal_width': 'continuous'
# }
# target_col = "class"

# # diabetes
# df_pq = pd.read_csv('dataset/diabetes/diabetes_train.csv')
# df_pq_test = pd.read_csv('dataset/diabetes/diabetes_test.csv')
# categorical_cols = []
# attributes_map = {
#     'Pregnancies': 'continuous',
#     'Glucose': 'continuous',
#     'BloodPressure': 'continuous',
#     'SkinThickness': 'continuous',
#     'Insulin': 'continuous',
#     'BMI': 'continuous',
#     'DiabetesPedigreeFunction': 'continuous',
#     'Age': 'continuous'
# }
# target_col = "class"

# # breast_cancer_wisconsin_prognostic
# df_pq = pd.read_csv('dataset/breast_cancer_wisconsin_prognostic/breast_cancer_wisconsin_prognostic_train.csv')
# df_pq_test = pd.read_csv('dataset/breast_cancer_wisconsin_prognostic/breast_cancer_wisconsin_prognostic_test.csv')
# categorical_cols = []
# attributes_map = {
#     'Time': 'continuous',
#     'radius1': 'continuous',
#     'texture1': 'continuous',
#     'perimeter1': 'continuous',
#     'area1': 'continuous',
#     'smoothness1': 'continuous',
#     'compactness1': 'continuous',
#     'concavity1': 'continuous',
#     'concave_points1': 'continuous',
#     'symmetry1': 'continuous',
#     'fractal_dimension1': 'continuous',
#     'radius2': 'continuous',
#     'texture2': 'continuous',
#     'perimeter2': 'continuous',
#     'area2': 'continuous',
#     'smoothness2': 'continuous',
#     'compactness2': 'continuous',
#     'concavity2': 'continuous',
#     'concave_points2': 'continuous',
#     'symmetry2': 'continuous',
#     'fractal_dimension2': 'continuous',
#     'radius3': 'continuous',
#     'texture3': 'continuous',
#     'perimeter3': 'continuous',
#     'area3': 'continuous',
#     'smoothness3': 'continuous',
#     'compactness3': 'continuous',
#     'concavity3': 'continuous',
#     'concave_points3': 'continuous',
#     'symmetry3': 'continuous',
#     'fractal_dimension3': 'continuous',
#     'tumor_size': 'continuous',
#     'lymph_node_status': 'continuous'
# }
# target_col = "class"

# # parkinsons
# df_pq = pd.read_csv('dataset/parkinsons/parkinsons_train.csv')
# df_pq_test = pd.read_csv('dataset/parkinsons/parkinsons_test.csv')
# categorical_cols = []
# attributes_map = {
#     'Fo': 'continuous',
#     'Fhi': 'continuous',
#     'Flo': 'continuous',
#     'Jitter': 'continuous',
#     'Jitter_1': 'continuous',
#     'RAP': 'continuous',
#     'PPQ': 'continuous',
#     'DDP': 'continuous',
#     'Shimmer': 'continuous',
#     'Shimmer_1': 'continuous',
#     'APQ3': 'continuous',
#     'APQ5': 'continuous',
#     'APQ': 'continuous',
#     'DDA': 'continuous',
#     'NHR': 'continuous',
#     'HNR': 'continuous',
#     'RPDE': 'continuous',
#     'DFA': 'continuous',
#     'spread1': 'continuous',
#     'spread2': 'continuous',
#     'D2': 'continuous',
#     'PPE': 'continuous'
# }
# target_col = "class"

# # tic_tac_toe
# df_pq = pd.read_csv('dataset/tic_tac_toe/tic_tac_toe_train.csv')
# df_pq_test = pd.read_csv('dataset/tic_tac_toe/tic_tac_toe_test.csv')
# categorical_cols = [
#     'top-left-square', 'top-middle-square', 'top-right-square',
#     'middle-left-square', 'middle-middle-square', 'middle-right-square',
#     'bottom-left-square', 'bottom-middle-square', 'bottom-right-square'
# ]
# attributes_map = {col: 'categorical' for col in categorical_cols}
# target_col = "class"

# # car_evaluation
# df_pq = pd.read_csv('dataset/car_evaluation/car_evaluation_train.csv')
# df_pq_test = pd.read_csv('dataset/car_evaluation/car_evaluation_test.csv')
# categorical_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
# attributes_map = {col: 'categorical' for col in categorical_cols}
# target_col = "class"

############################ data ################################################
# # banknote_authentication
# df_pq = pd.read_csv('dataset/banknote_authentication/banknote_authentication_train.csv')
# df_pq_test = pd.read_csv('dataset/banknote_authentication/banknote_authentication_test.csv')
# attributes_map = {
#     'variance': 'continuous',
#     'skewness': 'continuous',
#     'curtosis': 'continuous',
#     'entropy': 'continuous'
# }
# target_col = 'class'


# # sensor_readings_24
# df_pq = pd.read_csv('dataset/sensor_readings_24/sensor_readings_24_train.csv')
# df_pq_test = pd.read_csv('dataset/sensor_readings_24/sensor_readings_24_test.csv')
# attributes_map = {
#     'US1': 'continuous',
#     'US2': 'continuous',
#     'US3': 'continuous',
#     'US4': 'continuous',
#     'US5': 'continuous',
#     'US6': 'continuous',
#     'US7': 'continuous',
#     'US8': 'continuous',
#     'US9': 'continuous',
#     'US10': 'continuous',
#     'US11': 'continuous',
#     'US12': 'continuous',
#     'US13': 'continuous',
#     'US14': 'continuous',
#     'US15': 'continuous',
#     'US16': 'continuous',
#     'US17': 'continuous',
#     'US18': 'continuous',
#     'US19': 'continuous',
#     'US20': 'continuous',
#     'US21': 'continuous',
#     'US22': 'continuous',
#     'US23': 'continuous',
#     'US24': 'continuous'
# }
# target_col = 'class'


# # image_segmentation
# df_pq = pd.read_csv('dataset/image_segmentation/image_segmentation_train.csv')
# df_pq_test = pd.read_csv('dataset/image_segmentation/image_segmentation_test.csv')
# attributes_map = {
#     'region_centroid_col': 'continuous',
#     'region_centroid_row': 'continuous',
#     'region_pixel_count': 'continuous',
#     'short_line_density_5': 'continuous',
#     'short_line_density_2': 'continuous',
#     'vedge_mean': 'continuous',
#     'vegde_sd': 'continuous',
#     'hedge_mean': 'continuous',
#     'hedge_sd': 'continuous',
#     'intensity_mean': 'continuous',
#     'rawred_mean': 'continuous',
#     'rawblue_mean': 'continuous',
#     'rawgreen_mean': 'continuous',
#     'exred_mean': 'continuous',
#     'exblue_mean': 'continuous',
#     'exgreen_mean': 'continuous',
#     'value_mean': 'continuous',
#     'saturatoin_mean': 'continuous',
#     'hue_mean': 'continuous'
# }
# target_col = 'class'


# # seismic_bumps
# df_pq = pd.read_csv('dataset/seismic_bumps/seismic_bumps_train.csv')
# df_pq_test = pd.read_csv('dataset/seismic_bumps/seismic_bumps_test.csv')
# attributes_map = {
#     'seismic': 'categorical',
#     'seismoacoustic': 'categorical',
#     'shift': 'categorical',
#     'genergy': 'continuous',
#     'gpuls': 'continuous',
#     'gdenergy': 'continuous',
#     'gdpuls': 'continuous',
#     'ghazard': 'categorical',
#     'nbumps': 'continuous',
#     'nbumps2': 'continuous',
#     'nbumps3': 'continuous',
#     'nbumps4': 'continuous',
#     'nbumps5': 'continuous',
#     'nbumps6': 'continuous',
#     'nbumps7': 'continuous',
#     'nbumps89': 'continuous',
#     'energy': 'continuous',
#     'maxenergy': 'continuous'
# }
# target_col = 'class'

# df_pq = pd.read_csv('dataset/statlog/statlog_train.csv')
# df_pq_test = pd.read_csv('dataset/statlog/statlog_test.csv')
# attributes_map = {
#     'Attribute1': 'categorical',       # checking_account_status (A11-A14)
#     'Attribute2': 'continuous',         # duration (months)
#     'Attribute3': 'categorical',       # credit_history (A30-A34)
#     'Attribute4': 'categorical',       # purpose (A40-A49)
#     'Attribute5': 'continuous',         # credit_amount
#     'Attribute6': 'categorical',       # savings_account/bonds (A61-A65)
#     'Attribute7': 'categorical',       # present_employment_since (A71-A75)
#     'Attribute8': 'continuous',         # installment_rate
#     'Attribute9': 'categorical',       # personal_status_and_sex (A91-A95)
#     'Attribute10': 'categorical',      # other_debtors (A101-A103)
#     'Attribute11': 'continuous',        # present_residence_since
#     'Attribute12': 'categorical',      # property (A121-A124)
#     'Attribute13': 'continuous',        # age
#     'Attribute14': 'categorical',      # other_installment_plans (A141-A143)
#     'Attribute15': 'categorical',      # housing (A151-A153)
#     'Attribute16': 'continuous',        # number_of_existing_credits
#     'Attribute17': 'categorical',      # job (A171-A174)
#     'Attribute18': 'continuous',        # number_of_dependents
#     'Attribute19': 'categorical',      # telephone (A191-A192)
#     'Attribute20': 'categorical',      # foreign_worker (A201-A202)
# }
# target_col = 'class'

# # census_income
# df_pq = pd.read_csv('dataset/census_income/census_income_train.csv')
# df_pq = df_pq.dropna().reset_index(drop=True)
# df_pq_test = pd.read_csv('dataset/census_income/census_income_test.csv')
# df_pq_test = df_pq_test.dropna().reset_index(drop=True)
# attributes_map = {
#     'age': 'continuous',                  # 나이 (예: 39)
#     'workclass': 'categorical',          # 직군 (예: State-gov, Private 등)
#     'fnlwgt': 'continuous',               # 가중치 (final weight)
#     'education': 'categorical',          # 교육 수준 (예: Bachelors, HS-grad)
#     'education-num': 'continuous',        # 교육 연수 (예: 13)
#     'marital-status': 'categorical',     # 결혼 상태 (예: Never-married, Married)
#     'occupation': 'categorical',         # 직업 (예: Adm-clerical, Exec-managerial)
#     'relationship': 'categorical',       # 가족 관계 (예: Husband, Not-in-family)
#     'race': 'categorical',               # 인종 (예: White, Black, Asian-Pac-Islander)
#     'sex': 'categorical',                # 성별 (Male, Female)
#     'capital-gain': 'continuous',         # 자본 수익 (예: 2174)
#     'capital-loss': 'continuous',         # 자본 손실 (예: 0)
#     'hours-per-week': 'continuous',       # 주당 근무시간 (예: 40)
#     'native-country': 'categorical',     # 출생 국가 (예: United-States, Mexico)
# }
# target_col = 'class'
# df_pq['class'] = df_pq['class'].str.strip()
# df_pq['class'] = df_pq['class'].map({'>50K': 1, '<=50K': 0})
# df_pq_test['class'] = df_pq_test['class'].str.strip()
# df_pq_test['class'] = df_pq_test['class'].map({'>50K': 1, '<=50K': 0})



# # bank_marketing
# df_pq = pd.read_csv('dataset/Bank_Marketing/bank_marketing_train.csv')
# df_pq_test = pd.read_csv('dataset/Bank_Marketing/bank_marketing_test.csv')
# df_pq = df_pq.dropna().reset_index(drop=True)
# df_pq_test = df_pq_test.dropna().reset_index(drop=True)
# attributes_map = {
#     'age': 'continuous',                  # 나이 (예: 58)
#     'job': 'categorical',                # 직업 (예: management, technician, unemployed 등)
#     'marital': 'categorical',            # 결혼 상태 (예: married, single, divorced)
#     'education': 'categorical',          # 교육 수준 (예: primary, secondary, tertiary)
#     'default': 'categorical',            # 신용 불량 여부 (예: yes, no)
#     'balance': 'continuous',              # 계좌 잔고 (예: 2143)
#     'housing': 'categorical',            # 주택 융자 여부 (예: yes, no)
#     'loan': 'categorical',               # 개인 융자 여부 (예: yes, no)
#     'contact': 'categorical',            # 연락 수단 (예: cellular, telephone)
#     'day_of_week': 'continuous',        # 연락 일
#     'month': 'categorical',              # 연락 월 (예: may, jun, ...)
#     'duration': 'continuous',             # 마지막 통화 지속 시간 (초)
#     'campaign': 'continuous',             # 현재 캠페인에서의 연락 횟수
#     'pdays': 'continuous',                # 과거 캠페인 이후 경과 일수 (-1은 연락 없음)
#     'previous': 'continuous',             # 과거 캠페인 동안의 연락 횟수
#     'poutcome': 'categorical',           # 과거 캠페인 결과 (예: success, failure, nonexistent)
# }
# target_col = 'class'




X_train = df_pq.rename(columns={target_col: 'target'})
y_train = df_pq[target_col]
X_test = df_pq_test.drop(columns=[target_col])
y_test = df_pq_test[target_col]

# ,min_instances=10,node_purity=0.95 penguin
decision_tree = DecisionTreeClassifier(attributes_map,max_depth=3,node_purity=0.95)
decision_tree.fit(X_train)

X_train_input = X_train.drop(columns=['target'])
y_test_pred = decision_tree.predict(X_test)
y_train_pred = decision_tree.predict(X_train_input)


accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, average='macro')

print("\nTrain Set Evaluation")
print(f"train_Accuracy: {accuracy_train:.4f}")
print(f"test_Accuracy: {accuracy:.4f}")
print(f"test_Macro F1 Score: {f1:.4f}")



decision_tree.view(folder_name='figures', title='Quinlan-Tree')