import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def rule_penguin(x_row):
    if x_row['num_children'] < 2.5:
        if x_row['num_children'] < 0.5:
            if x_row['wife_age'] < 17.5:
                return 3
            else:
                return 1
        else:
            if x_row['wife_age'] < 22.5:
                return 3
            else:
                return 1
    else:
        if x_row['wife_edu_4'] < 0.5:
            if x_row['wife_age'] < 37.5:
                return 3
            else:
                return 1
        else:
            if x_row['wife_age'] < 32.5:
                return 3
            else:
                return 2


if __name__ == '__main__':
    file_name = 'contraceptive'
    df_pq = pd.read_csv('dataset/contraceptive/contraceptive_train.csv')
    df_pq_test = pd.read_csv('dataset/contraceptive/contraceptive_test.csv')
    categorical_cols = ['wife_edu', 'husband_edu', 'wife_religion', 'wife_working', 'husband_occupation', 'standard_of_living_index', 'media_exposure']
    target_col = 'class'
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

    y_pred_train = [rule_penguin(row) for _, row in X_train.iterrows()]

    print("train_Accuracy:", accuracy_score(y_train, y_pred_train))


    y_pred_test = [rule_penguin(row) for _, row in X_test.iterrows()]
    print("test_Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Macro F1 Score:", f1_score(y_test, y_pred_test, average='macro'))
