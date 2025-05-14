#! /usr/bin/env python3
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_string_dtype, is_bool
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

def check_if_categorical(feature):
    return (
            is_categorical_dtype(feature) or
            is_string_dtype(feature) or
            is_bool(feature)
    )


class Node:

    def get_indent(level):
        return "    " * level

    def __init__(self, data, target,
                 value="", min_samples_leaf=3, level=1, max_depth=3):
        """initializes a Node object

        Arguments:
            data {pd.DataFrame} -- training data for the tree
            target {str} -- name of the target feature

        Keyword Arguments:
            value {str} -- value to display for node (default: {""})
            min_samples_leaf {int} -- minimum number of examples at a leaf node (default: {3})
            level {int} -- depth level of node in the tree (default: {1})
            max_depth {int} -- maximum depth of the tree (default: {3})
        """

        self.data = data
        self.target = target
        self.value = value
        self.min_samples_leaf = min_samples_leaf
        self.left = None
        self.right = None
        self.prediction = None
        self.level = level
        self.max_depth = max_depth
        self.leaves = []

    def display(self):
        lines, _, _, _ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):  # stolen from https://stackoverflow.com/a/54074933/8650928
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.value
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.value
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.value
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.value
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    def get_gini(self, subset):
        """takes the vector of targets as input

        Arguments:
            subset {pd.DataFrame} -- the left side of the split data

        Returns:
            [float] -- gini index for given data subset
        """

        proportions = subset[self.target].value_counts(normalize=True)
        return 1 - (proportions ** 2).sum()

    def get_delta_i(self, subset):
        """gets the delta i for a given split

        Arguments:
            subset {pd.DataFrame} -- the left side of the split data

        Returns:
            [float] -- delta i for this split
        """

        gini = self.get_gini(self.data)

        left = subset
        right = self.data.drop(subset.index, axis=0)

        p_left = len(left) / len(self.data)
        p_right = 1 - p_left

        sub_left = p_left * self.get_gini(left)
        sub_right = p_right * self.get_gini(right)

        return gini - sub_left - sub_right

    def get_categorical_splits(self, feature):
        splits = {}
        for unique in self.data[feature].unique():
            splits[(feature, unique, 'categorical')] = self.data[
                self.data[feature] == unique]
        return splits

    def get_numerical_splits(self, feature):
        splits = {}
        uniques = self.data[feature].unique()
        for value in uniques:
            if value != max(uniques):
                splits[(feature, value, 'numerical')] = self.data[
                    self.data[feature] <= value]
        return splits

    def get_splits(self):
        features = self.data.columns.drop(self.target)
        all_splits = {}

        for feature in features:
            if check_if_categorical(self.data[feature]):
                all_splits.update(self.get_categorical_splits(feature))
            else:
                all_splits.update(self.get_numerical_splits(feature))
        return all_splits

    def get_best_split(self):
        all_splits = self.get_splits()
        delta_is = {}

        for key, split in all_splits.items():
            delta_is[key] = self.get_delta_i(split)

        return max(delta_is, key=delta_is.get)

    def is_pure(self):
        return len(self.data[self.target].unique()) == 1

    def too_small(self):
        return len(self.data) <= self.min_samples_leaf

    def too_deep(self):
        return self.level >= self.max_depth

    def no_splits(self):
        return self.get_splits() == {}

    def split(self):
        """Recursive function, that finds the best possible feature to split on in the dataset and creates a child node for each possible value of that feature.
        """

        if (self.is_pure() or self.too_deep() or
                self.no_splits() or self.too_small()):  # stop condition

            self.prediction = self.data[self.target].value_counts().idxmax()
            self.value = ' ({})'.format(self.prediction)
            return

        best_split = self.get_best_split()

        self.split_feature = best_split[0]
        self.split_value = best_split[1]
        self.split_type = best_split[2]

        if self.split_type == 'categorical':
            left_data = self.data[
                self.data[self.split_feature] == self.split_value]
            right_data = self.data[
                self.data[self.split_feature] != self.split_value]
            self.value = "{} = {}".format(
                self.split_feature, self.split_value
            )

        elif self.split_type == 'numerical':
            left_data = self.data[
                self.data[self.split_feature] <= self.split_value
                ]
            right_data = self.data[
                self.data[self.split_feature] > self.split_value
                ]
            self.value = "{} <= {}".format(
                self.split_feature, self.split_value
            )
        else:
            raise ValueError('splits can be either numerical or categorical')

        child_params = {
            'target': self.target,
            'min_samples_leaf': self.min_samples_leaf,
            'max_depth': self.max_depth,
            'level': self.level + 1
        }

        self.left = Node(left_data, **child_params)
        self.right = Node(right_data, **child_params)

        self.left.split()
        self.right.split()

        return

    def get_leaves(self):
        if self.left is None and self.right is None:
            return [self]
        if self.leaves != []:
            return self.leaves

        if self.left is not None:
            self.leaves.extend(self.left.get_leaves())
        if self.right is not None:
            self.leaves.extend(self.right.get_leaves())

        return self.leaves

    def count_leaves(self):
        self.get_leaves()
        return len(self.leaves)

    def predict(self, row):
        """하나의 샘플에 대해 예측을 수행하는 메소드.

        인자로 들어오는 row는 pd.Series 타입이어야 하며,
        트리의 분할 조건에 따라 적절한 자식 노드로 이동합니다.
        """
        # 리프 노드인 경우, 예측값을 반환합니다.
        if self.left is None and self.right is None:
            return self.prediction

        # 분할 기준에 따라 왼쪽 또는 오른쪽 서브트리로 이동합니다.
        if self.split_type == 'categorical':
            if row[self.split_feature] == self.split_value:
                return self.left.predict(row)
            else:
                return self.right.predict(row)
        elif self.split_type == 'numerical':
            if row[self.split_feature] <= self.split_value:
                return self.left.predict(row)
            else:
                return self.right.predict(row)
        else:
            raise ValueError('알 수 없는 split type입니다.')

    def predict_df(self, X):
        """여러 샘플에 대해 예측을 수행하는 메소드.

        X는 각 행이 하나의 샘플인 pd.DataFrame입니다.
        """
        return X.apply(self.predict, axis=1)


if __name__ == "__main__":

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
############################################## data #############################
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


    tree_model = Node(df_pq, target = target_col, max_depth=4)
    tree_model.split()


    X_train = df_pq.drop(columns=[target_col])
    y_train = df_pq[target_col]
    X_test = df_pq_test.drop(columns=[target_col])
    y_test = df_pq_test[target_col]

    pred_train = tree_model.predict_df(X_train)
    pred_test = tree_model.predict_df(X_test)
    tree_model.display()


    # 평가 지표 출력
    accuracy = accuracy_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test, average='macro')


    accuracy_train = accuracy_score(y_train, pred_train)
    f1_train = f1_score(y_train, pred_train, average='macro')

    print("\n📘 Train Set Evaluation")
    print(f"train_Accuracy: {accuracy_train:.4f}")
    print(f"test_Accuracy: {accuracy:.4f}")
    print(f"test_Macro F1 Score: {f1:.4f}")
