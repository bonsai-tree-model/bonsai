import importlib
import bonsai_ml

if __name__ == "__main__":
    df_pq = pd.read_csv('dataset/penguin/penguin_train.csv')
    df_pq_test = pd.read_csv('dataset/penguin/penguin_test.csv')
    attributes = {}
    attributes['Decision'] = 'Output'
    attributes['island'] = 'Categorical'
    attributes['culmen_length_mm'] = 'Numerical'
    attributes['culmen_depth_mm'] = 'Numerical'
    attributes['flipper_length_mm'] = 'Numerical'
    attributes['body_mass_g'] = 'Numerical'
    attributes['sex'] = 'Categorical'
    attributes['culmen_ratio'] = 'Numerical' # Extracted Feature

    max_nodes = 20
    point = list(range(1,3))
    df_pq_copy = copy.deepcopy(df_pq)
    bonsai_ml.RANGE_TOLERANCE = 1.0
    bonsai_ml.SAMPLE_RATIO = 0.001
    importlib.invalidate_caches()
    lazy_rule = lazy_import('rule')
    accuracy, tree = bonsai_ml.train(df_pq_copy, attributes, rule_path='rule', tree_path='nodes.tree', min_sample=2)
    save_tree(tree, 'rule_test')