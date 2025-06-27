import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, hamming_loss, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from lazypredict.Supervised import LazyClassifier
from xgboost import XGBClassifier
from pipeline_utils import run_multilabel_baseline

from skmultilearn.problem_transform import LabelPowerset  

from preprocess import preprocess_data
from pipeline_utils import (
    run_multilabel_baseline,
    get_labelpowerset_feature_importance,
    run_labelpowerset_gridsearch
)

file_path = "/Users/kegong/Desktop/Work/studiendaten_f__r_similarity_analyse_neu.xlsx"
df_filled, df_encoded, reduced_df = preprocess_data(file_path)



target_column = 'Wirkstoffschema adjuvante Therapie'
X_lazy = df_encoded[reduced_df.columns]
y_lazy = df_encoded[target_column]

X_train, X_test, y_train, y_test = train_test_split(X_lazy, y_lazy, test_size=0.4, random_state=42)
clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, _ = clf.fit(X_train, X_test, y_train, y_test)
print("\nLazyClassifier Results")
print(models)

col1 = 'Wirkstoffschema neoadjuvante Therapie'
col2 = 'Wirkstoffschema adjuvante Therapie'
multi_target_cols = [col1, col2]

y_raw = df_encoded[multi_target_cols].astype(str)
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y_raw)

X_multi = df_encoded[reduced_df.columns]
X_multi = StandardScaler().fit_transform(X_multi)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.4, random_state=42)


baseline_results = run_multilabel_baseline(
    X={'train': X_train_m, 'test': X_test_m},
    y={'train': y_train_m, 'test': y_test_m}
)
print("\nMulti-Label Baseline Comparison")
print(baseline_results)


lp_model = LabelPowerset(RandomForestClassifier(n_estimators=100, random_state=42))
lp_model.fit(X_train_m, y_train_m)
lp_feature_importance_df = get_labelpowerset_feature_importance(
    lp_model, feature_names=reduced_df.columns, top_n=20
)
print("\n LabelPowerset Feature Importances")
print(lp_feature_importance_df.head(20))

 
grid_search = run_labelpowerset_gridsearch(X_train_m, y_train_m)

print("\nLabelPowerset GridSearch Best Params")
print(grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

y_pred_gs = grid_search.best_estimator_.predict(X_test_m)
print("Final Test Subset Accuracy:", round(accuracy_score(y_test_m, y_pred_gs), 4))
print("Final Test Hamming Loss:", round(hamming_loss(y_test_m, y_pred_gs), 4))