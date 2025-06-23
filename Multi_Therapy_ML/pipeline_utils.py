import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import accuracy_score, hamming_loss, make_scorer
import seaborn as sns
import matplotlib.pyplot as plt

def run_multilabel_baseline(X, y):
    models = {
        "MultiOutput-RF": MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
        "ClassifierChain-RF": ClassifierChain(RandomForestClassifier(n_estimators=100, random_state=42)),
        "LabelPowerset-XGB": LabelPowerset(RandomForestClassifier(n_estimators=100, random_state=42))
    }

    results = []

    for name, model in models.items():
        try:
            model.fit(X['train'], y['train'])
            y_pred = model.predict(X['test'])

            if not isinstance(y_pred, np.ndarray):
                y_pred = y_pred.toarray()

            subset_acc = accuracy_score(y['test'], y_pred)
            hamming = hamming_loss(y['test'], y_pred)

            results.append({
                'Model': name,
                'Subset Accuracy': round(subset_acc, 4),
                'Hamming Loss': round(hamming, 4)
            })
        except Exception as e:
            print(f"Error in {name}: {e}")
            results.append({
                'Model': name,
                'Subset Accuracy': 'Error',
                'Hamming Loss': 'Error'
            })

    return pd.DataFrame(results)


def get_labelpowerset_feature_importance(model, feature_names, top_n=20):
    importances = model.classifier.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
    plt.title(f"Top {top_n} Feature Importances (LabelPowerset-RF)")
    plt.tight_layout()
    plt.show()

    return importance_df


def run_labelpowerset_gridsearch(X_train, y_train):
    def multilabel_score(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    scorer = make_scorer(multilabel_score, greater_is_better=True)

    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    lp_model = LabelPowerset(RandomForestClassifier(random_state=42))

    grid_search = GridSearchCV(estimator=lp_model,
                               param_grid=param_grid,
                               scoring=scorer,
                               cv=3,
                               verbose=2)

    grid_search.fit(X_train, y_train)
    return grid_search