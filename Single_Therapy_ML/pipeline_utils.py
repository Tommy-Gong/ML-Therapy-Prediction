from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt



def encode_target(df_encoded, target_column):
    y = df_encoded[target_column]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

def prepare_features(df_encoded, reduced_df, target_column=None):
    X = df_encoded[reduced_df.columns]
    if target_column and target_column in X.columns:
        X = X.drop(columns=[target_column])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

def train_model(X_train, y_train, model_type='rf'):
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'bagging':
        model = BaggingClassifier(n_estimators=100, random_state=42)
    elif model_type == 'extratree':
        model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model_type")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc

def plot_feature_importance(model, feature_names, top_n=5):
    if not hasattr(model, "feature_importances_"):
        print("Feature importance not available.")
        return pd.DataFrame()
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
    plt.title(f"Top {top_n} Feature Importances")
    plt.show()
    return importance_df

def run_pipeline(df_encoded, reduced_df, target_column, model_type='rf', test_size=0.2, top_n_features=5):
    y_encoded, label_encoder = encode_target(df_encoded, target_column)
    X = prepare_features(df_encoded, reduced_df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
    model = train_model(X_train, y_train, model_type=model_type)
    y_pred, acc = evaluate_model(model, X_test, y_test)
    print(f"{model_type.upper()} Accuracy: {acc:.4f}")
    importance_df = plot_feature_importance(model, X.columns, top_n=top_n_features)
    comparison_df = pd.DataFrame({
        'Actual': label_encoder.inverse_transform(y_test),
        'Predicted': label_encoder.inverse_transform(y_pred)
    })
    print("\nPrediction Sample")
    print(comparison_df.head(10))
    return {
        "accuracy": acc,
        "importance_df": importance_df,
        "comparison_df": comparison_df
    }