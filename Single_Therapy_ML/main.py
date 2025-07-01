import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

from lazypredict.Supervised import LazyClassifier
from preprocess import preprocess_data  

from itertools import combinations
import re

from pipeline_utils import run_pipeline 
from correlation_visualizer import top_pairs



file_path = "/Users/kegong/Desktop/Work/studiendaten_f__r_similarity_analyse_neu.xlsx"
df_filled, df_encoded, reduced_df, cramer_matrix, pearson_corr_matrix = preprocess_data(file_path)
reduced_df = df_encoded.select_dtypes(include=[np.number])

print("full data shape:", df_filled.shape)
print("encode data shape:", df_encoded.shape)
print("high correlation column removed shape:", reduced_df.shape)


target_column = 'Wirkstoffschema adjuvante Therapie'
# X_lazy = df_encoded[reduced_df.columns]
valid_cols = reduced_df.columns.intersection(df_encoded.columns)
X_lazy = df_encoded[valid_cols]

y_lazy = df_encoded[target_column]

X_train, X_test, y_train, y_test = train_test_split(X_lazy, y_lazy, test_size=0.4, random_state=42)
clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print("\nLazyClassifier Results")
print(models)



results = run_pipeline(
    df_encoded=df_encoded,
    reduced_df=reduced_df,
    target_column='Wirkstoffschema neoadjuvante Therapie',
    model_type='extratree',
    test_size=0.2,
    top_n_features=20
)

corr_matrix, top_pairs = top_pairs(reduced_df, top_n=5)