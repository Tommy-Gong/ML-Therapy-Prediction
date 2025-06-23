import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_data(file_path, cramer_threshold=0.75):
    col1 = 'Wirkstoffschema neoadjuvante Therapie'
    col2 = 'Wirkstoffschema adjuvante Therapie'
    lower = [s.strip().lower() for s in [
        'entfällt', 'fehlt', 'entfällt/fehlt', 'n.r.', 'enfällt', 'k.a.', 'k.a', 'keine angabe'
    ]]
    unwanted = ['fehlt']

    df = pd.read_excel(file_path, engine="openpyxl")

    def clean_except(df, lower, unwanted, col1, col2):
        def clean_cell_general(x):
            return x.strip().lower() if isinstance(x, str) else x

        def clean_cell_with_nulling(x):
            if isinstance(x, str):
                x_cleaned = x.strip().lower()
                return None if x_cleaned in lower or x_cleaned in unwanted else x_cleaned
            return x

        for col in df.columns:
            if col in [col1, col2]:
                df[col] = df[col].apply(clean_cell_general)
            else:
                df[col] = df[col].apply(clean_cell_with_nulling)
        return df

    df = clean_except(df, lower, unwanted, col1, col2)

    columns_to_drop = ['Eingangsbuchnummer', 'Progress', 'Anzahl Metastasen extrahepatisch']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    df.columns = df.columns.str.strip()
    df_filled = df.copy()

    numerical_median_fill = [
        'Alter des Patienten bei Diagnose', 'BMI', 'PFS', 'OS',
        'Tumor/Metastasen Durchmesser', 'Anzahl Metastasen intrahepatisch',
        'Anzahl Metastasen extrahepatisch'
    ]

    for col in numerical_median_fill:
        if col in df_filled.columns:
            df_filled[col] = pd.to_numeric(df_filled[col], errors='coerce')
            median_val = df_filled[col].median()
            df_filled[col] = df_filled[col].fillna(median_val)

    categorical_mode_fill = [
        'T-Status', 'N-Status', 'M-Status', 'V-Status', 'L-Status', 'Pn-Status',
        'Grading', 'R-Status', 'Tumorseite', 'Histologie', 'Vitalstatus',
        'Tumorbedingt verstorben', 'synchrone oder metachrone Metastasierung',
        'Tumormarker 1', 'Tumormarker 2', 'Therapieerfolg neoadjuvante Therapie',
        'adjuvante Chemotherapie', 'Vorerkrankungen', 'early Progress 1=nein; 2= ja',
        'neoadjuvante Chemotherapie'
    ]

    for col in categorical_mode_fill:
        if col in df_filled.columns:
            mode_val = df_filled[col].mode()[0]
            df_filled[col] = df_filled[col].fillna(mode_val)

    drop_col = [
        'Geburtsdatum', 'Progress Datum', 'OP Datum', 'Nachsorge Datum', 'Todesdatum',
        'Ort metastasen extrahepatisch',
        'Adjuvante Therapie des Primärtumors oder vorhergeganener Metastase',
        'Wirkstoffschema der Adjuvanten Therapie des Primärtumors oder vorhergegangener Metastase',
        'Pn-Status'
    ]
    df_filled = df_filled.drop(columns=[col for col in drop_col if col in df_filled.columns])

    df_encoded = df_filled.copy()
    exclude_cols = ['Todesdatum', 'Progress Datum', 'Nachsorge Datum', col1, col2]
    categorical_columns = [
        col for col in df_encoded.select_dtypes(include=['object', 'category']).columns if col not in exclude_cols
    ]

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = df_encoded[col].astype(str).str.strip()
        df_encoded[col] = df_encoded[col].replace('nan', np.nan)
        df_encoded[col] = df_encoded[col].fillna('Missing')
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    target_column = col1
    cat_cols_for_cramer = df_filled.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in cat_cols_for_cramer:
        cat_cols_for_cramer.remove(target_column)

    target_cols_to_keep = {col1, col2}
    to_drop = set()
    cramer_matrix = pd.DataFrame(index=cat_cols_for_cramer, columns=cat_cols_for_cramer)

    for col_a, col_b in combinations(cat_cols_for_cramer, 2):
        if col_a == col_b:
            cramer_matrix.loc[col_a, col_b] = 1.0
            continue
        try:
            confusion_matrix = pd.crosstab(df_filled[col_a], df_filled[col_b])
            if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
                continue
            chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            denom = min(k - 1, r - 1)
            if denom == 0:
                continue
            v = np.sqrt(phi2 / denom)
            cramer_matrix.loc[col_a, col_b] = v
            cramer_matrix.loc[col_b, col_a] = v
            if v > cramer_threshold and col_b not in target_cols_to_keep:
                to_drop.add(col_b)
        except Exception:
            continue

    cramer_matrix = cramer_matrix.astype(float)
    print(f"(Not Dropping!) Cramér's V > {cramer_threshold}: {sorted(to_drop)}")

    reduced_df = df_encoded.select_dtypes(include=[np.number])
    if target_column in reduced_df.columns:
        reduced_df = reduced_df.drop(columns=[target_column])

    corr_matrix = reduced_df.corr(method='pearson').abs()
    print("Pearson Correlation Matrix:\n", corr_matrix)

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap of Numerical Features (No Dropping)")
    plt.tight_layout()
    plt.show()

    return df_filled, df_encoded, reduced_df, cramer_matrix, corr_matrix, to_drop