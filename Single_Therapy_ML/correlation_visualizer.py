import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def top_pairs(reduced_df, top_n=5):
   

    corr_matrix = reduced_df.corr(method='pearson').abs()


    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    corr_pairs = []
    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            corr_pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))


    corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)

    print(f"\nTop {top_n} Most Correlated Pairs")
    for pair in corr_pairs_sorted[:top_n]:
        print(f"{pair[0]} vs {pair[1]}: Pearson r = {pair[2]:.4f}")

    return corr_matrix, corr_pairs_sorted[:top_n]