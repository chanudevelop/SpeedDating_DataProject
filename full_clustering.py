import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns



def fullClustering(data) :


    # data preprocessing for carrier (true == 1, false == 0 )
    for col in data.columns:
        if data[col].dtype == 'bool':
            data[col] = data[col].astype(int)

    # data normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    def kmeans_clustering(n_clusters): 
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_scaled)
        
        # data['cluster'] = kmeans.labels_

        # Two-dimensional conversion and visualization with principal component analysis (PCA)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        pca_df['cluster'] = kmeans.labels_
        
        return pca_df


    # data visualization
    fig, axes = plt.subplots(4, 5, figsize=(18,12)) # clustering range : 1-20 


    for i, ax in enumerate(axes.flat, start=1):

        pca_df = kmeans_clustering(i)
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='viridis', s=10, alpha=0.7, ax=ax) 
        ax.set_title(f'K-Means Clustering with {i} Clusters')
        ax.set_xlabel('Principal Component 1') 
        ax.set_ylabel('Principal Component 2') 
        ax.legend(title='Cluster')

    plt.tight_layout()
    plt.show()
