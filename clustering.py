import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def clustering(X, y):
    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def kmeans_clustering(n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_scaled)

        data['cluster'] = kmeans.labels_

        # 주성분 분석 (PCA)로 2차원으로 변환하여 시각화
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        pca_df['cluster'] = data['cluster']

        return pca_df

    # 클러스터 수를 변경하면서 각 클러스터링 결과를 서브플롯에 넣어 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, ax in enumerate(axes.flat, start=2):
        pca_df = kmeans_clustering(i)
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='viridis', s=100, alpha=0.7, ax=ax)
        ax.set_title(f'K-Means Clustering with {i} Clusters')
        ax.set_xlabel('Principal Component 1')  # 변경가능
        ax.set_ylabel('Principal Component 2')  # 변경가능
        ax.legend(title='Cluster')

    plt.tight_layout()
    plt.show()

    # training data, test data 분리 (size = 0.7)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # 선형 회귀 모델 초기화 및 학습
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # 테스트 데이터셋에 대해 예측 수행
    y_pred = regressor.predict(X_test)

    # 모델 성능 평가
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')


# 데이터셋 불러오기
data = pd.read_csv('cleaned_speed_data.csv')

# 직업 데이터 전처리 이어하기 (true == 1, false ==0 )
for col in data.columns:
    if data[col].dtype == 'bool':
        data[col] = data[col].astype(int)

# X(특성), y(타겟)
# X값 고려대상 : age, income, career (?)
# y값 고려대상 : prob, like, met (?)

X = data.drop(['prob'], axis=1)
y = data['prob']

clustering(X, y)
