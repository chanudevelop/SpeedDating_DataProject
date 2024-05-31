import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold

class KNNClassifier:
    def __init__(self):
        self.points = pd.DataFrame()
        self.labels = pd.DataFrame()

    def fit(self, points: pd.DataFrame, labels: pd.DataFrame):
        self.points = points.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)

    def predict(self, target: pd.DataFrame, k: int):
        # 타겟 포인트 정규화
        target_norm = (target - self.points.mean()) / self.points.std()
        
        # 데이터셋 정규화
        points_norm = (self.points - self.points.mean()) / self.points.std()
        
        # 유클리드 거리 계산
        distances = np.sqrt(((points_norm - target_norm.iloc[0]) ** 2).sum(axis=1))
        
        # k개의 가장 가까운 이웃 찾기
        nearest_neighbors = distances.nsmallest(k).index
        
        # 가장 가까운 이웃 중 가장 흔한 레이블 반환
        return self.labels.loc[nearest_neighbors].mode()[0]
    

def find_best_k(X, y, k_values, n_splits=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    avg_accuracies = []

    for k in k_values:
        accuracies = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_val = X_scaled[train_index], X_scaled[test_index]
            y_train, y_val = y[train_index], y[test_index]
            
            knn_classifier = KNNClassifier()
            knn_classifier.fit(pd.DataFrame(X_train, columns=X.columns), y_train)
            y_pred = [knn_classifier.predict(pd.DataFrame([X_val[i]], columns=X.columns), k) for i in range(X_val.shape[0])]
            accuracy = accuracy_score(y_val, y_pred)
            accuracies.append(accuracy)
        
        avg_accuracies.append(np.mean(accuracies))
    
    best_k = k_values[np.argmax(avg_accuracies)]
    return best_k, avg_accuracies

# 데이터셋 불러오기
data = pd.read_csv('cleaned_speed_data.csv')

# True/False 값을 1/0으로 변환
for col in data.columns:
    if data[col].dtype == 'bool':
        data[col] = data[col].astype(int)

# 특성(X)과 타겟(y) 정의
X = data.drop('dec', axis=1)
y = data['dec']

# k 값 후보들 정의
k_values = range(1, 10)

# 최적의 k 값 찾기
best_k, accuracies = find_best_k(X, y, k_values)

print(f'Best k: {best_k}')
print(f'Accuracies for each k: {accuracies}')

# 데이터셋을 학습과 테스트로 분리 (70% 학습, 30% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 특성 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN 분류기 초기화 및 학습
knn_classifier = KNNClassifier()
knn_classifier.fit(pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train)

# 테스트 데이터셋에 대해 예측 수행
#k = 5
#y_pred = [knn_classifier.predict(pd.DataFrame([X_test_scaled[i]], columns=X_test.columns), k) for i in range(X_test.shape[0])]

#최적의 k 값으로 최종 모델 학습 및 평가
y_pred = [knn_classifier.predict(pd.DataFrame([X_test_scaled[i]], columns=X_test.columns), best_k) for i in range(X_test.shape[0])]

# 정확도 출력 및 분류 보고서 출력
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# 새로운 데이터 포인트 예측
new_data_point = pd.DataFrame({
    'gender': [0],
    'age': [22],
    'income': [70000],
    'attr': [6],
    'sinc': [8],
    'intel': [7],
    'fun': [6],
    'amb': [5],
    'shar': [5],
    'like': [7],
    'prob': [8],
    'met': [2],
    'career_Arts': [False],
    'career_Business': [False],
    'career_Consulting': [False],
    'career_Education': [False],
    'career_Engineering': [False],
    'career_Entertainment': [False],
    'career_Finance': [False],
    'career_Government': [False],
    'career_Healthcare': [False],
    'career_Legal': [True],
    'career_Other': [False],
    'career_Real Estate': [False],
    'career_Science': [False],
    'career_Social Work': [False],
    'career_Sports': [False],
    'career_Technology': [False]
})

# 새로운 데이터 포인트에 대해 k=5일 때 'dec' 값 예측
new_data_point_scaled = scaler.transform(new_data_point)
predicted_dec = knn_classifier.predict(pd.DataFrame(new_data_point_scaled, columns=new_data_point.columns), k)
print(f'Predicted dec for new data point: {predicted_dec}')