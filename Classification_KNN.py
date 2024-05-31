import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class KNNClassifier:
    def __init__(self):
        self.points = pd.DataFrame()
        self.labels = pd.DataFrame()

    def fit(self, points: pd.DataFrame, labels: pd.DataFrame):
        self.points = points.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)

    def predict(self, target: pd.DataFrame, k: int): #K-NN algorithm
        target_norm = (target - self.points.mean()) / self.points.std() #target point normalizaion
        points_norm = (self.points - self.points.mean()) / self.points.std() #dataset normalization
        distances = np.sqrt(((points_norm - target_norm.iloc[0]) ** 2).sum(axis=1)) #calculate Euclid distance
        nearest_neighbors = distances.nsmallest(k).index #finding k-nearest neighbors
        return self.labels.loc[nearest_neighbors].mode()[0] #return the most common label among the nearest neighbors

def load_and_prepare_data(data): #load dataset
    for col in data.columns:
        if data[col].dtype == 'bool':
            data[col] = data[col].astype(int)
    return data

def split_and_scale_data(data): #split / scale dataset
    X = data.drop('dec', axis=1) #define features(X) and target(y)
    y = data['dec']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #split the dataset into train / test sets (70% training, 30% testing)
    scaler = StandardScaler() #feature scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def train_knn(X_train_scaled, X_train, y_train):#initialize and train the KNN classifier
    knn_classifier = KNNClassifier()
    knn_classifier.fit(pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train)
    return knn_classifier

def evaluate_knn(knn_classifier, X_test_scaled, X_test, y_test, k):#evaluation
    y_pred = [knn_classifier.predict(pd.DataFrame([X_test_scaled[i]], columns=X_test.columns), k) for i in range(X_test.shape[0])]
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def predict_new_data(knn_classifier, new_data_point, scaler, k):#predict dec for new data point
    new_data_point_scaled = scaler.transform(new_data_point)
    predicted_dec = knn_classifier.predict(pd.DataFrame(new_data_point_scaled, columns=new_data_point.columns), k)
    return predicted_dec
