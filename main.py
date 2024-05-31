import CleaningData
import Classification_KNN
import full_clustering
import evaluation_analysis
import DataInspection_Exploration
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('speed_data_data.csv')

    # print data inspection and show plot
    DataInspection_Exploration.DataExploration(df)

    # preprocess data
    not_scaled_preprocessed_data, preprocessed_data = CleaningData.PreprocessData(df)

    # test about knn classification

    # Split and scale the dataset
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = Classification_KNN.split_and_scale_data(
        preprocessed_data)

    # train KNN classifier
    knn_classifier = Classification_KNN.train_knn(X_train_scaled, X_train, y_train)

    # evaluation(accuracy, report)
    k = 5
    accuracy, report = Classification_KNN.evaluate_knn(knn_classifier, X_test_scaled, X_test, y_test, k)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)

    # predict dec for new data point
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

    predicted_dec = Classification_KNN.predict_new_data(knn_classifier, new_data_point, scaler, k)
    print(f'Predicted dec for new data point: {predicted_dec}')

    # test about k means clustering
    full_clustering.fullClustering(preprocessed_data)

    # evaluate and analyze data
    evaluation_analysis.evaluate_and_analyze(preprocessed_data)

    # compare test data with original data
    # print results as commercial statement
    evaluation_analysis.compare_test_data_with_cluster_data(preprocessed_data, evaluation_analysis.test_data, not_scaled_preprocessed_data)


