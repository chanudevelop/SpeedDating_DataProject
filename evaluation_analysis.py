import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

import Classification_KNN


def trained_clustering_model(data, n_clusters):
    # train data with k means and return model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans


def tune_clustering_hyperparameter(data):
    # candidates' range of n_clusters
    range_n_clusters = range(2, 30)
    silhouette_avg = pd.Series()

    # use silhouette score
    for n_clusters in range_n_clusters:
        # initialize k means
        kmeans = trained_clustering_model(data, n_clusters)
        # silhouette scores
        silhouette_avg.loc[n_clusters] = silhouette_score(data, kmeans.labels_)

    # when silhouette score is highest, k is optimal
    return silhouette_avg.idxmax()


def visualize_clustering(model, data, n_clusters, silhouette_avg, label):
    # analyze using plotting

    # visualize silhouette plot and scatter plot as subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # calculate cluster results
    cluster_labels = model.predict(data)

    # plot
    set_silhouette_subplot(ax1, model, data, n_clusters, cluster_labels, silhouette_avg)
    set_scatter_subplot(ax2, model, data, n_clusters, cluster_labels)
    plt.suptitle(
        f"For {label} data\nSilhouette analysis for KMeans clustering\nwith n_clusters = %d" % n_clusters,
        fontsize=14,
        fontweight='bold'
    )
    plt.show()


def set_silhouette_subplot(subplot, model, data, n_clusters, cluster_labels, silhouette_avg):
    # this subplot is the silhouette plot
    subplot.set_xlim([-0.2, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette plot
    subplot.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, model.labels_)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        subplot.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )

        # Label the silhouette plots with their cluster numbers at the middle
        subplot.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10

        subplot.set_title(f"Number of Cluster: {n_clusters}\nSilhouette Score: {round(silhouette_avg, 3)}")
        subplot.set_xlabel("The silhouette coefficient values")
        subplot.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        subplot.axvline(x=silhouette_avg, color="red", linestyle="--")

        subplot.set_yticks([])  # Clear the yaxis labels / ticks
        subplot.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])


def set_scatter_subplot(subplot, model, train_data, n_clusters, cluster_labels):
    # convert multi-dimension data to 2 dimension for visualizing to scatter plot
    pca = PCA(n_components=2)
    pca_data = pd.DataFrame(data=pca.fit_transform(train_data), columns=['PC1', 'PC2'])
    pca_data['cluster'] = cluster_labels

    # Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    subplot.scatter(data=pca_data, x='PC1', y='PC2', marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

    subplot.set_title("The visualization of the clustered data.")
    subplot.set_xlabel("Principal Component 1")
    subplot.set_ylabel("Principal Component 2")


def evaluate_analyze_clustering(data, label='all'):
    # Z-score scaling
    data_scaled = StandardScaler().fit_transform(data)

    # calculate optimal k (n_clusters)
    k = tune_clustering_hyperparameter(data_scaled)

    # train model
    clustering_model = trained_clustering_model(data_scaled, k)

    # evaluate model using silhouette score
    # worst: -1.0, best: 1.0
    # show this value in plot graph subtitle
    score = silhouette_score(data_scaled, clustering_model.labels_)

    # analyze model using plotting
    visualize_clustering(clustering_model, data_scaled, k, score, label)


def k_fold_cv_knn(X, y, n_neighbors):
    # evaluate using 10-fold cross validation
    kfold = KFold(10, shuffle=True, random_state=42)
    accuracy = []

    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Z-score scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        # train knn classifier model
        knn = Classification_KNN.KNNClassifier()
        knn.fit(pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train)
        y_pred = [knn.predict(pd.DataFrame([X_test_scaled[i]], columns=X_test.columns), n_neighbors) for i in
                  range(X_test.shape[0])]

        # calculate accuracy
        accuracy.append(accuracy_score(y_test, y_pred))

    mean_accuracy = sum(accuracy) / len(accuracy)
    return mean_accuracy


def confusion_matrix_knn(y_test, y_pred):
    # analyze using confusion matrix
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return matrix, report


def visualize_roc_curve_knn(y_test, y_pred, label):
    # analyze using ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(15, 5))

    # diagonal line
    plt.plot([0, 1], [0, 1], label='STR')
    # ROC curve
    plt.plot(fpr, tpr, label='ROC')

    plt.title(f"ROC curve of {label} data")
    plt.xlabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.show()


def evaluate_analyze_knn(data, label='all'):
    # dec is target variable
    X = data.drop(['dec'], axis=1)
    y = data['dec']

    # optimal k from k-fold cv in previous code
    n_neighbors = 5

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Z-score scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # train and predict with knn classifier model
    knn = Classification_KNN.KNNClassifier()
    knn.fit(pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train)
    y_pred = [knn.predict(pd.DataFrame([X_test_scaled[i]], columns=X_test.columns), n_neighbors) for i in
              range(X_test.shape[0])]

    # evaluate model using 10-fold cross validation
    accuracy = k_fold_cv_knn(X, y, n_neighbors)
    print(f"\n\nAverage accuracy of {label} data using 10-Fold cross validation: {accuracy}")

    # analyze model using confusion matrix and ROC curve
    matrix, report = confusion_matrix_knn(y_test, y_pred)
    print(f"Confusion Matrix of {label} data:\n", matrix)
    print(f"Classification report of {label} data:\n", report)

    # analyze by plotting ROC curve
    visualize_roc_curve_knn(y_test, y_pred, label)

    # analyze using AUC (the area under the ROC curve)
    # best case: 1.0
    # worst case: 0.5
    auc = roc_auc_score(y_test, y_pred)
    print(f"AUC score of {label} data: {auc}")


def evaluate_and_analyze(data):
    for col in data.columns:
        if data[col].dtype == 'bool':
            data[col] = data[col].astype(int)

    # analyze all data
    # clustering
    evaluate_analyze_clustering(data)

    # k nearest neighbors
    evaluate_analyze_knn(data)


def compare_test_data_with_cluster_data(train_data, test_data, compare_data):
    # compare test data with same gender, date successful data in same cluster

    for col in train_data.columns:
        if train_data[col].dtype == 'bool':
            train_data[col] = train_data[col].astype(int)

    for col in compare_data.columns:
        if compare_data[col].dtype == 'bool':
            compare_data[col] = compare_data[col].astype(int)

    # train clustering
    # n_clusters value 15 is optimal k getting from previous code
    model = trained_clustering_model(train_data, 15)
    y_pred = model.predict(test_data)[0]

    clusters = pd.DataFrame()
    clusters['data_index'] = train_data.index.values
    clusters['cluster'] = model.labels_

    # get same cluster data with test data
    pred_cluster_idx = clusters[clusters['cluster'] == y_pred]

    # get speed date successful data in same cluster data
    pred_cluster = compare_data.iloc[pred_cluster_idx['data_index']]
    matched_pred_cluster = pred_cluster[pred_cluster['dec'] == 1]

    # filter with same gender
    same_gender_matched_pred_cluster = matched_pred_cluster.loc[
        pred_cluster['gender'] == compare_data['gender'][0]]

    print(same_gender_matched_pred_cluster)

    # get mean of data
    matched_pred_cluster_mean = same_gender_matched_pred_cluster.mean()

    # compare only with important 7 columns about attractive
    important_columns = ["attr", "sinc", "intel", "fun", "amb", "shar", "like"]

    # get difference between successful data and test data
    difference = matched_pred_cluster_mean[important_columns] - test_data[important_columns]
    # Description of the top three columns or fewer where the difference fall
    inferior_column_data = (difference[difference > 0]
                            .sort_values(by=difference.index[0], ascending=False, axis=1)
                            .iloc[:, 0:3]
                            .dropna(axis=1))

    superior_column_data = (difference[difference < 0]
                            .sort_values(by=difference.index[0], ascending=True, axis=1)
                            .iloc[:, 0:3]
                            .dropna(axis=1))

    column_full_names_dict = {"attr": "attractiveness", "sinc": "sincerity", "intel": "intelligence", "fun": "fun",
                              "amb": "ambitiousness", "shar": "shared interests", "like": "overall rating"}

    # print difference as commercial statement
    print("This is your strength")
    for idx, column in enumerate(superior_column_data):
        full_name = column_full_names_dict[column]
        value = superior_column_data[column][0]
        print(f"Your {full_name} is {test_data[column][0]} points, which is %.2f greater than the successful data." % -value)
        if idx == 0:
            print(f"Your biggest strength is {full_name}.")
        elif idx == 1:
            print(f"You have a talent for {full_name}!")
        else:
            print(f"You're a bit of {full_name}.")
        print()

    print("\nand this is your weakness")
    for idx, column in enumerate(inferior_column_data):
        full_name = column_full_names_dict[column]
        value = inferior_column_data[column][0]
        print(f"Your {full_name} is {test_data[column][0]} points, which is %.2f lower than the successful data." % value)
        if idx == 0:
            print(f"What you lack the most is {full_name}, do your best!")
        elif idx == 1:
            print(f"You should try to increase your {full_name}!")
        else:
            print(f"It's better to pay a little attention to your {full_name}.")
        print()


# test data for comparing with successful data
test_data = pd.DataFrame({
    'gender': [0],
    'age': [22],
    'income': [70000],
    'dec': [0],
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
