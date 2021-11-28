import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn import model_selection

def train_knn_model(X_train_scale, y_train, X_test_scale, y_test):
    # Train
    knn = KNeighborsClassifier(n_neighbors = 5, leaf_size=1, p =1) #p=1 manhattan distance
    knn.fit(X_train_scale, y_train.values.ravel())

    # Predict
    y_pred = knn.predict(X_test_scale)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(knn, X_test_scale, y_test, normalize='true', cmap=plt.cm.Blues, ax=ax)
    plt.title('KNN model Test DB confusion matrix')
    plt.savefig('./img/knn_cf_matrix.png')

    report = classification_report(y_test, y_pred)
    print("Classification Report:",)
    print(report)

    # Ratio of correctly predicted observation to the total observations
    acc_score = accuracy_score(y_test,y_pred)
    print("Accuracy:",acc_score)

    return knn

def train_rf_model(X_train, y_train, X_text, y_test):
    rf = RandomForestClassifier(n_estimators = 1800, random_state = 42, max_depth=None)

    # Train on non scale data
    rf.fit(X_train, y_train.values.ravel())

    # Predict
    y_pred = rf.predict(X_test)

    # Evaluate
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(rf, X_test, y_test, normalize='true', cmap=plt.cm.Blues, ax=ax)
    plt.title('RF model Test DB confusion matrix')
    plt.savefig('./img/rf_cf_matrix.png')

    report = classification_report(y_test, y_pred)
    print("Classification Report:",)
    print(report)

    # Ratio of correctly predicted observation to the total observations
    acc_score = accuracy_score(y_test,y_pred)
    print("Accuracy:",acc_score)

    return rf

def train_mlp_model(X_train_scale, y_train, X_test_scale, y_test):
    # loss = log loss
    mlp = MLPClassifier( hidden_layer_sizes=(500,500,300), max_iter=500,
                        solver='sgd', verbose=False,  random_state=21)

    # Train
    mlp.fit(X_train_scale, y_train.values.ravel())

    # Predict
    y_pred = mlp.predict(X_test_scale)

    plt.plot(mlp.loss_curve_)
    plt.title('leaning curve : loss')
    plt.savefig('./img/mlp_lc.png')

    # Evaluate
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(mlp, X_test_scale, y_test, normalize='true', cmap=plt.cm.Blues, ax=ax)
    plt.title('MLP model Test DB confusion matrix')
    plt.savefig('./img/mlp_cf_matrix.png')

    report = classification_report(y_test, y_pred)
    print("Classification Report:",)
    print(report)

    # Ratio of correctly predicted observation to the total observations
    acc_score = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc_score)

    return mlp

if __name__ == "__main__":

    # Load and prepare data
    data_df = pd.read_csv('./data/features_3_sec.csv')

    # Remove highly correlated features
    df = data_df.drop(['filename', 'length', 'mfcc2_mean', 'rolloff_mean'], axis=1)

    # Train/test split
    X = df.drop(['label'], axis =1)
    y = df[['label']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, shuffle=True, stratify=y)

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaler.fit(X_train) # Fit to train DB only because test DB has to be unknown for the model
    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)

    knn = train_knn_model(X_train_scale, y_train, X_test_scale, y_test)
    rf = train_rf_model(X_train, y_train, X_test, y_test)
    mlp = train_mlp_model(X_train_scale, y_train, X_test_scale, y_test)

    # Model selection
    df_shuffle = df.sample(frac = 1)
    X = df_shuffle.drop(['label'], axis =1)
    y = df_shuffle[['label']]

    scaler = StandardScaler()
    scaler.fit(X)
    X_scale = scaler.transform(X)

    results = []
    names = []

    seed = 42
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(knn, X_scale, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append('KNN')

    cv_results = model_selection.cross_val_score(rf, X, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append('RF')

    cv_results = model_selection.cross_val_score(mlp, X_scale, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append('MLP')

    fig, ax = plt.subplots()
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.title('Accuracy on 10-Fold')
    plt.savefig('./img/kfold_accuracy_comp.png')
    plt.show()