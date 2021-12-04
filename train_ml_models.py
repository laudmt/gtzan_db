import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn import model_selection

def perform_kfold_cv(data_df):
    X = data_df.drop(['label', 'dataset_name', 'filename', 'length', 'mfcc2_mean', 'rolloff_mean'], axis=1)
    y = data_df[['label']]
    groups_by_dataset_name_list = data_df['dataset_name'].tolist()

    scaler = StandardScaler()
    scaler.fit(X) # Fit to train DB only because test DB has to be unknown for the model
    X_scale = scaler.transform(X)

    n_splits = 10
    gkf = GroupKFold(n_splits = n_splits)
    gkf.split(X, y.values.ravel(), groups = groups_by_dataset_name_list)
        
    results = []
    names = []

    cv_results = model_selection.cross_val_score(knn, X_scale, y.values.ravel(), cv=gkf, groups= groups_by_dataset_name_list, scoring='accuracy')
    results.append(cv_results)
    names.append('KNN')

    cv_results = model_selection.cross_val_score(rf, X, y.values.ravel(), cv=gkf, groups= groups_by_dataset_name_list, scoring='accuracy')
    results.append(cv_results)
    names.append('RF')

    cv_results = model_selection.cross_val_score(mlp, X_scale, y.values.ravel(), cv=gkf, groups= groups_by_dataset_name_list, scoring='accuracy')
    results.append(cv_results)
    names.append('MLP')

    fig, ax = plt.subplots()
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.title('Accuracy on 10-Fold')
    plt.savefig('./img/kfold_accuracy_comp.png')
    plt.clf()

def knn_fine_tuning(X_train, y_train):
    #List Hyperparameters that we want to tune.
    leaf_size = list(range(1,5))
    n_neighbors = list(range(5,15))
    p=[1,2]
    #Convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    #Create new KNN object
    knn = KNeighborsClassifier()
    #Use GridSearch
    clf = GridSearchCV(knn, hyperparameters, cv=5)
    #Fit the model
    best_model = clf.fit(X_train, y_train.values.ravel())
    #Print The value of best Hyperparameters
    print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors']) 


def train_knn_model(X_train_scale, y_train, X_test_scale, y_test):
    # Train
    knn = KNeighborsClassifier(n_neighbors = 14, leaf_size=2, p =1) #p=1 manhattan distance
    knn.fit(X_train_scale, y_train.values.ravel())

    # Predict
    y_pred = knn.predict(X_test_scale)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(knn, X_test_scale, y_test, normalize='true', cmap=plt.cm.Blues, ax=ax)
    plt.title('KNN model Test DB confusion matrix')
    plt.savefig('./img/knn_cf_matrix.png')
    plt.clf()
    plt.cla()

    report = classification_report(y_test, y_pred)
    print("Classification Report:",)
    print(report)

    # Ratio of correctly predicted observation to the total observations
    acc_score = accuracy_score(y_test,y_pred)
    print("Accuracy:",acc_score)

    return knn

def rf_fine_tuning(X_train, y_train):
    #List Hyperparameters that we want to tune.
    n_estimators = list(range(200, 2000, 100))
    max_depth = list([2,4,8,16,32])

    #Convert to dictionary
    hyperparameters = dict(max_depth=max_depth, n_estimators=n_estimators)
    #Create new RF object
    rf_2 = RandomForestClassifier()
    #Use GridSearch
    clf = GridSearchCV(rf_2, hyperparameters, cv=5)
    #Fit the model
    best_model = clf.fit(X_train, y_train.values.ravel())
    #Print The value of best Hyperparameters
    print('Best n_estimators:', best_model.best_estimator_.get_params()['n_estimators'])
    print('Best max_depth:', best_model.best_estimator_.get_params()['max_depth'])


def train_rf_model(X_train, y_train, X_text, y_test):
    rf = RandomForestClassifier(n_estimators = 1400, random_state = 42, max_depth=32)

    # Train on non scale data
    rf.fit(X_train, y_train.values.ravel())

    # Predict
    y_pred = rf.predict(X_test)

    # Evaluate
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(rf, X_test, y_test, normalize='true', cmap=plt.cm.Blues, ax=ax)
    plt.title('RF model Test DB confusion matrix')
    plt.savefig('./img/rf_cf_matrix.png')
    plt.clf()

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
    plt.clf()

    # Evaluate
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(mlp, X_test_scale, y_test, normalize='true', cmap=plt.cm.Blues, ax=ax)
    plt.title('MLP model Test DB confusion matrix')
    plt.savefig('./img/mlp_cf_matrix.png')
    plt.cla()
    plt.clf()

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
    data_df['dataset_name'] = data_df.filename.str.extract(r'(\w+\.\d+)\.[0-9]\.wav')

    # sample 30 dataset for our test db
    # Make sure that in our split, we don't have chunks of the same song in train and test db
    test_db_dataset = []
    for name, group in data_df.groupby('label'):
        test_db_dataset += group.drop_duplicates(subset=['dataset_name']).sample(3).dataset_name.tolist()

    # Train/test split
    test_df = data_df[data_df.dataset_name.isin(test_db_dataset)]
    train_df = data_df[~data_df.dataset_name.isin(test_db_dataset)]

    # Shuffle
    test_df = test_df.sample(n=len(test_df), random_state=42)
    train_df = train_df.sample(n=len(train_df), random_state=42)

    # Remove highly correlated features
    X_train = train_df.drop(['label', 'dataset_name', 'filename', 'length', 'mfcc2_mean', 'rolloff_mean'], axis=1)
    y_train = train_df[['label']]

    X_test = test_df.drop(['label', 'dataset_name', 'filename', 'length', 'mfcc2_mean', 'rolloff_mean'], axis=1)
    y_test = test_df[['label']]

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaler.fit(X_train) # Fit to train DB only because test DB has to be unknown for the model
    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)

    # Train models
    knn = train_knn_model(X_train_scale, y_train, X_test_scale, y_test)
    rf = train_rf_model(X_train, y_train, X_test, y_test)
    mlp = train_mlp_model(X_train_scale, y_train, X_test_scale, y_test)

    # Compare models with kfold cross validation
    perform_kfold_cv(data_df)