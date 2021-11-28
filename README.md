# gtzan_db
 
This work aims to analyze the GTZAN database (https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) and build a model that is capable of classify a wave file into a category of 10 music genres.

A few machine learning and deep learning models are built and compared.<br>
GTZAN_Dataset.ipynb : database observation and models fine tuning <br>
train_ml_models : train ML models and MLP model<br>
train_cnn_models : train deep learning models<br>

## K nearest neighbors, Random Forest and Multi Layer Perceptron
<img src="img/knn_cf_matrix.png" width="30%" height="30%" />
<img src="img/rf_cf_matrix.png" width="30%" height="30%" />
<img src="img/mlp_cf_matrix.png" width="30%" height="30%" />
<br>
<img src="img/kfold_accuracy_comp.png" width="30%" height="30%" />
<br>
<br>

## Deep learning : CNN
<img src="img/learning_curve_dl.png" width="50%" height="50%" />
<br>
<img src="img/confusion_matrix_dl.png" width="30%" height="30%" />
