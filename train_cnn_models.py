import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_curves(log, epochs_number):
    acc = log.history['accuracy']
    val_acc = log.history['val_accuracy']

    loss = log.history['loss']
    val_loss = log.history['val_loss']

    epochs_range = range(epochs_number)

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./img/learning_curve_dl.png')
    plt.show()

def plot_dl_confusion_matrix(ds, model):
    predictions = np.array([])
    labels = np.array([])
    for x, y in ds.unbatch():
        predictions = np.concatenate([predictions, model.predict_classes(tf.expand_dims(x, axis=0))])
        labels = np.concatenate([labels, [y.numpy()]])

    cf_m = tf.math.confusion_matrix(labels = labels, predictions = predictions).numpy()
    cf_mm = cf_m.astype('float') / cf_m.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cf_mm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    ax.set_title('Confusion Matrix with labels for Validation DB\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    plt.savefig('./img/confusion_matrix_dl.png')
    plt.show()


# Load images
data_path = './data/images_original'
data_dir = pathlib.Path(data_path)
img_height = 288
img_width = 432
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                validation_split=0.2,
                                                                subset='training',
                                                                seed=1233,
                                                                image_size=(img_height, img_width),
                                                                batch_size=batch_size)
                                                                
valid_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                validation_split=0.2,
                                                                subset='validation',
                                                                seed=1233,
                                                                image_size=(img_height, img_width),
                                                                batch_size=batch_size)

class_names = train_ds.class_names
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Build CNN model
num_classes = len(class_names)
epochs_number = 10

model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(128, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  # tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), # from_logits=True then need softmax before predicting 
  metrics=['accuracy'])

log = model.fit(train_ds, validation_data=valid_ds, epochs=epochs_number)
plot_training_curves(log, epochs_number)
plot_dl_confusion_matrix(train_ds, model)
plot_dl_confusion_matrix(valid_ds, model)