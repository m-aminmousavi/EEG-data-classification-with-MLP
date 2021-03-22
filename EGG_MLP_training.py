# Necessary Packages
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reading and splitting the dataset into features and label
data = pd.read_csv('completetrainingfull.csv')
x = data.iloc[:, 1:].to_numpy()
y = data.iloc[:, :1].to_numpy()

y -= y.min()
y = tf.keras.utils.to_categorical(y)

# Splitting the dataset into train and test parts
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# x_train = x[:900]
# y_train = y[:900]
# x_test = x[900:]
# y_test = y[900:]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


### Scaling the features
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

### PCA
# pca = PCA(n_components=80)
# pca.fit(x_train)
# print(pca.explained_variance_[0] / pca.explained_variance_)

# x_train = pca.transform(x_train)
# x_test = pca.transform(x_test)


### Network
print("[INFO] Creating a model...")
model = tf.keras.models.Sequential()

model.add(L.Dense(64, input_shape=x_train.shape[1:]))
model.add(L.BatchNormalization())
model.add(L.LeakyReLU())
model.add(L.Dropout(0.3))

model.add(L.Dense(64))
model.add(L.BatchNormalization())
model.add(L.LeakyReLU())
model.add(L.Dropout(0.3))

model.add(L.Dense(y_test.shape[1], activation='softmax'))

print("[INFO] Model Architecture:\n\n")
print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

### Train
print("[INFO] Training the model...")
history = model.fit(
    x_train, y_train,
    validation_data = (x_test, y_test),
    batch_size=16,
    epochs=200,
)

# Saving the model
print("[INFO] Saving the `.h5` format of the trained model...")
model.save('final_model.h5')

# Training and Testing process visualization
plt.figure(figsize=(7, 5), dpi=150)
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(loc='best')
plt.show()


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='best')
plt.show()
