# mlp_and_gan_training.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.backend import clear_session
from tensorflow.random import set_seed

# Parameters
num_of_classes = 4
data_shape = (7,)
times_to_run = 50
mlp_epochs = 40
latent_dim = 100
gan_epochs = 5000
valid_split = 0.20
selection_seed = 150
seed_multiplier = 1000000

# Data Preparation
dataset = pd.read_csv("./data/dataset.csv")
labels = dataset.Class.values - 1
features = dataset.drop(columns="Class").values
X_train, X_test, Y_train, Y_test = train_test_split(
    features, labels, test_size=0.5, random_state=selection_seed, stratify=labels
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLP Training
history, all_test_loss, all_test_acc = [], [], []
for i in range(times_to_run):
    set_seed(i * seed_multiplier)
    model = Sequential([
        Dense(128, input_dim=7, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(num_of_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(0.0002, 0.5), loss="categorical_crossentropy", metrics=["accuracy"])
    history_temp = model.fit(X_train_scaled, to_categorical(Y_train), epochs=mlp_epochs, batch_size=64, validation_split=valid_split, verbose=0)
    history.append(history_temp)
    test_loss, test_acc = model.evaluate(X_test_scaled, to_categorical(Y_test), verbose=0)
    all_test_loss.append(test_loss)
    all_test_acc.append(test_acc)
    clear_session()

# MLP Results
train_acc, val_acc = [], []
for h in history:
    train_acc.append(h.history["accuracy"])
    val_acc.append(h.history["val_accuracy"])

plt.plot(np.mean(train_acc, axis=0), label="Training Accuracy")
plt.plot(np.mean(val_acc, axis=0), label="Validation Accuracy")
plt.title("MLP Training and Validation Accuracy")
plt.legend()
plt.savefig("./results/MLP_Accuracy.png")
