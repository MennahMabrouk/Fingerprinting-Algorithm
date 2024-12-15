from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from keras.backend import clear_session
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy.random import seed
from tensorflow.random import set_seed as set_random_seed
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ipywidgets as widgets

# Parameters
num_of_classes = 4
data_shape = (7, 1)
times_to_run = 50
mlp_epochs = 40
valid_split = 0.20
latent_dim = 100
gan_epochs = 5000
selection_seed = 150
seed_multiplier = 1000000

# UI for experiment
cb1 = widgets.Checkbox(description="Generate missing data only")
slider1 = widgets.FloatSlider(value=0.1, min=0.05, max=1, step=0.05)
slider2 = widgets.IntSlider(value=250, min=0, max=1000, step=250)
vb = widgets.VBox(children=[slider2])


def checkbox(button):
    if button["new"]:
        vb.children = []
        slider2.value = 250 - int(slider1.value * 250)
    else:
        vb.children = [slider2]


cb1.observe(checkbox, names="value")

fraction_of_data = slider1.value
data_to_gen = slider2.value

# Data Preparation
dataset = pd.read_csv("./data/dataset.csv")
labels = dataset.Class.values - 1
features = dataset.drop(columns="Class").values

tr_fea, X_test, tr_label, Y_test = train_test_split(
    features, labels, test_size=0.5, random_state=selection_seed, stratify=labels
)

X_train, Z_train, Y_train = [], [], []
for idx in range(num_of_classes):
    number_filter = np.where(tr_label == idx)
    X_filtered, Y_filtered = tr_fea[number_filter], tr_label[number_filter]
    num_of_data = max(1, int(fraction_of_data * X_filtered.shape[0]))  # Ensure at least 1 sample
    RandIndex = np.random.choice(X_filtered.shape[0], num_of_data, replace=False)
    Z_train.append(X_filtered[RandIndex])
    X_train.extend(X_filtered[RandIndex])
    Y_train.extend(Y_filtered[RandIndex])

X_train, Y_train = np.asarray(X_train, dtype=np.float32), np.asarray(Y_train, dtype=np.float32)
X_train, Y_train = shuffle(X_train, Y_train)

Y_train_encoded, Y_test_encoded = to_categorical(Y_train), to_categorical(Y_test)
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

# Debugging Shapes
print(f"X_train shape: {X_train_transformed.shape}")
print(f"Y_train_encoded shape: {Y_train_encoded.shape}")
print(f"X_test_transformed shape: {X_test_transformed.shape}")
print(f"Y_test_encoded shape: {Y_test_encoded.shape}")

# Training MLP
all_test_loss, all_test_acc, history = [], [], []
for i in tqdm(range(times_to_run)):
    seed(i * seed_multiplier)
    set_random_seed(i * seed_multiplier)

    model = Sequential([
        Input(shape=(7,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(4, activation="softmax"),
    ])
    model.compile(optimizer=Adam(0.0002, 0.5), loss="categorical_crossentropy", metrics=["accuracy"])
    effective_batch_size = min(64, X_train_transformed.shape[0])  # Adjust batch size if needed
    history_temp = model.fit(
        X_train_transformed, Y_train_encoded, epochs=mlp_epochs,
        batch_size=effective_batch_size, validation_split=valid_split, verbose=0
    )
    history.append(history_temp)
    test_loss, test_acc = model.evaluate(X_test_transformed, Y_test_encoded, verbose=0)
    all_test_loss.append(test_loss)
    all_test_acc.append(test_acc)
    clear_session()

# MLP Results
trainacc, trainloss, valacc, valloss = [], [], [], []
for h in history:
    trainacc.append(h.history["accuracy"])
    trainloss.append(h.history["loss"])
    valacc.append(h.history["val_accuracy"])
    valloss.append(h.history["val_loss"])

plt.plot(range(1, len(trainacc[0]) + 1), np.mean(trainacc, axis=0), label="Training Accuracy")
plt.plot(range(1, len(valacc[0]) + 1), np.mean(valacc, axis=0), label="Validation Accuracy")
plt.title("MLP Training and Validation Accuracy")
plt.legend()
plt.savefig("./results/MLP_Accuracy.png")

# GAN Components
def build_generator():
    model = Sequential([
        Dense(256, input_dim=latent_dim), LeakyReLU(0.2), BatchNormalization(0.8),
        Dense(512), LeakyReLU(0.2), BatchNormalization(0.8),
        Dense(1024), LeakyReLU(0.2), BatchNormalization(0.8),
        Dense(np.prod(data_shape), activation="tanh"), Reshape(data_shape)
    ])
    noise = Input(shape=(latent_dim,))
    gendata = model(noise)
    return Model(noise, gendata)


def build_discriminator():
    model = Sequential([
        Flatten(input_shape=data_shape),
        Dense(512), LeakyReLU(0.2),
        Dense(256), LeakyReLU(0.2),
        Dense(1, activation="sigmoid")
    ])
    data = Input(shape=data_shape)
    validity = model(data)
    return Model(data, validity)

# GAN Training
gen_data = []
for i in tqdm(range(num_of_classes)):
    discriminator = build_discriminator()
    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5), metrics=["accuracy"])
    generator = build_generator()
    discriminator.trainable = False
    noise = Input(shape=(latent_dim,))
    validity = discriminator(generator(noise))
    combined = Model(noise, validity)
    combined.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))

    scaler = MinMaxScaler((-1, 1))
    Z_train_transformed = scaler.fit_transform(Z_train[i])
    Z_train_transformed = np.expand_dims(Z_train_transformed, axis=2)

    for epoch in range(gan_epochs):
        if Z_train_transformed.shape[0] < 64:
            raise ValueError("Insufficient samples for GAN training. Check your data preparation.")
        idx = np.random.randint(0, Z_train_transformed.shape[0], 64)
        real_data = Z_train_transformed[idx]
        noise = np.random.normal(0, 1, (64, latent_dim))
        fake_data = generator.predict(noise)

        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((64, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((64, 1)))

        # Train generator
        g_loss = combined.train_on_batch(noise, np.ones((64, 1)))
    gen_data.append(generator.predict(np.random.normal(0, 1, (data_to_gen, latent_dim))))
    clear_session()

# Add debug statements to ensure generated data and labels are correct
gen_data = np.asarray(gen_data, dtype=np.float32)
gen_label = np.tile(np.arange(num_of_classes), (data_to_gen,)).flatten()
assert len(gen_label) == len(gen_data.reshape(-1, data_shape[0]))


# MLP Results
trainacc, trainloss, valacc, valloss = [], [], [], []
for h in history:
    trainacc.append(h.history["accuracy"])
    trainloss.append(h.history["loss"])
    valacc.append(h.history["val_accuracy"])
    valloss.append(h.history["val_loss"])

plt.plot(range(1, len(trainacc[0]) + 1), np.mean(trainacc, axis=0), label="Training Accuracy")
plt.plot(range(1, len(valacc[0]) + 1), np.mean(valacc, axis=0), label="Validation Accuracy")
plt.title("MLP Training and Validation Accuracy")
plt.legend()
plt.savefig("./results/MLP_Accuracy.png")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Input
import numpy as np

latent_dim = 100  # Example, adjust based on your dataset
data_shape = (7,)  # Example, adjust based on your dataset


def build_generator():
    # Input layer with shape (latent_dim,)
    noise = Input(shape=(latent_dim,))

    # First dense layer
    x = Dense(256)(noise)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(0.8)(x)

    # Second dense layer
    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(0.8)(x)

    # Third dense layer
    x = Dense(1024)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(0.8)(x)

    # Output layer, reshaping to the required data shape
    x = Dense(np.prod(data_shape), activation="tanh")(x)
    output = Reshape(data_shape)(x)  # Reshaping the output to match data_shape

    # Creating the model
    model = Model(inputs=noise, outputs=output)

    return model


# Now build the generator model
generator = build_generator()
generator.summary()  # Check model summary to verify the architecture


def build_discriminator():
    model = Sequential([
        Flatten(input_shape=data_shape),
        Dense(512), LeakyReLU(0.2),
        Dense(256), LeakyReLU(0.2),
        Dense(1, activation="sigmoid")
    ])
    data = Input(shape=data_shape)
    validity = model(data)
    return Model(data, validity)


# GAN Training
gen_data = []
for i in tqdm(range(num_of_classes)):
    discriminator = build_discriminator()
    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5), metrics=["accuracy"])
    generator = build_generator()
    discriminator.trainable = False
    noise = Input(shape=(latent_dim,))
    validity = discriminator(generator(noise))
    combined = Model(noise, validity)
    combined.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))

    scaler = MinMaxScaler((-1, 1))
    Z_train_transformed = scaler.fit_transform(Z_train[i])
    Z_train_transformed = np.expand_dims(Z_train_transformed, axis=2)

    for epoch in range(gan_epochs):
        idx = np.random.randint(0, Z_train_transformed.shape[0], 64)
        real_data = Z_train_transformed[idx]
        noise = np.random.normal(0, 1, (64, latent_dim))
        fake_data = generator.predict(noise)

        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((64, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((64, 1)))

        # Train generator
        g_loss = combined.train_on_batch(noise, np.ones((64, 1)))
    gen_data.append(generator.predict(np.random.normal(0, 1, (data_to_gen, latent_dim))))
    clear_session()


trainacc = []
trainloss = []
valacc = []
valloss = []
for i in range(len(history)):
    trainacc.append(history[i].history["accuracy"])
    trainloss.append(history[i].history["loss"])
    valacc.append(history[i].history["val_accuracy"])
    valloss.append(history[i].history["val_loss"])

acc = np.mean(trainacc, axis=0)
val_acc = np.mean(valacc, axis=0)
loss = np.mean(trainloss, axis=0)
val_loss = np.mean(valloss, axis=0)
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy for {}%".format(fraction_of_data * 100))
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss for {}%".format(fraction_of_data * 100))
plt.legend()
plt.savefig("./results/original/Train - {}%.png".format(fraction_of_data * 100))


def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(data_shape), activation="tanh"))
    model.add(Reshape(data_shape))
    return model



def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=data_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dense(1, activation="sigmoid"))
    data = Input(shape=data_shape)
    validity = model(data)
    return Model(data, validity)


def train(epochs, features, batch_size=128):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    for epoch in range(epochs):
        idx = np.random.randint(0, features.shape[0], batch_size)
        data = features[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_data = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(data, valid)
        d_loss_fake = discriminator.train_on_batch(gen_data, fake)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)


gen_data = []
for i in tqdm(range(num_of_classes)):
    discriminator = build_discriminator()
    discriminator.compile(
        loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5), metrics=["accuracy"]
    )

    generator = build_generator()

    # GAN: Combined model
    noise = Input(shape=(latent_dim,))
    fake_data = generator(noise)
    discriminator.trainable = False
    validity = discriminator(fake_data)

    combined = Model(inputs=noise, outputs=validity)
    combined.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))

    minimaxscaler = MinMaxScaler((-1, 1))
    Z_train_transformed = minimaxscaler.fit_transform(Z_train[i])
    Z_train_transformed = np.expand_dims(Z_train_transformed, axis=2)
    train(epochs=gan_epochs, features=Z_train_transformed, batch_size=64)
    noise = np.random.normal(0, 1, (data_to_gen, latent_dim))
    gen_data_temp = generator.predict(noise)
    gen_data_temp = np.asarray(gen_data_temp, dtype=np.float32)
    gen_data_temp = np.squeeze(gen_data_temp)
    gen_data_temp = minimaxscaler.inverse_transform(gen_data_temp)
    gen_data.append(gen_data_temp)
    clear_session()
    del discriminator
    del generator
    del combined

gen_data = np.asarray(gen_data, dtype=np.float32)

gen_label = []
for i in range(num_of_classes):
    gen_label_temp = np.tile(i, data_to_gen)
    gen_label.extend(gen_label_temp)

gen_label = np.asarray(gen_label, dtype=np.float32)
gen_label_encoded = to_categorical(gen_label)

gen_data_reshaped = gen_data.reshape(num_of_classes * data_to_gen, data_shape[0])

X_train_gan, Y_train_gan = shuffle(
    gen_data_reshaped, gen_label_encoded, random_state=5
)

new_x_train = np.concatenate((X_train, X_train_gan), axis=0)
new_y_train = np.concatenate((Y_train_encoded, Y_train_gan), axis=0)

new_x_train, new_y_train = shuffle(new_x_train, new_y_train, random_state=15)
new_x_train_transformed = scaler.fit_transform(new_x_train)

all_test_loss_gan = []
all_test_acc_gan = []
ganhistory = []

for i in tqdm(range(times_to_run)):
    seed(i * seed_multiplier)
    set_random_seed(i * seed_multiplier)

    model = Sequential()
    model.add(Input(shape=(7,)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(4, activation="softmax"))

    model.compile(
        optimizer=Adam(0.0002, 0.5), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    ganhistorytemp = model.fit(
        new_x_train_transformed,
        new_y_train,
        epochs=mlp_epochs,
        batch_size=64,
        validation_split=valid_split,
        verbose=0,
    )
    ganhistory.append(ganhistorytemp)

    test_loss, test_acc = model.evaluate(X_test_transformed, Y_test_encoded, verbose=0)
    all_test_acc_gan.append(test_acc)
    all_test_loss_gan.append(test_loss)
    del model
    clear_session()

gantrainacc = []
gantrainloss = []
ganvalacc = []
ganvalloss = []

for i in range(len(ganhistory)):
    gantrainacc.append(ganhistory[i].history["accuracy"])
    gantrainloss.append(ganhistory[i].history["loss"])
    ganvalacc.append(ganhistory[i].history["val_accuracy"])
    ganvalloss.append(ganhistory[i].history["val_loss"])

gan_acc = np.mean(gantrainacc, axis=0)
gan_val_acc = np.mean(ganvalacc, axis=0)
gan_loss = np.mean(gantrainloss, axis=0)
gan_val_loss = np.mean(ganvalloss, axis=0)

plt.plot(range(1, len(gan_acc) + 1), gan_acc, "bo", label="Training acc")
plt.plot(range(1, len(gan_val_acc) + 1), gan_val_acc, "b", label="Validation acc")
plt.title("GAN Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(range(1, len(gan_loss) + 1), gan_loss, "bo", label="Training loss")
plt.plot(range(1, len(gan_val_loss) + 1), gan_val_loss, "b", label="Validation loss")
plt.title("GAN Training and validation loss")
plt.legend()
plt.savefig("./results/GAN/Train_GAN.png")

AccMean = np.mean(all_test_acc)
LossMean = np.mean(all_test_loss)
AccStd = np.std(all_test_acc)
LossStd = np.std(all_test_loss)

GanAccMean = np.mean(all_test_acc_gan)
GanLossMean = np.mean(all_test_loss_gan)
GanAccStd = np.std(all_test_acc_gan)
GanLossStd = np.std(all_test_loss_gan)

lines = []
lines.append(f"Original Data (Each Class: {num_of_data} Real):")
lines.append(f"Accuracy mean: {AccMean}")
lines.append(f"Loss mean: {LossMean}")
lines.append(f"Accuracy STD: {AccStd}")
lines.append(f"Loss STD: {LossStd}")
lines.append(f"Maximum Accuracy: {np.max(all_test_acc)}")
lines.append(
    f"Loss of Maximum Accuracy: {all_test_loss[np.argmax(all_test_acc)]}\n"
)
lines.append("=== GAN Data ===")
lines.append(f"Original + GAN Data (Each Class: {num_of_data} Real + {data_to_gen} GAN):")
lines.append(f"Accuracy mean: {GanAccMean}")
lines.append(f"Loss mean: {GanLossMean}")
lines.append(f"Accuracy STD: {GanAccStd}")
lines.append(f"Loss STD: {GanLossStd}")
lines.append(f"Maximum Accuracy: {np.max(all_test_acc_gan)}")
lines.append(
    f"Loss of Maximum Accuracy: {all_test_loss_gan[np.argmax(all_test_acc_gan)]}"
)

file_dir = "./results/Test_Results.txt"
with open(file_dir, "w") as filehandle:
    for items in lines:
        filehandle.write(f"{items}\n")
