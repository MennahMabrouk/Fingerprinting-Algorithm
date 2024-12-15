import numpy as np
from keras.models import Model
from keras.layers import Dense, LeakyReLU, Reshape, Input, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LayerNormalization

latent_dim = 100
num_of_classes = 4
data_shape = (7,)  # Define the shape of the data
epochs = 5000
batch_size = 64
data_to_generate = 250


# Build Generator Model
def build_generator():
    # Define input shape for the generator
    noise = Input(shape=(latent_dim,))

    # First Dense layer
    x = Dense(256)(noise)
    x = LeakyReLU(0.2)(x)
    x = LayerNormalization()(x)  # Using LayerNormalization instead of BatchNormalization

    # Second Dense layer
    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    x = LayerNormalization()(x)  # Using LayerNormalization instead of BatchNormalization

    # Third Dense layer
    x = Dense(1024)(x)
    x = LeakyReLU(0.2)(x)
    x = LayerNormalization()(x)  # Using LayerNormalization instead of BatchNormalization

    # Output layer
    x = Dense(np.prod(data_shape), activation="tanh")(x)
    generated_data = Reshape(data_shape)(x)

    return Model(noise, generated_data)


# Build Discriminator Model
def build_discriminator():
    data = Input(shape=data_shape)  # Explicit input shape for the discriminator
    x = Flatten()(data)
    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    validity = Dense(1, activation="sigmoid")(x)

    return Model(data, validity)


# GAN Training Function
def train_gan(generator, discriminator, combined, features):
    valid = np.ones((batch_size, 1))  # Labels for real data
    fake = np.zeros((batch_size, 1))  # Labels for fake data

    for epoch in range(epochs):
        # Select a random batch of real data
        idx = np.random.randint(0, features.shape[0], batch_size)
        real_data = features[idx]

        # Generate a batch of fake data
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)

        # Train the discriminator on real and fake data
        d_loss_real = discriminator.train_on_batch(real_data, valid)  # Real data, labeled as 1
        d_loss_fake = discriminator.train_on_batch(fake_data, fake)  # Fake data, labeled as 0

        # Train the generator (by training the combined model)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)  # Train generator with labels for real data

        # Print the progress every 100 epochs
        if epoch % 100 == 0:
            print(f"{epoch}/{epochs} [D loss: {0.5 * (d_loss_real + d_loss_fake)}] [G loss: {g_loss}]")


# Prepare Data
Z_train = np.random.rand(data_to_generate, data_shape[0])  # Example data, you should replace this with your real data
scaler = MinMaxScaler((-1, 1))  # Normalize the data to the range (-1, 1)
Z_train_scaled = scaler.fit_transform(Z_train)

# Build Models
discriminator = build_discriminator()
discriminator.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5), metrics=["accuracy"])

generator = build_generator()

# We want to train the generator through the combined model (discriminator frozen)
discriminator.trainable = False
noise = Input(shape=(latent_dim,))
validity = discriminator(generator(noise))  # Generate validity based on generated data

# Combined model (generator + discriminator)
combined = Model(noise, validity)
combined.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))

# Train GAN
train_gan(generator, discriminator, combined, Z_train_scaled)

# Generate Data
gen_data = generator.predict(np.random.normal(0, 1, (data_to_generate, latent_dim)))

# Generate Labels (just a repetitive set for visualization purposes)
gen_label = np.tile(np.arange(num_of_classes), data_to_generate // num_of_classes)

# Plot Generated Data
plt.figure(figsize=(10, 5))
plt.scatter(gen_data[:, 0], gen_data[:, 1], c=gen_label, cmap="viridis")  # Plot first two features
plt.title("Generated Data")
plt.savefig("./results/Generated_Data.png")
