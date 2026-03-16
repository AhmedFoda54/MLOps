import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import torch
import json

with open("config.json") as f:
    configs = json.load(f)["runs"]

mlflow.set_experiment("Assignment3_Ahmed")


latent_dim = 100
epochs = 50
batch_size = 128
learning_rate = 0.001
for config in configs:
    with mlflow.start_run():

        mlflow.log_params(config)
        epochs = config["epochs"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]
        latent_dim = config["latent_dim"]
        
        mlflow.set_tag("student_id", "202202064")
        print(f"Running experiment with config: {config}")

        (x_train, _), (_, _) = keras.datasets.mnist.load_data()

        x_train = x_train.astype("float32") / 127.5 - 1.0
        x_train = x_train.reshape(x_train.shape[0], 784)

        data = x_train

        generator = keras.Sequential([
            layers.Dense(128, activation="relu", input_shape=(latent_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(784, activation="tanh")
        ])

        discriminator = keras.Sequential([
            layers.Dense(256, activation="relu", input_shape=(784,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])

        discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        discriminator.trainable = False

        gan_input = layers.Input(shape=(latent_dim,))
        fake_image = generator(gan_input)
        gan_output = discriminator(fake_image)

        gan = keras.Model(gan_input, gan_output)

        gan.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="binary_crossentropy"
        )

        half_batch = batch_size // 2

        for epoch in range(epochs):

            idx = np.random.randint(0, data.shape[0], half_batch)
            real_imgs = data[idx]

            noise = np.random.normal(0, 1, (half_batch, latent_dim))
            fake_imgs = generator.predict(noise, verbose=0)

            d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((half_batch, 1)))

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

            loss = d_loss_real[0]
            acc = d_loss_real[1]

            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("accuracy", acc, step=epoch)

            if epoch % 5 == 0:
                print("Epoch", epoch, "| Loss:", loss, "| Accuracy:", acc)

        mlflow.keras.log_model(generator, "generator_model")