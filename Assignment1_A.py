#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


(x_train, _), (_, _) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 127.5 - 1.0
x_train = x_train.reshape(x_train.shape[0], 784)

df = pd.DataFrame(x_train)
df.to_csv("mnist_data.csv", index=False)

data = pd.read_csv("mnist_data.csv").values


# In[4]:


latent_dim = 100

generator = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(latent_dim,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(784, activation="tanh")
])


# In[5]:


discriminator = keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

discriminator.compile(
    optimizer=keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# In[6]:


discriminator.trainable = False

gan_input = layers.Input(shape=(latent_dim,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)

gan = keras.Model(gan_input, gan_output)

gan.compile(
    optimizer=keras.optimizers.Adam(),
    loss="binary_crossentropy"
)


# In[10]:


epochs = 50
batch_size = 128
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

    if epoch % 5 == 0:
        print("Epoch", epoch, "| D Loss:", d_loss_real[0], "| D Acc:", d_loss_real[1]*100, "% | G Loss:", g_loss)


# In[11]:


idx = np.random.randint(0, data.shape[0], 1000)
real_imgs = data[idx]

noise = np.random.normal(0, 1, (1000, latent_dim))
fake_imgs = generator.predict(noise, verbose=0)

d_loss_real = discriminator.evaluate(real_imgs, np.ones((1000, 1)), verbose=0)
d_loss_fake = discriminator.evaluate(fake_imgs, np.zeros((1000, 1)), verbose=0)

final_acc = (d_loss_real[1] + d_loss_fake[1]) / 2

print("Final Discriminator Accuracy:", final_acc * 100)

