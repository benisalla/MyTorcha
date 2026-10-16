import numpy as np
import pandas as pd
from Models.SimpleFC import SimpleFC
from NN.Trainer import Trainer

# load data
dataset = pd.read_csv('./data/train.csv')

# splitting data
data = np.array(dataset)
size, dim = data.shape

tr_size = int(0.9 * size)
vl_size = size - tr_size

print(f"train data size : {tr_size}")
print(f"validation data size : {vl_size}")


# shuffle data first (to simulate a normal distribution)
np.random.shuffle(data)

# build val data and preprocess it
val_data = data[0:tr_size]
Y_val = val_data[:, 0]
X_val = val_data[:, 1:dim]
X_val = X_val / 255.   # scaling

tr_data = data[tr_size:tr_size+vl_size]
Y_train = tr_data[:, 0]
X_train = tr_data[:, 1:dim]
X_train = X_train / 255.  # scaling


# initializing Model and Trainer
model = SimpleFC(784, [], 10, act_fun='relu')
trainer = Trainer(model=model,
                  lr=0.2,
                  batch_size=16,
                  loss="cross_entropy",
                  optimizer="rms_prop",
                  beta1=0.99, beta2=0.994)


# run test
trainer.fit(X_train, Y_train, 60)


# test our model
trainer.predict(X_train[10:24], Y_train[10:24])
