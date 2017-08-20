from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from termcolor import colored
from keras.utils import plot_model
from keras import callbacks
import keras


def normalizer(x):
    return (x - x.mean()) / x.std()


data = pd.read_csv('train_cleaned.csv')

best_cols = ['City_Category', 'Product_Category_1', 'Product_Category_2',
             'Product_Category_3', 'pid_4_f_lett_1', 'pid', 'Purchase']
# data = data.loc[:, best_cols]

# for col in data.columns:
#     data[col] = normalizer(data[col])

target = data.Purchase
data = data.drop('Purchase', axis=1)

# data size ==> 548000
x_train, x_test, y_train, y_test = train_test_split(data.values, target.values, test_size=.1)

model = Sequential()
model.add(Dense(units=20, input_dim=len(data.columns), activation='relu'))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))

# model = load_model('./keras_log/log')

tb_callb = callbacks.TensorBoard('./keras_log/tblogs', write_grads=True, write_graph=True)

reduce_lr_callb = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.6,
                                              patience=1, min_lr=0.00000000001, mode='auto', verbose=1)

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['mae'])
# keras.metrics
model.fit(x_train, y_train, validation_split=0.2, epochs=1, batch_size=100, callbacks=[reduce_lr_callb, tb_callb])
# loss = model.evaluate(, y_test, batch_size=2068)

# print(model.metrics_names[0], ':', loss[0], '\t',
#       colored(str(model.metrics_names[1]) + ': ' + str(loss[1]), color='green', attrs=['bold']))


model.save('./keras_log/log')

# test = pd.read_csv('test_cleaned.csv')
#
# submission = pd.DataFrame(columns=['User_ID', 'Product_ID', 'Purchase'])
# tmp_test = pd.read_csv('test.csv')
# submission.User_ID = tmp_test.User_ID
# submission.Product_ID = tmp_test.Product_ID
#
# submission.Purchase = model.predict(test.values)
# submission.to_csv('generated_sub_nn.csv', index=False)
