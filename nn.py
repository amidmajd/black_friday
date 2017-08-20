import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from termcolor import colored

hide_arch = [8, 8]
layers = 2
batch_size = 50
ln_rate = .01
epoch = 1000
restore = True


def normalizer(x):
    return (x - x.mean()) / x.std()


def feed(X, net, layers):
    out = [
        tf.nn.relu(tf.nn.xw_plus_b(X, net[0]['W'], net[0]['B']))
    ]
    for i in range(1, layers):
        out.append(tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(out[-1], net[i]['W'], net[i]['B'])), keep_prob=0.65))

    return tf.nn.xw_plus_b(out[-1], net[layers]['W'], net[layers]['B'])


def score(x, y_true):
    # with tf.Session() as sess:
    #     saver.restore(sess, './tflog/model.ckpt')
    #     y_pred = np.array(sess.run([y_hat], feed_dict={X: x}))[0].ravel()
    y_pred = np.array(sess.run([y_hat], feed_dict={X: x}))[0].ravel()

    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (u / v)


data = pd.read_csv('train_cleaned.csv')

best_cols = ['City_Category', 'Product_Category_1', 'Product_Category_2',
             'Product_Category_3', 'pid_4_f_lett_1', 'pid', 'Purchase']
data = data.loc[:, best_cols]

for col in ['Product_Category_1', 'Product_Category_2',
            'Product_Category_3', 'pid', 'Purchase']:
    data[col] = normalizer(data[col])

target = data.Purchase
data = data.drop('Purchase', axis=1)

# data size ==> 548000
x_train, x_test, y_train, y_test = train_test_split(data.values, target.values, test_size=2068)

# data = data.iloc[:16000, :]
# target = target.iloc[:16000]
# x_train, x_test, y_train, y_test = train_test_split(data.values, target.values, test_size=1000)

arch = [len(data.columns)] + hide_arch + [1]

net = {

    i: {
        'W': tf.Variable(tf.truncated_normal(shape=[arch[i], arch[i + 1]], name='W{}'.format(i))),
        'B': tf.Variable(tf.truncated_normal(shape=[arch[i + 1]], name='B{}'.format(i)))
    }

    for i in range(0, len(arch) - 1)

    }

X = tf.placeholder(shape=[None, len(data.columns)], dtype=tf.float32, name='data')
Y = tf.placeholder(shape=[None], dtype=tf.float32, name='target')

y_hat = feed(X, net, layers)

cost = tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(Y, y_hat), 2)))
optimizer = tf.train.AdamOptimizer(learning_rate=ln_rate).minimize(cost)

sample_size = len(y_train)

saver = tf.train.Saver()
with tf.Session() as sess:
    if restore:
        try:
            open('./tflog/model.ckpt.meta', 'r')
            saver.restore(sess, './tflog/model.ckpt')
            print('\n++Save found! , restoring...')
        except Exception as e:
            sess.run([tf.global_variables_initializer()])
            print('\n--Save Not found! , initializing...')
    else:
        sess.run([tf.global_variables_initializer()])

    for i in range(epoch):
        total = 0
        for j in range(int(sample_size / batch_size)):
            x, y = x_train[(j * batch_size): (j + 1) * batch_size], y_train[(j * batch_size): (j + 1) * batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={
                X: x,
                Y: y
            })

            total += c

        print(colored(str(i + 1) + '/' + str(epoch) + ' :', 'red'), 'cost =',
              colored(total, color='green'), end='')
        saver.save(sess, './tflog/model.ckpt')

        print(colored('\t   score :', 'cyan', 'on_grey', ['bold']), '{:.5f} %'.format(score(x_test, y_test) * 100))
