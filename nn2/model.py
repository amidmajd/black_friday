import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
sns.set()
np.random.seed(100)

n_iter = 100
lr_rate = 0.001
keep_p = 1
batch_size = 100
log_dir = 'logs/model/'

def normalize(data):
    me = data.mean()
    std = data.std()
    return (data - me) / std, me, std

def unormalize(data, me, std):
    return (data * std) + me


def feed(X, net):
    l1 = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(X, net[1]['W'], net[1]['b'])), keep_prob)
    l2 = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(l1, net[2]['W'], net[2]['b'])), keep_prob)
    l3 = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(l2, net[3]['W'], net[3]['b'])), keep_prob)
    l4 = tf.nn.xw_plus_b(l3, net[4]['W'], net[4]['b'])
    return l4


# def add_layer(input, in_size, o_size, layer_name, active_func=None):
#     with tf.name_scope(layer_name):
#         W = tf.Variable(tf.truncated_normal(shape = [in_size, o_size],
#                                             dtype = tf.float32,
#                                             name = 'W'))
#         tf.summary.histogram(layer_name + '/W', W)
#         b = tf.Variable(tf.truncated_normal(shape = [o_size],
#                                             dtype = tf.float32,
#                                             name = 'b'))
#         tf.summary.histogram(layer_name + '/b', W)
#
#         layer = tf.nn.dropout(tf.nn.xw_plus_b(input, W, b), keep_prob)
#
#     if active_func is None:
#         return layer
#     else:
#         return active_func(layer)

testDataset = pd.read_csv('../test_cleaned.csv')
testDataset.Occupation, _, __ = normalize(testDataset.Occupation)
testDataset.pid, _, __ = normalize(testDataset.pid)


dataSet = pd.read_csv('../train_cleaned.csv')
cols = ['User_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
       'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
       'Product_Category_2', 'Product_Category_3', 'Purchase',
       'pid_4_f_lett_0', 'pid_4_f_lett_1', 'pid_4_f_lett_2', 'pid_4_f_lett_3',
       'pid_4_f_lett_4', 'pid_2_l_lett', 'pid'] # 17 + 1
# best_cols = ['City_Category', 'Product_Category_1', 'Product_Category_2',
#              'Product_Category_3', 'pid_4_f_lett_1', 'pid', 'Purchase']
# dataSet = dataSet.loc[:, best_cols]
dataSet.Occupation, _, __ = normalize(dataSet.Occupation)
# dataSet.Purchase, p_me, p_std = normalize(dataSet.Purchase)
dataSet.pid, _, __ = normalize(dataSet.pid)

y_data = dataSet.Purchase
x_data = dataSet.drop('Purchase', axis=1)

x_train, x_test, y_train, y_test = train_test_split(np.array(x_data).reshape(-1,17),
                                                    np.array(y_data).reshape(-1), test_size=10068)
# print(x_train.shape)
# print(y_train.shape)
with tf.name_scope('inputs'):
    X = tf.placeholder(shape=[None, 17], dtype=tf.float32, name='data')
    Y = tf.placeholder(shape=[None], dtype=tf.float32, name='target')
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
####network####
# l1 = add_layer(X, 17, 30, 'l1', tf.nn.relu)
# l2 = add_layer(l1, 30, 30, 'l2', tf.nn.relu)
# l3 = add_layer(l2, 30, 45, 'l3', tf.nn.relu)
# l4 = add_layer(l3, 45, 20, 'l4', tf.nn.relu)
# Y_hat = add_layer(l4, 20, 1, 'predictions')

net = {
    1: {
        'W': tf.Variable(tf.truncated_normal(shape=[17, 30], dtype=tf.float32)),
        'b': tf.Variable(tf.truncated_normal(shape=[30], dtype=tf.float32))
    },
    2: {
        'W': tf.Variable(tf.truncated_normal(shape=[30, 30], dtype=tf.float32)),
        'b': tf.Variable(tf.truncated_normal(shape=[30], dtype=tf.float32))
    },
    3: {
        'W': tf.Variable(tf.truncated_normal(shape=[30, 45], dtype=tf.float32)),
        'b': tf.Variable(tf.truncated_normal(shape=[45], dtype=tf.float32))
    },

    4: {
        'W': tf.Variable(tf.truncated_normal(shape=[45, 1], dtype=tf.float32)),
        'b': tf.Variable(tf.truncated_normal(shape=[1], dtype=tf.float32))
    }
}
Y_hat = tf.reshape(feed(X, net), [-1])
####network####

with tf.name_scope('TRAIN_STEP'):
    cost = tf.reduce_mean(tf.pow(tf.subtract(Y, Y_hat), 2))
    # cost = tf.pow(tf.subtract(Y, Y_hat), 2)
    rg = 0.1 * tf.nn.l2_loss(net[1]['W'])+ \
         0.1 * tf.nn.l2_loss(net[3]['W'])+ \
         0.1 * tf.nn.l2_loss(net[4]['W'])+ \
         0.1 * tf.nn.l2_loss(net[2]['W'])
    cost = cost + rg
    tf.summary.scalar('cost', cost)

    optimizer = tf.train.AdamOptimizer(lr_rate).minimize(cost)

n_sample = len(x_train)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
merged = tf.summary.merge_all()

feature_dec = PCA(n_components=1)
one_dim_f = feature_dec.fit_transform(x_test)
# print(one_dim_f)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(one_dim_f, y_test)
plt.ion()
plt.show()
with tf.Session() as sess:
    print('Training Process Started...')
    train_writer = tf.summary.FileWriter('logs/train/', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test/', sess.graph)
    # sess.run(init)
    # print(np.shape(sess.run(Y_hat, feed_dict={X:x_train[:100], keep_prob:1})))
    saver.restore(sess, log_dir)

    for i in range(n_iter):
        total_c = 0
        for j in range(n_sample // batch_size):
            # print('    batch {}/{}'.format(j+1,n_sample // batch_size))
            x_batch = x_train[j*batch_size : (j+1)*batch_size]
            y_batch = y_train[j*batch_size : (j+1)*batch_size]

            c, _= sess.run([cost, optimizer], feed_dict={X:x_batch, Y:y_batch, keep_prob:keep_p})
            total_c += c

        train_mrg = sess.run(merged,
                          feed_dict={X:x_train[:len(x_test)], Y:y_train[:len(y_test)], keep_prob:1.0})
        train_writer.add_summary(train_mrg, i)

        test_mrg = sess.run(merged,
                          feed_dict={X:x_test, Y:y_test, keep_prob:1.0})
        test_writer.add_summary(test_mrg, i)

        print('epoch {}/{}'.format(i+1, n_iter), '  cost :', total_c)

        # print(y_test[:10].reshape(-1,1))
        # print(np.array(sess.run([Y_hat], feed_dict={X: x_test[:10], keep_prob:1})).reshape(-1,1))

        ax.clear()
        ax.scatter(one_dim_f, y_test, c='blue', s=15)
        ax.scatter(one_dim_f,
                    sess.run(Y_hat, feed_dict={X:x_test, keep_prob:1}),
                    c='red',
                    s=10)
        ax.set_title('Step: {}    Loss: {}'.format(i+1, total_c))
        plt.pause(0.5)

    print('Train finished, saving model...')
    saver.save(sess, log_dir)

    print('Saving submission file...')
    y_pred = sess.run(Y_hat, feed_dict={X: np.array(testDataset).reshape(-1,17),
                                        keep_prob: 1.0})
    y_pred = np.array(y_pred).reshape(-1,1)



submission = pd.DataFrame(columns=['User_ID', 'Product_ID', 'Purchase'])
tmp_test = pd.read_csv('../test.csv')
submission.User_ID = tmp_test.User_ID
submission.Product_ID = tmp_test.Product_ID


submission.Purchase = y_pred
submission.to_csv('generated_sub_nn.csv', index=False)
