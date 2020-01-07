import numpy as np
import tensorflow as tf


def sampler(p1, p2):
    return 1


# rdefault = tf.tile(tf.constant(1, shape=(1, 1)), (1, 256))

# b2 = (W == 0) & (b == 0)
# b3 = tf.cast(tf.tile(tf.constant(b2, shape=(1, 1)), (1, 22)), tf.bool)
# t2 = tf.cast(b3, tf.bool)

fa = tf.constant([12.0, 24.0])
fb = tf.constant([15.5])


# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# tf.reshape(prediction_dict[rea_hack_sig]['pred_color'][0, :], (1, -1)),
# temp add
z = tf.reshape(x, (1, -1))

# xx = tf.constant([[], [], [], [], []], dtype=tf.int32)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong

for i in range(3):
    print(sess.run(train, {x: x_train, y: y_train}))

# evaluate training accuracy
# curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
# print("xxxxxxxxxxx")
# print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
# print(sess.run([p1]))
#
# print(" ===> nan")
# print(sess.run([q_values_nan]))
#
# print(" ===> is not nan")
# print (sess.run([q_values_not_nan]))

t = {"a": linear_model, "b": linear_model}
print(sess.run(t))


# tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
# aaaa =  tf.where(tf.equal(idx, -2), -2, idx + 1)
# aaaa = tf.case([tf.equal(idx, -2), -2], default=1)
# print(sess.run([t]))
