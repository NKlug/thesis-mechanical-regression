from gradient_descent import find_optimal_p0
from swiss_roll_dataset import generate_swiss_roll_dataset
import tensorflow as tf

if __name__ == '__main__':
    X = tf.concat(generate_swiss_roll_dataset(1, jitter=0.1, coils=0.65), axis=0)
    X = tf.reshape(X, (-1))
    Y = tf.concat([tf.ones(100), -tf.ones(100)], axis=0)
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)

    p0 = find_optimal_p0(X, Y, steps=100000, checkpoint_its=2000)