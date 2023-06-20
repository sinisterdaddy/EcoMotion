import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class QNetwork():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        initializer = tf.initializers.glorot_normal()
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[
                                     None, self.state_size])
        self.normalized_inputs = tf.divide(self.inputs, [[180.0, 180.0]])

        # Hidden layer 1
        self.W1 = tf.Variable(initializer(
            shape=[self.state_size, 10]), dtype=tf.float32, name='W1')
        self.b1 = tf.Variable(tf.zeros([10]), dtype=tf.float32, name='b1')
        h1 = tf.nn.relu(tf.matmul(self.normalized_inputs, self.W1) + self.b1)
        h1_drop = tf.nn.dropout(h1, keep_prob=0.75)

        # Hidden layer 2
        self.W2 = tf.Variable(initializer(
            shape=[10, 6]), dtype=tf.float32, name='W2')
        self.b2 = tf.Variable(tf.zeros([6]), dtype=tf.float32, name='b2')
        h2 = tf.nn.relu(tf.matmul(h1_drop, self.W2) + self.b2)
        h2_drop = tf.nn.dropout(h2, keep_prob=0.75)

        # Output layer
        self.W3 = tf.Variable(initializer(
            shape=[6, self.action_size]), dtype=tf.float32, name='W3')
        self.b3 = tf.Variable(
            tf.zeros([self.action_size]), dtype=tf.float32, name='b3')
        self.Q_values = tf.matmul(h2_drop, self.W3) + self.b3
        self.predictions = tf.argmax(input=self.Q_values, axis=1)

        # Training
        self.target_values = tf.placeholder(dtype=tf.float32, shape=[None])
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
        self.actions_onehot = tf.one_hot(
            indices=self.actions, depth=self.action_size, on_value=1, off_value=0)
        self.predictions_onehot = tf.cast(self.actions_onehot, tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(
            self.predictions_onehot, self.Q_values), axis=1)

        self.loss = tf.reduce_mean(tf.square(self.target_values - self.Q))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.train_op = self.optimizer.minimize(self.loss)
