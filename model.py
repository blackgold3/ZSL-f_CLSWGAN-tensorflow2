import tensorflow as tf

class generator(tf.keras.Model):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = tf.keras.layers.Dense(4096)          
        self.fc2 = tf.keras.layers.Dense(2048)
    
    def call(self, att, noise):
        inputs = tf.concat([att, noise], axis=1)
        gx = tf.nn.leaky_relu(self.fc1(inputs))
        gx = tf.nn.relu(self.fc2(gx))
        return gx

class discriminor(tf.keras.Model):
    def __init__(self):
        super(discriminor, self).__init__()
        self.fc1 = tf.keras.layers.Dense(1024)
        self.fc2 = tf.keras.layers.Dense(1)
    
    def call(self, g_x, att):
        inputs = tf.concat([g_x, att], axis=1)
        wd = tf.nn.leaky_relu(self.fc1(inputs))
        wd = self.fc2(wd)
        return wd

class softmax(tf.keras.Model):
    def __init__(self, class_num, regularizer):
        super(softmax, self).__init__()
        self.fc1 = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(l=regularizer))
        self.fc2 = tf.keras.layers.Dense(class_num, kernel_regularizer=tf.keras.regularizers.l2(l=regularizer))

    def call(self, inputs):
        pre = tf.nn.relu(self.fc1(inputs))
        pre = self.fc2(pre)
        pre = tf.nn.softmax(pre)
        return pre

class presoftmax(tf.keras.Model):
    def __init__(self, class_num):
        super(presoftmax, self).__init__()
        self.fc1 = tf.keras.layers.Dense(1024)
        self.fc2 = tf.keras.layers.Dense(class_num)

    def call(self, inputs):
        pre = tf.nn.relu(self.fc1(inputs))
        pre = self.fc2(pre)
        pre = tf.nn.softmax(pre)
        return pre
