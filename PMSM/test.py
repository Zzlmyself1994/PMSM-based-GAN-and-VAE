
#from keras.datasets import mnist
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('-----------------------x_train-----------------------')
print(x_train)
print('------------------------y_train----------------------')
print(y_train)
'''

import numpy as np
import tensorflow as tf

#flags = tf.app.flags
#FLAGS = flags.FLAGS

#flags.DEFINE_string(
#    "train_data",
#    "C:/Users/Zzlmyself/PycharmProjects/PMSM/PMSM_train1.csv",
#    "Path to the training data.")
#tf.logging.set_verbosity(tf.logging.INFO)


#data=FLAGS.train_data

#print(data)

#path = np.genfromtxt("PMSM_train2.csv",dtype=np.str)
#path= path.astype(np.float)
#path= path.astype('float32')


training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename='PMSM_train1.csv',
      target_dtype=np.int,
      features_dtype=np.float64)
print(training_set.data,training_set.target)



