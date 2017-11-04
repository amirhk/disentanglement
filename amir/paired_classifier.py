
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn import mixture

from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D,UpSampling2D, Flatten, Reshape, Dropout
from keras.models import Model
from keras import backend as K
from keras import objectives , utils,optimizers
from keras.datasets import mnist
from mpl_toolkits.mplot3d import Axes3D
from keras.callbacks import Callback, LearningRateScheduler
import tensorflow as tf

import sys
sys.path.append('../utils')
#
#from importDatasets import importMnist
#from importDatasets import importMnistFashion
#from importDatasets import importOlivetti
#from importDatasets import importSquareAndCross


from datetime import datetime
from importDatasets import importMnistAndSvhn
# -----------------------------------------------------------------------------
#                                                                    Fetch Data
# -----------------------------------------------------------------------------

(dataset_name_1, dataset_name_2, x_train_1,
     x_train_2, y_train, x_test_1, y_test_1,
     x_test_2, y_test_2, num_classes) = importMnistAndSvhn()

training_size = 40000
x_val_1 = x_train_1[training_size:,:]
x_train_1 =x_train_1[:training_size,:]

y_val = y_train[training_size:,:]
y_train = y_train[:training_size,:]

x_val_2 = x_train_2[training_size:,:]
x_train_2 =x_train_2[:training_size,:]

x_test_2 = x_test_2[:10000,:]
y_test_2 = y_test_2[:10000,:]


batch_size = 100
latent_dim_x_1 = 10
latent_dim_x_2 = 10
latent_dim_y = 10
epochs = 200
intermediate_dim = 500
epsilon_std = 1.0
learning_rate = 0.0003
original_dim_1 = 784
original_dim_2  = 32*32*3

# -----------------------------------------------------------------------------
#                                                                   Build Model
# -----------------------------------------------------------------------------

dataset_name = dataset_name_1 + dataset_name_2

experiment_name = dataset_name + \
  '_____z_dim_' + str(latent_dim_y)

  # if ~ os.path.isdir('../experiments'):
  #   os.makedirs('../experiments')
experiment_dir_path = '../experiments/exp' + \
  '_____' + \
  str(datetime.now().strftime('%Y-%m-%d_____%H-%M-%S')) + \
  '_____' + \
  experiment_name
os.makedirs(experiment_dir_path)


########## Meta #########################################################################################

def sampling(args):
    z_mean, z_log_var = args
    latent_dim = int(z_mean.shape[1])
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def build_z(args):
    z_1 ,z_2 = args
    return tf.concat([z_1,z_2],1)

########## Autoencoder 1 Network ########################################################################

x_1 = Input(batch_shape=(batch_size, original_dim_1))
x_reshaped_1 = Reshape((28,28,1))
h_e_1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')
h_e_1_2 = MaxPooling2D((2, 2), padding='same')
h_e_1_3 = Conv2D(32, (3, 3), activation='relu', padding='same')
h_e_1_4 = MaxPooling2D((2, 2), padding='same')
h_e_1_5 = Conv2D(16, (3, 3), activation='relu', padding='same')
h_e_1_6 = MaxPooling2D((2, 2), padding='same')
h_e_1_7 = Flatten()

z_1 = Dense(latent_dim_x_1)

####### Autoencoder 2 Network ###########################################################################

x_2 = Input(batch_shape=(batch_size, original_dim_2))
x_reshaped_2 = Reshape((32,32,3))
h_e_2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')
h_e_2_2 = MaxPooling2D((2, 2), padding='same')
h_e_2_3 = Conv2D(96, (3, 3), activation='relu', padding='same')
h_e_2_4 = MaxPooling2D((2, 2), padding='same')
h_e_2_5 = Conv2D(64, (3, 3), activation='relu', padding='same')
h_e_2_6 = MaxPooling2D((2, 2), padding='same')
h_e_2_7 = Conv2D(8, (3, 3), activation='relu', padding='same')
h_e_2_8 = Flatten()

z_2 = Dense(latent_dim_x_2)

## Classifier Network ###################################################################################

h_d_y_1 = Dense(intermediate_dim, activation='relu')
h_d_y_2 = Dropout(0.5)
h_d_y_3 = Dense(intermediate_dim, activation='relu')
h_d_y_4 = Dropout(0.5)
h_d_y_5 = Dense(intermediate_dim, activation='relu')
h_d_y_6 = Dropout(0.5)
y_decoded = Dense(10, activation='softmax')

yy_1 = Input(batch_shape = (batch_size, 10))
yy_2 = Input(batch_shape = (batch_size, 10))

##### Build model 1 #####################################################################################

_x_reshaped_1 = x_reshaped_1(x_1)
_h_e_1_1 = h_e_1_1(_x_reshaped_1)
_h_e_1_2 = h_e_1_2(_h_e_1_1)
_h_e_1_3 = h_e_1_3(_h_e_1_2)
_h_e_1_4 = h_e_1_4(_h_e_1_3)
_h_e_1_5 = h_e_1_5(_h_e_1_4)
_h_e_1_6 = h_e_1_6(_h_e_1_5)
_h_e_1_7 = h_e_1_7(_h_e_1_6)

_z_1 = z_1(_h_e_1_7)

##### Build model 2 #####################################################################################

_x_reshaped_2 = x_reshaped_2(x_2)
_h_e_2_1 = h_e_2_1(_x_reshaped_2)
_h_e_2_2 = h_e_2_2(_h_e_2_1)
_h_e_2_3 = h_e_2_3(_h_e_2_2)
_h_e_2_4 = h_e_2_4(_h_e_2_3)
_h_e_2_5 = h_e_2_5(_h_e_2_4)
_h_e_2_6 = h_e_2_6(_h_e_2_5)
_h_e_2_7 = h_e_2_7(_h_e_2_6)
_h_e_2_8 = h_e_2_8(_h_e_2_7)

_z_2 = z_2(_h_e_2_8)

##### Build Classifier ##################################################################################

_h_d_y_1_1 = h_d_y_1(_z_1)
_h_d_y_1_2 = h_d_y_2(_h_d_y_1_1)
_h_d_y_1_3 = h_d_y_3(_h_d_y_1_2)
_h_d_y_1_4 = h_d_y_4(_h_d_y_1_3)
_h_d_y_1_5 = h_d_y_5(_h_d_y_1_4)
_h_d_y_1_6 = h_d_y_6(_h_d_y_1_5)
_y_decoded_1 = y_decoded(_h_d_y_1_6)

_h_d_y_2_1 = h_d_y_1(_z_2)
_h_d_y_2_2 = h_d_y_2(_h_d_y_2_1)
_h_d_y_2_3 = h_d_y_3(_h_d_y_2_2)
_h_d_y_2_4 = h_d_y_4(_h_d_y_2_3)
_h_d_y_2_5 = h_d_y_5(_h_d_y_2_4)
_h_d_y_2_6 = h_d_y_6(_h_d_y_2_5)
_y_decoded_2 = y_decoded(_h_d_y_2_6)

###### Define Loss ######################################################################################

def vae_loss(x, _x_decoded):
    y_loss_1 = 10 * objectives.categorical_crossentropy(yy_1, _y_decoded_1)
    y_loss_2 = 100 * objectives.categorical_crossentropy(yy_2, _y_decoded_2)
    return y_loss_1 + y_loss_2

model = Model(inputs = [x_1, x_2, yy_1, yy_2],outputs = [_y_decoded_1, _y_decoded_2])
my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.1)
model.compile(optimizer=my_adam, loss=vae_loss)

############################################################################
############################################################################
#### Build another model (NO NEED) #########################################

_y_decoded_1_ = _y_decoded_1
_y_decoded_2_ = _y_decoded_2

vaeencoder = Model(inputs = [x_1, x_2], outputs = [_y_decoded_1_, _y_decoded_2_])

############################################################################
############################################################################
############################################################################
# -----------------------------------------------------------------------------
#                                                                   Train Model
## -----------------------------------------------------------------------------
not_hot_y_test_1 = np.argmax(y_test_1, axis = 1)
not_hot_y_test_2 = np.argmax(y_test_2, axis = 1)

Accuracy = np.zeros((epochs,2))
ii=0
pickle.dump((ii),open('counter','wb'))
text_file_name = experiment_dir_path + '/accuracy_log.txt'
class ACCURACY(Callback):

    def on_epoch_end(self,batch,logs = {}):
        ii= pickle.load(open('counter', 'rb'))
        b_1, b_2  = vaeencoder.predict([x_test_1, x_test_2], batch_size = batch_size)

        Accuracy[ii, 0]

        lll_1 = np.argmax(b_1, axis =1)
        lll_2 = np.argmax(b_2, axis =1)
        lll_1= np.reshape(lll_1,(len(not_hot_y_test_1),))
        lll_2= np.reshape(lll_2,(len(not_hot_y_test_1),))
        n_error_1 = np.count_nonzero(lll_1 - not_hot_y_test_1)
        n_error_2 = np.count_nonzero(lll_2 - not_hot_y_test_2)
        ACC_1 = 1 - n_error_1 / len(not_hot_y_test_1)
        ACC_2 = 1 - n_error_2 / len(not_hot_y_test_1)
        Accuracy[ii,:] = [ACC_1 , ACC_2]
        print('\n accuracy_mnist = ', ACC_1, ' accuracy_svhn = ', ACC_2, '\n\n')
        ii= ii + 1
        pickle.dump((ii),open('counter', 'wb'))
        with open(text_file_name, 'a') as text_file:
          print('Epoch #{} Accuracy MNIST:{} Accuracy SVHN:{} \n'.format(ii, ACC_1, ACC_2), file=text_file)

accuracy = ACCURACY()

def scheduler(epoch):
    # if epoch > 200:
    #     return float(0.0001)
    if epoch > 100:
        return float(0.0003)
    else:
        return float(0.001) # initial_lrate

change_lr = LearningRateScheduler(scheduler)


model_weights = pickle.load(open('paired_classifier' + str(latent_dim_y) + 'd_trained_on_' + dataset_name, 'rb'))
model.set_weights(model_weights)

model.fit([x_train_1, x_train_2, y_train, y_train], [y_train, y_train],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data =([x_val_1, x_val_2, y_val, y_val], [y_val, y_val]),
        callbacks = [accuracy, change_lr])

model_weights = model.get_weights()
pickle.dump((model_weights), open('paired_classifier' + str(latent_dim_y) + 'd_trained_on_' + dataset_name, 'wb'))

