
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
x_train_1 = x_train_1[:training_size,:]

y_val = y_train[training_size:,:]

y_train = y_train[:training_size,:]

x_val_2 = x_train_2[training_size:,:]
x_train_2 = x_train_2[:training_size,:]

x_test_2 = x_test_2[:10000,:]
y_test_2 = y_test_2[:10000,:]



batch_size = 100
latent_dim_x_1 = 5
latent_dim_x_2 = 50
latent_dim_y = 10
epochs = 100
intermediate_dim = 500
epsilon_std = 1.0
learning_rate = 0.001
original_dim_1 = 784
original_dim_2  = 32*32*3

# -----------------------------------------------------------------------------
#                                                                   Build Model
# -----------------------------------------------------------------------------

dataset_name = dataset_name_2

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

########## Autoencoder 1 Network ########################################################

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

h_d_x_1_1 = Dense(4*4*8, activation = 'relu')
h_d_x_1_2 = Reshape((4,4,8))
h_d_x_1_3 = Conv2D(16, (3, 3), activation='relu', padding='same')
h_d_x_1_4 = UpSampling2D((2, 2))
h_d_x_1_5 = Conv2D(32, (3, 3), activation='relu', padding='same')
h_d_x_1_6 = UpSampling2D((2, 2))
h_d_x_1_7 = Conv2D(32, (3, 3), activation='relu')
h_d_x_1_8 = UpSampling2D((2, 2))
x_decoded_reshaped_1 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')
x_decoded_1 = Flatten()

####### Autoencoder 2 Network ###########################################################

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

h_d_x_2_1 = Dense(4*4*8, activation = 'relu')
h_d_x_2_2 = Reshape((4,4,8))
h_d_x_2_3 = Conv2D(64, (3, 3), activation='relu', padding='same')
h_d_x_2_4 = UpSampling2D((2, 2))
h_d_x_2_5 = Conv2D(96, (3, 3), activation='relu', padding='same')
h_d_x_2_6 = UpSampling2D((2, 2))
h_d_x_2_7 = Conv2D(64, (3, 3), activation='relu', padding='same')
h_d_x_2_8 = UpSampling2D((2, 2))
x_decoded_reshaped_2 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')
x_decoded_2 = Flatten()


##### Build model 1 #########################################################################################
_x_reshaped_1 = x_reshaped_1(x_1)
_h_e_1_1 = h_e_1_1(_x_reshaped_1)
_h_e_1_2 = h_e_1_2(_h_e_1_1)
_h_e_1_3 = h_e_1_3(_h_e_1_2)
_h_e_1_4 = h_e_1_4(_h_e_1_3)
_h_e_1_5 = h_e_1_5(_h_e_1_4)
_h_e_1_6 = h_e_1_6(_h_e_1_5)
_h_e_1_7 = h_e_1_7(_h_e_1_6)

_z_1 = z_1(_h_e_1_7)

_h_d_x_1_1 = h_d_x_1_1(_z_1)
_h_d_x_1_2 = h_d_x_1_2(_h_d_x_1_1)
_h_d_x_1_3 = h_d_x_1_3(_h_d_x_1_2)
_h_d_x_1_4 = h_d_x_1_4(_h_d_x_1_3)
_h_d_x_1_5 = h_d_x_1_5(_h_d_x_1_4)
_h_d_x_1_6 = h_d_x_1_6(_h_d_x_1_5)
_h_d_x_1_7 = h_d_x_1_7(_h_d_x_1_6)
_h_d_x_1_8 = h_d_x_1_8(_h_d_x_1_7)
_x_decoded_reshaped_1 = x_decoded_reshaped_1(_h_d_x_1_8)
_x_decoded_1 = x_decoded_1(_x_decoded_reshaped_1)

##### Build model 2 #########################################################################################
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

_h_d_x_2_1 = h_d_x_2_1(_z_2)
_h_d_x_2_2 = h_d_x_2_2(_h_d_x_2_1)
_h_d_x_2_3 = h_d_x_2_3(_h_d_x_2_2)
_h_d_x_2_4 = h_d_x_2_4(_h_d_x_2_3)
_h_d_x_2_5 = h_d_x_2_5(_h_d_x_2_4)
_h_d_x_2_6 = h_d_x_2_6(_h_d_x_2_5)
_h_d_x_2_7 = h_d_x_2_7(_h_d_x_2_6)
_h_d_x_2_8 = h_d_x_2_8(_h_d_x_2_7)
_x_decoded_reshaped_2 = x_decoded_reshaped_2(_h_d_x_2_8)
_x_decoded_2 = x_decoded_2(_x_decoded_reshaped_2)


###### Define Loss ###########################################################################

def ae_loss(x, _x_decoded):

    xent_loss_1 = original_dim_1 * objectives.binary_crossentropy(x_1, _x_decoded_1)
    # xent_loss_2 = original_dim_2 * objectives.binary_crossentropy(x_2, _x_decoded_2)
    return xent_loss_1
    # return xent_loss_2

model = Model(inputs = [x_1],outputs = [_x_decoded_1])
# model = Model(inputs = [x_2],outputs = [_x_decoded_2])
my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.1)
model.compile(optimizer=my_adam, loss=ae_loss)



# vaeencoder = Model(inputs = [x_1,x_2],outputs = [_x_decoded_1_,_x_decoded_2_,_y_decoded_1_,_y_decoded_2_])
############################################################################
############################################################################
############################################################################
# -----------------------------------------------------------------------------
#                                                                   Train Model
## -----------------------------------------------------------------------------
# y_test_label_1 = np.argmax(y_test_1,axis =1)
# y_test_label_2 = np.argmax(y_test_2,axis =1)

# #y_test_label = np.reshape(y_test_label,(y_test_label.shape[0],1))
# #y_test_label = np.reshape(y_test_label,(y_test_label.shape[0],1))



# Accuracy = np.zeros((epochs,2))
# ii=0
# pickle.dump((ii),open('counter','wb'))
# text_file_name = experiment_dir_path + '/accuracy_log.txt'
# class ACCURACY(Callback):

#     def on_epoch_end(self,batch,logs = {}):
#         ii= pickle.load(open('counter', 'rb'))
#         _,_, b_1,b_2  = vaeencoder.predict([x_test_1,x_test_2], batch_size = batch_size)


#         Accuracy[ii, 0]

#         lll_1 = np.argmax(b_1, axis =1)
#         lll_2 = np.argmax(b_2, axis =1)
#         lll_1= np.reshape(lll_1,(10000,))
#         lll_2= np.reshape(lll_2,(10000,))
#         n_error_1 = np.count_nonzero(lll_1 - y_test_label_1)
#         n_error_2 = np.count_nonzero(lll_2 - y_test_label_2)
#         ACC_1 = 1 - n_error_1 / 10000
#         ACC_2 = 1 - n_error_2 / 10000
#         Accuracy[ii,:] = [ACC_1 , ACC_2]
#         print('\n accuracy_mnist = ', ACC_1 , ' accuracy_svhn = ', ACC_2)
#         ii= ii + 1
#         pickle.dump((ii),open('counter', 'wb'))
#         with open(text_file_name, 'a') as text_file:
#           print('Epoch #{} Accuracy MNIST:{} Accuracy SVHN:{} \n'.format(ii, ACC_1, ACC_2), file=text_file)


# accuracy = ACCURACY()


class RECONSTRUCTION(Callback):

    def getFigureOfSamplesForInput(self, x_samples, image_dim, image_channels, number_of_sample_images, grid_x=range(10), grid_y=range(10)):
        if image_channels == 1:
            figure = np.zeros((image_dim * number_of_sample_images, image_dim * number_of_sample_images))
            for i in range(number_of_sample_images):
                for j in range(number_of_sample_images):
                    digit = x_samples[i * number_of_sample_images + j].reshape(image_dim, image_dim)
                    figure[i * image_dim: (i + 1) * image_dim,
                           j * image_dim: (j + 1) * image_dim] = digit
            return figure
        elif image_channels == 3:
            figure = np.zeros((image_dim * number_of_sample_images, image_dim * number_of_sample_images, image_channels))
            for i in range(number_of_sample_images):
                for j in range(number_of_sample_images):
                    digit = x_samples[i * number_of_sample_images + j, :].reshape(image_dim, image_dim, image_channels)
                    figure[i * image_dim: (i + 1) * image_dim,
                           j * image_dim: (j + 1) * image_dim, :] = digit
            return figure


    def plotAndSaveOriginalAndReconstructedImages(self, original_x, reconstructed_x, image_dim, image_channels, file_name):
        number_of_sample_images = 10

        plt.figure()

        ax = plt.subplot(1,2,1)
        x_samples = original_x
        canvas = self.getFigureOfSamplesForInput(x_samples, image_dim, image_channels, number_of_sample_images)
        plt.imshow(canvas)
        ax.set_title('Original Images', fontsize=8)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(1,2,2)
        x_samples = reconstructed_x
        canvas = self.getFigureOfSamplesForInput(x_samples, image_dim, image_channels, number_of_sample_images)
        plt.imshow(canvas)
        ax.set_title('Reconstructed Images', fontsize=8)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig(file_name)
        plt.close('all')


    def on_epoch_end(self,batch,logs = {}):

        timestamp_string = str(datetime.now().strftime('%Y-%m-%d_____%H-%M-%S'))
        image_file_name_1 = experiment_dir_path + '/' + timestamp_string + '_reconstructed_samples_1.png'
        # image_file_name_2 = experiment_dir_path + '/' + timestamp_string + '_reconstructed_samples_2.png'

        reconstructed_x_test_1 = model.predict([x_test_1], batch_size = batch_size)
        # reconstructed_x_test_2 = model.predict([x_test_2], batch_size = batch_size)

        original_x = x_test_1
        reconstructed_x = reconstructed_x_test_1
        image_dim = 28
        image_channels = 1
        file_name = image_file_name_1
        self.plotAndSaveOriginalAndReconstructedImages(original_x, reconstructed_x, image_dim, image_channels, file_name)

        # original_x = x_test_2
        # reconstructed_x = reconstructed_x_test_2
        # image_dim = 32
        # image_channels = 3
        # file_name = image_file_name_2
        # self.plotAndSaveOriginalAndReconstructedImages(original_x, reconstructed_x, image_dim, image_channels, file_name)


reconstruction = RECONSTRUCTION()


def scheduler(epoch):
    # initial_lrate = 0.001
    # # if epoch == 0:
    # #     model.optimizer.lr = 0.001 # model.lr.set_value(0.001)
    # if epoch == 25:
    #     model.optimizer.lr = 0.0003 # model.lr.set_value(0.0003)
    # elif epoch == 50:
    #     model.optimizer.lr = 0.0001 # model.lr.set_value(0.0001)
    # return float(model.optimizer.lr) # return model.lr.get_value()
    initial_lrate = 0.001
    if epoch > 25:
        return float(0.0003)
    else:
        return initial_lrate

change_lr = LearningRateScheduler(scheduler)


# model_weights = pickle.load(open('simple_autoencoder' + str(latent_dim_y) + 'd_trained_on_' + dataset_name, 'rb'))
# model.set_weights(model_weights)


model.fit([x_train_1], [x_train_1],
# model.fit([x_train_2], [x_train_2],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data =([x_val_1], [x_val_1]),
        # validation_data =([x_val_2], [x_val_2]),
        callbacks = [change_lr, reconstruction])

model_weights = model.get_weights()
pickle.dump((model_weights), open('simple_autoencoder' + str(latent_dim_y) + 'd_trained_on_' + dataset_name, 'wb'))
















