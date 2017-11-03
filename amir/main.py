
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn import mixture

from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D,UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import objectives , utils,optimizers
from keras.datasets import mnist
from mpl_toolkits.mplot3d import Axes3D
from keras.callbacks import Callback
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
y_val = np.tile(y_val,(1,2))
y_train = y_train[:training_size,:]
y_train = np.tile(y_train,(1,2))

x_val_2 = x_train_2[training_size:,:]
x_train_2 =x_train_2[:training_size,:]

x_test_2 = x_test_2[:10000,:]
y_test_2 = y_test_2[:10000,:]



batch_size = 100
latent_dim = 15
latent_dim_y = 10
epochs = 1000
intermediate_dim = 500
epsilon_std = 1.0
learning_rate = 0.01
original_dim_1 = 784
original_dim_2  = 32*32*3

# -----------------------------------------------------------------------------
#                                                                   Build Model
# -----------------------------------------------------------------------------

dataset_name = dataset_name_1 + dataset_name_2

experiment_name = dataset_name + \
  '_____z_dim_' + str(latent_dim)

  # if ~ os.path.isdir('../experiments'):
  #   os.makedirs('../experiments')
experiment_dir_path = '../experiments/exp' + \
  '_____' + \
  str(datetime.now().strftime('%Y-%m-%d_____%H-%M-%S')) + \
  '_____' + \
  experiment_name
os.makedirs(experiment_dir_path)


########## Meta ########################################################

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

########## Autoencoder 1 Network ########################################################

x_1 = Input(batch_shape=(batch_size, original_dim_1))
x_reshaped_1 = Reshape((28,28,1))
h_e_1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')
h_e_1_2 = MaxPooling2D((2, 2), padding='same')
h_e_1_3 = Conv2D(16, (3, 3), activation='relu', padding='same')
h_e_1_4 = MaxPooling2D((2, 2), padding='same')
h_e_1_5 = Conv2D(8, (3, 3), activation='relu', padding='same')
h_e_1_6 = MaxPooling2D((2, 2), padding='same')
h_e_1_7 = Flatten()

z_mean_1 = Dense(latent_dim)
z_log_var_1 = Dense(latent_dim)

h_d_x_1_1 = Dense(4*4*8, activation = 'relu')
h_d_x_1_2 = Reshape((4,4,8))
h_d_x_1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')
h_d_x_1_4 = UpSampling2D((2, 2))
h_d_x_1_5 = Conv2D(16, (3, 3), activation='relu', padding='same')
h_d_x_1_6 = UpSampling2D((2, 2))
h_d_x_1_7 = Conv2D(16, (3, 3), activation='relu')
h_d_x_1_8 = UpSampling2D((2, 2))
x_decoded_reshaped_1 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')
x_decoded_1 = Flatten()



###### Autoencoder 2 Network ###########################################################

x_2 = Input(batch_shape=(batch_size, original_dim_2))
x_reshaped_2 = Reshape((32,32,3))
h_e_2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')
h_e_2_2 = MaxPooling2D((2, 2), padding='same')
h_e_2_3 = Conv2D(64, (3, 3), activation='relu', padding='same')
h_e_2_4 = MaxPooling2D((2, 2), padding='same')
h_e_2_5 = Conv2D(32, (3, 3), activation='relu', padding='same')
h_e_2_6 = MaxPooling2D((2, 2), padding='same')
h_e_2_7 = Flatten()

z_mean_2 = Dense(latent_dim)
z_log_var_2 = Dense(latent_dim)

h_d_x_2_1 = Dense(4*4*8, activation = 'relu')
h_d_x_2_2 = Reshape((4,4,8))
h_d_x_2_3 = Conv2D(32, (3, 3), activation='relu', padding='same')
h_d_x_2_4 = UpSampling2D((2, 2))
h_d_x_2_5 = Conv2D(64, (3, 3), activation='relu', padding='same')
h_d_x_2_6 = UpSampling2D((2, 2))
h_d_x_2_7 = Conv2D(64, (3, 3), activation='relu', padding='same')
h_d_x_2_8 = UpSampling2D((2, 2))
x_decoded_reshaped_2 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')
x_decoded_2 = Flatten()



## Classifier Network ##############################################################

def build_z(args):
    z_1 ,z_2 = args
    return tf.concat([z_1[:,:latent_dim_y],z_2[:,:latent_dim_y]],1)

z_y_reshape = Reshape((2,10))

h_d_y_1 = Dense(intermediate_dim, activation='relu')
h_d_y_2 = Dense(intermediate_dim, activation='relu')
h_d_y_3 = Dense(intermediate_dim, activation='relu')
y_decoded_reshaped = Dense(10, activation='softmax')
y_decoded = Reshape((20,))

yy = Input(batch_shape = (batch_size,20))

##### Build model 1 #########################################################################################
_x_reshaped_1 = x_reshaped_1(x_1)
_h_e_1_1 = h_e_1_1(_x_reshaped_1)
_h_e_1_2 = h_e_1_2(_h_e_1_1)
_h_e_1_3 = h_e_1_3(_h_e_1_2)
_h_e_1_4 = h_e_1_4(_h_e_1_3)
_h_e_1_5 = h_e_1_5(_h_e_1_4)
_h_e_1_6 = h_e_1_6(_h_e_1_5)
_h_e_1_7 = h_e_1_7(_h_e_1_6)

_z_mean_1 = z_mean_1(_h_e_1_7)
_z_log_var_1 = z_log_var_1(_h_e_1_7)
z_1 = Lambda(sampling, output_shape=(latent_dim,))([_z_mean_1, _z_log_var_1])

_h_d_x_1_1 = h_d_x_1_1(z_1)
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

_z_mean_2 = z_mean_2(_h_e_2_7)
_z_log_var_2 = z_log_var_2(_h_e_2_7)
z_2 = Lambda(sampling, output_shape=(latent_dim,))([_z_mean_2, _z_log_var_2])

_h_d_x_2_1 = h_d_x_2_1(z_2)
_h_d_x_2_2 = h_d_x_2_2(_h_d_x_2_1)
_h_d_x_2_3 = h_d_x_2_3(_h_d_x_2_2)
_h_d_x_2_4 = h_d_x_2_4(_h_d_x_2_3)
_h_d_x_2_5 = h_d_x_2_5(_h_d_x_2_4)
_h_d_x_2_6 = h_d_x_2_6(_h_d_x_2_5)
_h_d_x_2_7 = h_d_x_2_7(_h_d_x_2_6)
_h_d_x_2_8 = h_d_x_2_8(_h_d_x_2_7)
_x_decoded_reshaped_2 = x_decoded_reshaped_2(_h_d_x_2_8)
_x_decoded_2 = x_decoded_2(_x_decoded_reshaped_2)



##### Build Classifier #################################################################################

_z_y = Lambda(build_z)([z_1, z_2])

_z_y_reshape = z_y_reshape(_z_y)

_h_d_y_1 = h_d_y_1(_z_y_reshape)
_h_d_y_2 = h_d_y_2(_h_d_y_1)
_h_d_y_3 = h_d_y_3(_h_d_y_2)
_y_decoded_reshaped = y_decoded_reshaped(_h_d_y_3)
_y_decoded = y_decoded(_y_decoded_reshaped)

###### Define Loss ###########################################################################

model = Model(inputs = [x_1,x_2,yy],outputs = [_x_decoded_1,_x_decoded_2,_y_decoded])

def vae_loss(x, _x_decoded):

    xent_loss_1 = original_dim_1 * objectives.binary_crossentropy(x_1, _x_decoded_1)
    xent_loss_2 = original_dim_2 * objectives.binary_crossentropy(x_2, _x_decoded_2)
    kl_loss_1 = - 0.5 * K.sum(1 + _z_log_var_1 - K.square(_z_mean_1) - K.exp(_z_log_var_1), axis=-1)
    kl_loss_2 = - 0.5 * K.sum(1 + _z_log_var_2 - K.square(_z_mean_2) - K.exp(_z_log_var_2), axis=-1)
    y_loss= 20 * objectives.categorical_crossentropy(yy, _y_decoded)
    return xent_loss_1 + xent_loss_2 + kl_loss_1 + kl_loss_2 + y_loss

my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.1)

model.compile(optimizer=my_adam, loss=vae_loss)

############################################################################
############################################################################
#### Build another model####################################################

_x_reshaped_1_ = x_reshaped_1(x_1)
_h_e_1_1_ = h_e_1_1(_x_reshaped_1_)
_h_e_1_2_ = h_e_1_2(_h_e_1_1_)
_h_e_1_3_ = h_e_1_3(_h_e_1_2_)
_h_e_1_4_ = h_e_1_4(_h_e_1_3_)
_h_e_1_5_ = h_e_1_5(_h_e_1_4_)
_h_e_1_6_ = h_e_1_6(_h_e_1_5_)
_h_e_1_7_ = h_e_1_7(_h_e_1_6_)

_z_mean_1_ = z_mean_1(_h_e_1_7_)

_h_d_x_1_1_ = h_d_x_1_1(_z_mean_1_)
_h_d_x_1_2_ = h_d_x_1_2(_h_d_x_1_1_)
_h_d_x_1_3_ = h_d_x_1_3(_h_d_x_1_2_)
_h_d_x_1_4_ = h_d_x_1_4(_h_d_x_1_3_)
_h_d_x_1_5_ = h_d_x_1_5(_h_d_x_1_4_)
_h_d_x_1_6_ = h_d_x_1_6(_h_d_x_1_5_)
_h_d_x_1_7_ = h_d_x_1_7(_h_d_x_1_6_)
_h_d_x_1_8_ = h_d_x_1_8(_h_d_x_1_7_)
_x_decoded_reshaped_1_ = x_decoded_reshaped_1(_h_d_x_1_8_)
_x_decoded_1_ = x_decoded_1(_x_decoded_reshaped_1_)

#####################################

_x_reshaped_2_ = x_reshaped_2(x_2)
_h_e_2_1_ = h_e_2_1(_x_reshaped_2_)
_h_e_2_2_ = h_e_2_2(_h_e_2_1_)
_h_e_2_3_ = h_e_2_3(_h_e_2_2_)
_h_e_2_4_ = h_e_2_4(_h_e_2_3_)
_h_e_2_5_ = h_e_2_5(_h_e_2_4_)
_h_e_2_6_ = h_e_2_6(_h_e_2_5_)
_h_e_2_7_ = h_e_2_7(_h_e_2_6_)

_z_mean_2_ = z_mean_2(_h_e_2_7_)

_h_d_x_2_1_ = h_d_x_2_1(_z_mean_2_)
_h_d_x_2_2_ = h_d_x_2_2(_h_d_x_2_1_)
_h_d_x_2_3_ = h_d_x_2_3(_h_d_x_2_2_)
_h_d_x_2_4_ = h_d_x_2_4(_h_d_x_2_3_)
_h_d_x_2_5_ = h_d_x_2_5(_h_d_x_2_4_)
_h_d_x_2_6_ = h_d_x_2_6(_h_d_x_2_5_)
_h_d_x_2_7_ = h_d_x_2_7(_h_d_x_2_6_)
_h_d_x_2_8_ = h_d_x_2_8(_h_d_x_2_7_)
_x_decoded_reshaped_2_ = x_decoded_reshaped_2(_h_d_x_2_8_)
_x_decoded_2_ = x_decoded_2(_x_decoded_reshaped_2_)

#######################################################

_z_y_ = Lambda(build_z)([_z_mean_1_, _z_mean_2_])


_z_y_reshape_ = z_y_reshape(_z_y_)


_h_d_y_1_ = h_d_y_1(_z_y_reshape_)
_h_d_y_2_ = h_d_y_2(_h_d_y_1_)
_h_d_y_3_ = h_d_y_3(_h_d_y_2_)
_y_decoded_reshaped_ = y_decoded_reshaped(_h_d_y_3_)
_y_decoded_ = y_decoded(_y_decoded_reshaped_)


vaeencoder = Model(inputs = [x_1, x_2], outputs = [_x_decoded_1_, _x_decoded_2_, _y_decoded_])

############################################################################
############################################################################
############################################################################
# -----------------------------------------------------------------------------
#                                                                   Train Model
## -----------------------------------------------------------------------------
_, _, b = vaeencoder.predict([x_test_1,x_test_2],batch_size = batch_size)

b = np.reshape(b,(x_test_1.shape[0],2,10))

y_test_label_1 = np.argmax(y_test_1,axis =1)
y_test_label_2 = np.argmax(y_test_2,axis =1)

#y_test_label = np.reshape(y_test_label,(y_test_label.shape[0],1))
#y_test_label = np.reshape(y_test_label,(y_test_label.shape[0],1))
y_test_label = np.concatenate((y_test_label_1,y_test_label_2),axis = 0)


Accuracy = np.zeros((epochs,1))
ii=0
pickle.dump((ii),open('counter','wb'))
text_file_name = experiment_dir_path + '/accuracy_log.txt'
class ACCURACY(Callback):

    def on_epoch_end(self,batch,logs = {}):
        ii= pickle.load(open('counter', 'rb'))
        _,_, b  = vaeencoder.predict([x_test_1,x_test_2], batch_size = batch_size)
        b = np.reshape(b,(x_test_1.shape[0],2,10))

        Accuracy[ii, 0]

        lll = np.argmax(b, axis =2)
        lll = np.reshape(lll,(20000,))
        n_error = np.count_nonzero(lll - y_test_label)
        ACC = 1 - n_error / 20000
        Accuracy[ii,0] = ACC
        print('\n accuracy = ', ACC)
        ii= ii + 1
        pickle.dump((ii),open('counter', 'wb'))
        with open(text_file_name, 'a') as text_file:
          print('Epoch #{} Accuracy:{} \n'.format(ii, ACC), file=text_file)

accuracy = ACCURACY()

class RECONSTRUCTION(Callback):

    def on_epoch_end(self,batch,logs = {}):

        timestamp_string = str(datetime.now().strftime('%Y-%m-%d_____%H-%M-%S'))
        image_file_name_1 = experiment_dir_path + '/' + timestamp_string + '_reconstructed_samples_1.png'
        image_file_name_2 = experiment_dir_path + '/' + timestamp_string + '_reconstructed_samples_2.png'

        reconstructed_x_test_1, reconstructed_x_test_2, _ = vaeencoder.predict([x_test_1,x_test_2], batch_size = batch_size)

        tmp = 4
        plt.figure(figsize=(tmp + 1, tmp + 1))
        for i in range(tmp):
          for j in range(tmp):
            ax = plt.subplot(tmp, tmp, i*tmp+j+1)
            # plt.imshow(x_train[i*tmp+j].reshape(sample_dim, sample_dim, sample_channels))
            plt.imshow(reconstructed_x_test_1[i*tmp+j].reshape(28,28))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(image_file_name_1)

        tmp = 4
        plt.figure(figsize=(tmp + 1, tmp + 1))
        for i in range(tmp):
          for j in range(tmp):
            ax = plt.subplot(tmp, tmp, i*tmp+j+1)
            # plt.imshow(x_train[i*tmp+j].reshape(sample_dim, sample_dim, sample_channels))
            plt.imshow(reconstructed_x_test_2[i*tmp+j].reshape(32,32,3))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(image_file_name_2)

reconstruction = RECONSTRUCTION()

#model_weights = pickle.load(open('weights_vaesdr_' + str(latent_dim) + 'd_trained_on_' + dataset_name, 'rb'))
#model.set_weights(model_weights)

model.fit([x_train_1,x_train_2, y_train],[x_train_1,x_train_2,y_train],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data =([x_val_1,x_val_2,y_val],[x_val_1,x_val_2,y_val]),
        callbacks = [accuracy,reconstruction])

model_weights = model.get_weights()
pickle.dump((model_weights), open('weights_vaesdr_' + str(latent_dim) + 'd_trained_on_' + dataset_name, 'wb'))
############################################################################################################

# -----------------------------------------------------------------------------
#                                                                      Analysis
# -----------------------------------------------------------------------------

###### Builder Encoder ######################################################################
#encoder = Model(x, _z_mean)
#
#x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], linewidth = 0, c=y_test_label)
#
##### build generator #########################################################################
#generator_input = Input(shape=(latent_dim,))
#
#_h_g_x_1_ = h_d_x_1(generator_input)
#_h_g_x_2_ = h_d_x_2(_h_g_x_1_)
#_h_g_x_3_ = h_d_x_3(_h_g_x_2_)
#_h_g_x_4_ = h_d_x_4(_h_g_x_3_)
#_h_g_x_5_ = h_d_x_5(_h_g_x_4_)
#_h_g_x_6_ = h_d_x_6(_h_g_x_5_)
#_h_g_x_7_ = h_d_x_7(_h_g_x_6_)
#_h_g_x_8_ = h_d_x_8(_h_g_x_7_)
#_x_generated_reshaped = x_decoded_reshaped(_h_g_x_8_)
#_x_generated_ = x_decoded(_x_generated_reshaped)
#
#generator = Model(generator_input,_x_generated_)
#                                        # -------------------------------------
#                                        #                               Fit GMM
#                                        # -------------------------------------
#
## display a 2D plot of the digit classes in the latent space
#x_train_encoded = encoder.predict(x_train, batch_size=batch_size)
#
#n_components = num_classes
#cv_type = 'full'
#gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
#gmm.fit(x_train_encoded)
#
#x_decoded, b  = vaeencoder.predict(x_test,batch_size = batch_size)
#
#
#                                        # -------------------------------------
#                                        #                                 Plots
#                                        # -------------------------------------
#
#
#
#def getFigureOfSamplesForInput(x_samples, sample_dim, number_of_sample_images, grid_x, grid_y):
#    figure = np.zeros((sample_dim * number_of_sample_images, sample_dim * number_of_sample_images))
#    for i, yi in enumerate(grid_x):
#        for j, xi in enumerate(grid_y):
#            digit = x_samples[i * number_of_sample_images + j, :].reshape(sample_dim, sample_dim)
#            figure[i * sample_dim: (i + 1) * sample_dim,
#                   j * sample_dim: (j + 1) * sample_dim] = digit
#    return figure
#
#
#number_of_sample_images = 10
#grid_x = norm.ppf(np.linspace(0.05, 0.95, number_of_sample_images))
#grid_y = norm.ppf(np.linspace(0.05, 0.95, number_of_sample_images))
#
#plt.figure()
##
##ax = plt.subplot(1,3,1)
##x_samples_a = x_test
##canvas = getFigureOfSamplesForInput(x_samples_a, sample_dim, number_of_sample_images, grid_x, grid_y)
##plt.imshow(canvas, cmap='Greys_r')
##ax.set_title('Original Test Images', fontsize=8)
##ax.get_xaxis().set_visible(False)
##ax.get_yaxis().set_visible(False)
#
#ax = plt.subplot(1,3,1)
#x_samples_b = x_decoded
#canvas = getFigureOfSamplesForInput(x_samples_b, sample_dim, number_of_sample_images, grid_x, grid_y)
#plt.imshow(canvas, cmap='Greys_r')
#ax.set_title('Reconstructed Test Images', fontsize=8)
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
#
#ax = plt.subplot(1,3,2)
#x_samples_c = gmm.sample(10000)#(number_of_sample_images*number_of_sample_images)
#x_samples_c = np.random.permutation(x_samples_c[0]) # need to randomly permute because gmm.sample samples 1000 from class 1, then 1000 from class 2, etc.
#x_samples_c = generator.predict(x_samples_c)
#canvas = getFigureOfSamplesForInput(x_samples_c, sample_dim, number_of_sample_images, grid_x, grid_y)
#plt.imshow(canvas, cmap='Greys_r')
#ax.set_title('Generated Images', fontsize=8)
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
#
#plt.show()
## plt.savefig('images/'+ dataset_name + '_samples.png')
#
#
#
#x_samples_c = np.zeros((number_of_sample_images*number_of_sample_images,original_dim))
#ax = plt.subplot(1,3,3)
#for i in range(100):
#    KK = int(np.floor(i/10)*10 +1)
#    aux_sample = gmm.sample(100)#(number_of_sample_images*number_of_sample_images)
#    aux_sample = generator.predict(aux_sample[0])
#    x_samples_c[i,:] = aux_sample[KK-1,:]
#canvas = getFigureOfSamplesForInput(x_samples_c, sample_dim, number_of_sample_images, grid_x, grid_y)
#plt.imshow(canvas, cmap='Greys_r')
#ax.set_title('Generated Images', fontsize=8)
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
#
#plt.show()
#
#
#
#
#
#
#
#
#
