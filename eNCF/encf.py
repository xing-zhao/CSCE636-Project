import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import pickle
from time import time
from keras import backend as K
import argparse
import os, glob
import keras

warnings.filterwarnings('ignore')

#%matplotlib inline

def parse_args():
    parser = argparse.ArgumentParser(description="Run eNCF.")
    parser.add_argument('--dataset', nargs='?', default='Dataset/reviews_CDs_and_Vinyl_5',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs.')
    return parser.parse_args()


args = parse_args()
#dataset = 'reviews_CDs_and_Vinyl_5'
dataset = args.dataset
num_epochs = args.epochs

with open(dataset + '_Train_list.txt', "rb") as fp:
    Train_list = pickle.load(fp)
    
with open(dataset + '_Vali_list.txt', "rb") as fp:
    Vali_list = pickle.load(fp)
    
with open(dataset + '_Test_list.txt', "rb") as fp:
    Test_list = pickle.load(fp)

Train_list = np.array(Train_list, dtype=int)
Vali_list = np.array(Vali_list, dtype=int)
Test_list = np.array(Test_list, dtype=int)

print (np.max(Train_list[:,0]))

n_users = max(np.max(Train_list[:,0]), np.max(Vali_list[:,0]), np.max(Test_list[:,0]))
n_movies = max(np.max(Train_list[:,1]), np.max(Vali_list[:,1]), np.max(Test_list[:,1]))


n_latent_factors_user = 30
n_latent_factors_movie = 30
n_latent_factors_mf = 10
#n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())

movie_input = keras.layers.Input(shape=[1],name='Item-OneHot')
movie_embedding_mlp = keras.layers.Embedding(n_movies + 1, n_latent_factors_movie, name='MLP-Item-Vector')(movie_input)
movie_vec_mlp = keras.layers.Flatten(name='MLP-Item-Flatten')(movie_embedding_mlp)
movie_vec_mlp = keras.layers.Dropout(0.5, name='MLP-Item-Flatten-Dropout')(movie_vec_mlp)

movie_embedding_mf = keras.layers.Embedding(n_movies + 1, n_latent_factors_mf, name='MF-Item-Vector')(movie_input)
movie_vec_mf = keras.layers.Flatten(name='MF-Item-Flatten')(movie_embedding_mf)
movie_vec_mf = keras.layers.Dropout(0.5, name='MF-Item-Flatten-Dropout')(movie_vec_mf)


user_input = keras.layers.Input(shape=[1],name='User-OneHot')
user_vec_mlp = keras.layers.Flatten(name='MLP-User-Flatten')(keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='MLP-User-Vector')(user_input))
user_vec_mlp = keras.layers.Dropout(0.5, name='MLP-User-Flatten-Dropout')(user_vec_mlp)

user_vec_mf = keras.layers.Flatten(name='MF-User-Flatten')(keras.layers.Embedding(n_users + 1, n_latent_factors_mf,name='MF-User-Vector')(user_input))
user_vec_mf = keras.layers.Dropout(0.5, name='MF-User-Flatten-Dropout')(user_vec_mf)


concat = keras.layers.merge([movie_vec_mlp, user_vec_mlp], mode='concat',name='MLP-Concat')
concat_dropout = keras.layers.Dropout(0.5)(concat)
dense = keras.layers.Dense(100,name='Fully-Connected')(concat_dropout)
dense_batch = keras.layers.BatchNormalization(name='Batch')(dense)
dropout_1 = keras.layers.Dropout(0.5,name='Dropout-1')(dense_batch)
dense_2 = keras.layers.Dense(100,name='Fully-Connected-1')(dropout_1)
dense_batch_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)


dropout_2 = keras.layers.Dropout(0.5,name='Dropout-2')(dense_batch_2)
dense_3 = keras.layers.Dense(100,name='Fully-Connected-2')(dropout_2)
dense_4 = keras.layers.Dense(50,name='Fully-Connected-3', activation='relu')(dense_3)

pred_mf = keras.layers.merge([movie_vec_mf, user_vec_mf], mode='mul',name='Dot')


pred_mlp = keras.layers.Dense(30, activation='relu',name='Activation')(dense_4)

combine_mlp_mf = keras.layers.merge([pred_mf, pred_mlp], mode='concat',name='Concat-MF-MLP')
result_combine = keras.layers.Dense(100,name='Combine-MF-MLP')(combine_mlp_mf)
deep_combine = keras.layers.Dense(100,name='Fully-Connected-4')(result_combine)


result = keras.layers.Dense(1,name='Predicted-Rating')(deep_combine)


model = keras.Model([user_input, movie_input], result)
opt = keras.optimizers.Adam(lr =0.01)
model.compile(optimizer=opt,loss= 'mean_squared_error',metrics=['mae'])



best_loss = float("inf")
best_mae = float("inf")

train_loss_values = []
vali_loss_values = []
train_mae_values = []
vali_mae_values = []

for epoch in range(num_epochs):
    print ('Iteration ' + str(epoch))
    t1 = time()
    history = model.fit([Train_list[:,0], Train_list[:,1]], Train_list[:,2], 
                        verbose=0, 
                        batch_size = 1024)
    train_loss = history.history['loss'][0]
    train_loss_values.append(train_loss)
    train_mae = history.history['mean_absolute_error'][0]
    train_mae_values.append(train_mae)
    #print train_loss, train_mae
    t2 = time()
    print ('\tTraining [%.1f s]: MSE = %.4f, MAE = %.4f' % (t2-t1, train_loss, train_mae))
    
    loss, mae = model.evaluate([Vali_list[:,0], Vali_list[:,1]], Vali_list[:,2])
    vali_loss_values.append(loss)
    vali_mae_values.append(mae)
    
    print('\tValidation [%.1f s]: MSE = %.4f, MAE = %.4f ' % (time()-t2, loss, mae))
    if loss < best_loss:
        best_mae, best_loss, best_iter = mae, loss, epoch
        best_model_name = '%s_eNCF_%d.h5' %(dataset,time())
        model.save_weights(best_model_name, overwrite=True)
        

print("End. Best Iteration %d:  MSE = %.4f, MAE = %.4f. " %(best_iter, best_loss, best_mae))


print ('\nSave tringing history figures')
epoch = range(1, len(train_loss_values) + 1)

plt.plot(epoch, train_loss_values, 'bo', label = 'Training MSE')
plt.plot(epoch, vali_loss_values, 'b', label = 'Validation MSE')
plt.title('Training and Validation MSE')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend()


plt.tight_layout()
plt.savefig('Project_4_Training_and_Validation_MSE.jpg', dpi = 1000)

plt.clf()

epoch = range(1, len(train_loss_values) + 1)

plt.plot(epoch, train_mae_values, 'bo', label = 'Training MAE')
plt.plot(epoch, vali_mae_values, 'b', label = 'Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('epoch')
plt.ylabel('mae')
plt.legend()

plt.tight_layout()
plt.savefig('Project_4_Training_and_Validation_MAE.jpg', dpi = 1000)


print ('\nTesting on Best Model:')
model.load_weights(best_model_name)

test_mse, test_mae = model.evaluate([Vali_list[:,0], Vali_list[:,1]], Vali_list[:,2])

print ("Testing Result: MSE: %.4f, MAE: %.4f" % (test_mse, test_mae))

for file in glob.glob("*.h5"):
    os.remove(file)



























