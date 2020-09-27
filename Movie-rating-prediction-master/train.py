import sys
import os
import zipfile
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, LSTM, Convolution1D, Flatten
from keras.layers import Dropout, GlobalAveragePooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import optimizers
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, EarlyStopping



def model_covNet(layer):
    
    vocab_size = 5000
    embedding_size = 32
    max_review_length = 500
    
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
                        input_length=max_review_length))
    
    for i in layer:
        model.add(Convolution1D(i, 3, activation='relu', padding='same'))
        model.add(Convolution1D(i, 3, activation='relu', padding='same'))
        if i!=layer[-1]:
            model.add(MaxPooling1D(3))
        
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation=None))
    
    return model


def model_mlp(layer):
    vocab_size = 5000
    embedding_size = 32
    max_review_length = 500
    
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
                        input_length=max_review_length))
    model.add(Flatten())
    for i in layer:
       model.add(Dense(i, activation='relu'))
    model.add(Dense(1, activation=None))
#    
#    optim = optimizers.Adam(lr=0.001,
#                            decay=0.001)
    return model


def model_lstm(output):
    vocab_size = 5000
    embedding_size = 32
    max_review_length = 500
    
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
                        input_length=max_review_length))
    model.add(LSTM(output))
    model.add(Dense(1, activation=None))
    
#    optim = optimizers.Adam(lr=0.001,
#                            decay=0.001)
    return model


def training(model, output, isTrain):
    global np, sequence, Sequential, optimizers, TensorBoard, EarlyStopping
        
    print('Loading data...')
    if(isTrain==0):
        data = np.load('Temp/Under_90_min_tuning.npy')
    else:
        data = np.load('Temp/Train_80_Percent.npy')
    print(data.shape)
    data=data.item()
    
    reviews_feats = data['features']
    ratings = data['ratings']
    
    max_review_length = 500
    X = sequence.pad_sequences(reviews_feats, maxlen=max_review_length)
    ratings = np.array(ratings)

    X_train = X
    y_train = ratings
    
    
    
    data = np.load('Temp/Validation_10_Percent.npy')
    print(data.shape)
    data=data.item()
    
    reviews_feats = data['features']
    ratings = data['ratings']
    
    max_review_length = 500
    X = sequence.pad_sequences(reviews_feats, maxlen=max_review_length)
    ratings = np.array(ratings)
     
    X_test = X
    y_test = ratings

    
    
#    m,n = X.shape
#    print('Total data size: {}'.format((m,n)))
    
    # split data into training, validation and test set
    '''
    train_idx = np.load('train_idx.npy')
    test_idx = np.load('test_idx.npy')
    
    X_train = X[train_idx]
    y_train = ratings[train_idx]
    X_test = X[test_idx]
    y_test = ratings[test_idx]
    val_ratio = 0.1
    '''
    
    
    print('Training data size: {}'.format(X_train.shape))
    print('Validation data size: {}'.format(X_test.shape))
    #print('Validation ratio: {} % of training data'.format(val_ratio*100))
    
    if model == "lstm":
        model = model_lstm(output)
    elif model == "mlp":
        model = model_mlp(output)
    elif model == "covNet":
        model = model_covNet(output)
        
    
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics = ['mse'])
    
    
#    tensorboard = TensorBoard(log_dir='./logs', write_graph=True)
    earlystopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=2,
                                  verbose=0,
                                  mode='auto')
    
    model.fit(X_train, y_train,
              batch_size=64,
              epochs=10,
              callbacks=[earlystopping],
              shuffle=True,
              verbose=1)
    
    if isTrain==1:
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
            
        model.save_weights("model.h5")
        print("Saved model to disk")
    
    results = model.evaluate(X_test, y_test, verbose=0)
    print('Test RMSE: {}'.format(results[0]**0.5))
    
    return results[0]**0.5
    
    

if (len(sys.argv) > 1 and sys.argv[2]==".\hyperparameter.txt"):
    if os.path.exists('Temp/Train_80_Percent.npy'):
        os.remove("Temp/Train_80_Percent.npy")
    with zipfile.ZipFile(sys.argv[1],"r") as zip_ref:
        zip_ref.extractall(".")
    os.rename("Temp/processed_data.npy","Temp/Train_80_Percent.npy")
    f = open(sys.argv[2], 'r')
    arguments = f.readline().split(", ")
    model = arguments[0]
    print(model)
    if model == "lstm":
        output = arguments[1]
        rmse = training(model, int(output), 1)
    else:
        output = []
        for i in range(1, len(arguments)-1):
            layer = arguments[i].replace("[", "")
            layer = layer.replace("]", "")
            output.append(int(layer))
        rmse = training(model, output, 1)
    print(model, rmse)
    f.close()



