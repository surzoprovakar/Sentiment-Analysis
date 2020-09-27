from keras.models import model_from_json
import zipfile
import os
import sys
import numpy as np
import shutil
import tensorflow as tf
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, LSTM, Convolution1D, Flatten
from keras.layers import Dropout, GlobalAveragePooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import optimizers
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, EarlyStopping

print(sys.argv[2])
with zipfile.ZipFile(sys.argv[1],"r") as zip_ref:
    zip_ref.extractall(".")
os.rename("Temp/processed_data.npy","Temp/Test.npy")
data = np.load('Temp/Test.npy')
print(data.shape)
data=data.item()

reviews_feats = data['features']
ratings = data['ratings']

max_review_length = 500
X = sequence.pad_sequences(reviews_feats, maxlen=max_review_length)
ratings = np.array(ratings)

X_test = X
y_test = ratings


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(sys.argv[2])
print("Loaded model from disk")

loaded_model.compile(loss='mse',
              optimizer='adam',
              metrics = ['mse'])


results = loaded_model.evaluate(X_test, y_test, verbose=1)

print('Test RMSE: {}'.format(results[0]**0.5))
if os.path.isdir('Temp'):
    shutil.rmtree('Temp')
