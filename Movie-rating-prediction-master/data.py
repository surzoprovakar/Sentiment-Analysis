# -*- coding: utf-8 -*-

import sys
import shutil
import zipfile
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import os
nltk.download('stopwords')


def load_zip(filepath):
    with zipfile.ZipFile(filepath+"/data.zip","r") as zip_ref:
        zip_ref.extractall(".")

with zipfile.ZipFile(sys.argv[1],"r") as zip_ref:
        zip_ref.extractall(".")
        
    
with open('Movies_and_TV_5.json') as f:
    raw_data = [eval(x) for x in f.readlines()[0:10000]]

#with open('ratings.txt', 'w') as f:
#    f.write('reviewerID,asin,overall\n')
#    for d in raw_data:
#        f.write('{},{},{}\n'.format(d['reviewerID'], d['asin'], d['overall']))
     
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
        
        
def preprocess(data):
    all_words = []
    all_reviews = []
    for r in data:
        r = r['reviewText'].lower()
        r = ''.join([c for c in r if c not in punctuation])
        review = []
        for w in r.split():
            if w not in stop_words:
                all_words.append(w)
                review.append(w)
        review = ' '.join(review)
        all_reviews.append(review)
    return all_reviews, all_words

reviews, all_words = preprocess(raw_data)



lengths = []
for r in reviews:
    lengths.append(len(r.split()))
    

#plt.hist(lengths, bins=100)
#plt.show()


print('total words: ', len(all_words))
words_freq = nltk.FreqDist(all_words)
print('unique words: ', len(words_freq))


vocab_size = 5000
vocab = [x[0] for x in words_freq.most_common(vocab_size)]
vocab_set = set(vocab)
vocab2indx = dict(zip(vocab, range(vocab_size)))


#all_users = set()
#all_items = set()
#for d in raw_data:
#    all_users.add(d['reviewerID'])
#    all_items.add(d['asin'])
#print(len(all_users))
#print(len(all_items))


#user2indx = dict(zip(all_users, range(len(all_users))))
#item2indx = dict(zip(all_items, range(len(all_items))))


#def get_user_item(d):
#    return (user2indx[d['reviewerID']], item2indx[d['asin']])
#user_item = [get_user_item(d) for d in raw_data]
#
#
#np.save('user_item.npy', user_item)


def feature(review):
    feat = []
    for w in review.split():
        if w in vocab_set:
            feat.append(vocab2indx[w])
    return feat

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def save_zip(feats, ratings, path):
    directory = os.path.dirname('Temp/processed_data.npy')
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    np.save('Temp/processed_data.npy', {'features': feats,
                               'ratings': ratings})

    directory = os.path.dirname(path+'/data.zip')
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    zipf = zipfile.ZipFile(path+'/data.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir('Temp', zipf)
    zipf.close()
    
    
reviews_feats = [feature(r) for r in reviews]
ratings = [r['overall'] for r in raw_data]

l = len(reviews_feats)

feat_80 = reviews_feats[0:int(.8*l)]
ratings_80 = ratings[0:int(.8*l)]

feat_10_val = reviews_feats[int(.8*l):int(.9*l)]
ratings_10_val = ratings[int(.8*l):int(.9*l)]

feat_10_test = reviews_feats[int(.9*l):l]
ratings_10_test = ratings[int(.9*l):l]

feat_10_train = reviews_feats[0:int(.2*l)]
ratings_10_train = ratings[0:int(.2*l)]

feat_90_train = reviews_feats[0:int(.6*l)]
ratings_90_train = ratings[0:int(.6*l)]

feat_3_sample_val = feat_10_val[0:3]
ratings_3_sample_val = ratings_10_val[0:3]

#np.save('Temp\processed_data.npy', {'features': reviews_feats,
#                               'ratings': ratings})

train_80_path = 'Data/Train/Best_hyperparameter_80_percent'
val_10_path = 'Data/Validation/Validation_10_Percent'
test_10_path = 'Data/Test/Test_10_Percent'
train_10min_path = 'Data/Train/Under_10_min_training'
tune_90min_path = 'Data/Train/Under_90_min_tuning'
val_3_sample_path = 'Data/Validation/3_samples'
    

save_zip(feat_80, ratings_80, train_80_path)
save_zip(feat_10_val, ratings_10_val, val_10_path)
save_zip(feat_10_test, ratings_10_test, test_10_path)
save_zip(feat_10_train, ratings_10_train, train_10min_path)
save_zip(feat_90_train, ratings_90_train, tune_90min_path)
save_zip(feat_3_sample_val, ratings_3_sample_val, val_3_sample_path)

if os.path.exists('Movies_and_TV_5.json'):
    os.remove('Movies_and_TV_5.json')
if os.path.isdir('Temp'):
    shutil.rmtree('Temp')








