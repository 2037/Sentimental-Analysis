# import utils and set plt settings
import nlp_proj_utils as utils
import matplotlib.pyplot as plt
import numpy as np
import collections
import sklearn as sk
from numpy.random import choice
import pickle

# import tensorflow as tf
# print(tf.__version__)
# print("Number of available GPUs: ", tf.config.list_physical_devices("GPU"))

## Still Emoji
train_x, test_x, train_y, test_y = utils.load_emoji()


with open('testset/test_x.pkl','wb') as f:
  pickle.dump(test_x,f)

print("Train: ", train_x.shape, train_y.shape)
print("Test: ", test_x.shape, test_y.shape)
print(train_x[0:3], train_y[0:3])
print(test_x[0:3], test_y[0:3])
# fix ending with \t
old_test_x = test_x
test_x = []
for i in old_test_x:
  test_x.append(str.replace(i,'\t',''))
test_x = np.array(test_x)
print(test_x[0:3], test_y[0:3])
test_x[0:3]
import emoji
print(emoji.emojize('Python is :thumbs_up:'))
# idx = 0
for idx in range(5):
  print(train_x[idx], ": ", utils.label_to_emoji(train_y[idx]))
  print(test_x[idx], ": ", utils.label_to_emoji(test_y[idx]))
counters = collections.Counter(train_y)
counters = sorted(counters.items(), key=lambda x: x[0])
counters
for label, count in counters:
    print("Label: {} -> Emoji: {}".format(label, utils.label_to_emoji(label)))
counters = collections.Counter(train_y)
counters = sorted(counters.items(), key=lambda x: x[1], reverse=True)
counters
for label, count in counters:
    print("Label: {} -> Emoji: {}".format(label, utils.label_to_emoji(label)))

# Google: Skip-gram, CBOW (first word2vec model)
# Stanford: Glove
# Facebook: FastText (Skip-gram/CBOW + Char-CNN)
word_to_index, word_to_vec = utils.load_glove_vecs()
print(len(word_to_index), len(word_to_vec))
word = 'apple'
# word = 'car'
vector = word_to_vec[word]
print("Id: ", word_to_index[word])
print("Embedding: ", vector)
print("Embedding size: ", vector.shape)
print(word_to_vec[word])
print(word_to_vec[word].shape)
print(word_to_vec[word].reshape(1, -1))
print(word_to_vec[word].reshape(1, -1).shape)
from sklearn import metrics

def cosine_similarity(word1, word2):
    vectorize = lambda word: word_to_vec[word].reshape(1, -1)
    return metrics.pairwise.cosine_similarity(vectorize(word1), vectorize(word2)).ravel()[0]

import math
def cosine_similarity_np(word1, word2):
    vec1 = word_to_vec[word1]
    vec2 = word_to_vec[word2]
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# v1=[1, 2] v2=[3, 6] ||v1||2 < ||v2||2  cosine(v1, v2) = 0

print(cosine_similarity('engineer', 'engineering'))
print(cosine_similarity_np('engineer', 'engineering'))
cosine_similarity('engineer', 'engineer')
cosine_similarity('engineer', 'banana')
cosine_similarity('apple', 'banana')
cosine_similarity('apple', 'mac')
cosine_similarity('car', 'vehicle')
### Max sequence length
def sentences_to_indices(X, word_to_index, max_len, oov=0):
    """
    Return a array of indices of a given sentence. The sentence will be trimed/padded to max_len

    Args:
        X (np.ndarray): Input array of sentences, the shape is (m,)  where m is the number of sentences, each sentence is a str. 
        Example X: array(['Sentence 1', 'Setence 2'])
        word_to_index (dict[str->int]): map from a word to its index in vocabulary

    Return:
        indices (np.ndarray): the shape is (m, max_len) where m is the number of sentences
    """
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words[:max_len]:
            X_indices[i, j] = word_to_index.get(w, oov)
            j = j + 1
    return X_indices
MAX_SENTENCE_LEN = max([len(sentence.split()) for sentence in train_x])
print(MAX_SENTENCE_LEN)
sentences_to_indices(np.array(['food is awesome']), word_to_index, max_len=MAX_SENTENCE_LEN)
train_x_ids = sentences_to_indices(train_x, word_to_index, max_len=MAX_SENTENCE_LEN)
test_x_ids = sentences_to_indices(test_x, word_to_index, max_len=MAX_SENTENCE_LEN)
print(train_x_ids.shape, '\n', test_x_ids.shape)
train_x_ids[0:3]
NUM_LABELS = 5
train_y_onehot = utils.convert_to_one_hot(train_y, NUM_LABELS)
test_y_onehot = utils.convert_to_one_hot(test_y, NUM_LABELS)
print(train_y_onehot[:3])
print(test_y_onehot[:3])
print(train_y_onehot.shape, test_y_onehot.shape)
### Word Embeddings & One Hot
### Embedding Layer

# Keras requires vocab length: 1 + actual vocab
# Index = 0 -> padding
vocab_len = 1 + len(word_to_index)
emb_dim = 50

# Create our embedding matrix
emb_matrix = np.zeros([vocab_len, emb_dim])
for word, index in word_to_index.items():
    emb_matrix[index, :] = word_to_vec[word]
emb_matrix.shape
emb_matrix[0:3]
idx = 151204
emb_matrix[idx]
index_to_word = {v: k for k, v in word_to_index.items()}
word = index_to_word[idx]
print(word)
word_to_vec[word]

### Build the Model - NB, LR, RF, GBT, XGBoost
train_x[0:3], test_x[0:3]
train_y[0:3], test_y[0:3]
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer(stop_words='english', lowercase=False)
train_features = tfidf_vec.fit_transform(train_x)
test_features = tfidf_vec.transform(test_x)

with open("models/test_features.pkl","wb") as f:
  pickle.dump(test_features,f)
with open("models/tfidf_vec.pkl","wb") as f:
  pickle.dump(tfidf_vec,f)
test_x[:5]
#### NB
from sklearn.naive_bayes import MultinomialNB

mnb_model = MultinomialNB()
mnb_model.fit(train_features, train_y)

pred = mnb_model.predict(test_features)
print('Accuracy: %f' % metrics.accuracy_score(pred, test_y))
test_idx = choice(test_x_ids.shape[0])
pred = mnb_model.predict(test_features[test_idx])[0]
print("Naive Bayes")
print("Text: ",test_x[test_idx])
print("True Value: ", utils.label_to_emoji(test_y[test_idx]))
print("Prediction: ", utils.label_to_emoji(pred))
# customize input
your_words = 'so bad'
input = tfidf_vec.transform(np.array([your_words]))
pred = mnb_model.predict(input)[0]

print("Naive Bayes")
print("Text: ",your_words)
print("Prediction: ", utils.label_to_emoji(pred))
#### LR
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(train_features, train_y)

pred = lr_model.predict(test_features)
print('Accuracy: %f' % metrics.accuracy_score(pred, test_y))
test_idx = choice(test_x_ids.shape[0])
pred = lr_model.predict(test_features[test_idx])[0]
print("Logistic Regression")
print("Text: ",test_x[test_idx])
print("True Value: ", utils.label_to_emoji(test_y[test_idx]))
print("Prediction: ", utils.label_to_emoji(pred))
#### RF
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 1000)  #10, 100, 1000
rf_model.fit(train_features, train_y)

pred = rf_model.predict(test_features)
print('Accuracy: %f' % metrics.accuracy_score(pred, test_y))
test_idx = choice(test_x_ids.shape[0])
pred = rf_model.predict(test_features[test_idx])[0]
print("Random Forest Classifier")
print("Text: ",test_x[test_idx])
print("True Value: ", utils.label_to_emoji(test_y[test_idx]))
print("Prediction: ", utils.label_to_emoji(pred))
#### GBC
from sklearn.ensemble import GradientBoostingClassifier

gbc_model = GradientBoostingClassifier(n_estimators = 1000, verbose=1)  #10, 100, 1000
gbc_model.fit(train_features, train_y)

pred = gbc_model.predict(test_features)
print('Accuracy: %f' % metrics.accuracy_score(pred, test_y))
test_idx = choice(test_x_ids.shape[0])
pred = gbc_model.predict(test_features[test_idx])[0]
print("Gradient Boosting Classifier")
print("Text: ",test_x[test_idx])
print("True Value: ", utils.label_to_emoji(test_y[test_idx]))
print("Prediction: ", utils.label_to_emoji(pred))
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators = 10)
xgb_model.fit(train_features, train_y)

pred = xgb_model.predict(test_features)
print('Accuracy: %f' % metrics.accuracy_score(pred,test_y))
test_idx = choice(test_x_ids.shape[0])
pred = xgb_model.predict(test_features[test_idx])[0]
print("XGBClassifier")
print("Text: ",test_x[test_idx])
print("True Value: ", utils.label_to_emoji(test_y[test_idx]))
print("Prediction: ", utils.label_to_emoji(pred))
## save models
# save all models above
import pickle

model_names = [
("model_mnb",mnb_model),
("model_lr" ,lr_model),
("model_rf" ,rf_model),
("model_gbc",gbc_model),
("model_xgb",xgb_model)
]

for name,model in model_names:
  with open(f'models/{name}.pkl','wb') as f:
    print(model)
    pickle.dump(model,f)

# "models/model_mnb.pkl"
# "models/model_lr.pkl"
# "models/model_rf.pkl"
# "models/model_gbc.pkl"
# "models/model_xgb.pkl"



# tensorflow -----disable

# from tensorflow.keras.layers import Input, Dropout, LSTM, Embedding, Dense
# from tensorflow.keras.models import Model

# # Create embedding layer
# # Transfer learning: Learn from a pretrained embedding model
# # embedding = Embedding(
# #     input_dim=emb_matrix.shape[0], # 400001
# #     output_dim=emb_matrix.shape[1], # 50
# #     weights=[emb_matrix],
# #     trainable=False
# # )

# # 1. set trainable = False. freeze the pretrained embedding
# # 2. set trainable = True. co-train the pretrained embedding with other layers (preferred)
# # 3. Train embedding from scratch
# embedding = Embedding(
#     input_dim=emb_matrix.shape[0], # 400001
#     output_dim=emb_matrix.shape[1], # 50
# )
# print(train_x_ids.shape)
# print(train_x_ids.shape[1:])
# print(train_x_ids.dtype)
# # Build the model
# # train_x_ids: [# of examples, seq_length]
# # Define the batch_size in training loop
# input_ids = Input(shape=train_x_ids.shape[1:], dtype=train_x_ids.dtype)
# input_embedding = embedding(input_ids)

# # First LSTM layer
# # out = LSTM(units=128, return_sequences=True, recurrent_dropout=0.1)(input_embedding)
# # # Second LSTM layer
# # out = LSTM(units=128, return_sequences=False)(out)
# # out = Dense(units=32, activation='sigmoid')(out)
# # out = Dropout(rate=0.1)(out)
# # out = Dense(NUM_LABELS, activation='softmax')(out)

# # Only one LSTM layer, relu
# out = LSTM(units=128, return_sequences=False, recurrent_dropout=0.1)(input_embedding)
# out = Dense(units=32, activation='relu')(out)
# out = Dropout(rate=0.1)(out)
# out = Dense(NUM_LABELS, activation='softmax')(out)

# ## Only one LSTM layer, sigmoid
# # out = LSTM(units=128, return_sequences=False, recurrent_dropout=0.1)(input_embedding)
# # out = Dense(units=32, activation='sigmoid')(out)
# # out = Dropout(rate=0.1)(out)
# # out = Dense(NUM_LABELS, activation='softmax')(out)


# model = Model(inputs=[input_ids], outputs=out, name='emoji_lstm')
# model.summary()
# model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['acc'])
# # opt = tf.keras.optimizers.Adam(learning_rate=0.1)
# # model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt, metrics=['acc'])
# model
# from tensorflow.python.keras import backend as K
# print(K._get_available_gpus())
# history = model.fit(
#     x=train_x_ids,
#     y=train_y_onehot,
#     validation_data=(test_x_ids, test_y_onehot),
#     batch_size=32, # 2^n
#     epochs=100
# )
# model_dir = 'models/emoji_lstm_glove_pretrain_freeze'
# model.save(model_dir)