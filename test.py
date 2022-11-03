# import utils and set plt settings
import nlp_proj_utils as utils
import numpy as np
import sklearn as sk
from numpy.random import choice
import pickle

#load test set x_ids,y,x_vectors
with open('testset/test_y.pkl','rb') as f:
  test_y = pickle.load(f)

with open('testset/test_x_ids.pkl','rb') as f:
  test_x_ids = pickle.load(f)

with open('testset/test_x.pkl','rb') as f:
  test_x = pickle.load(f)

with open("models/test_features.pkl","rb") as f:
  test_features = pickle.load(f)

with open("models/tfidf_vec.pkl","rb") as f:
  tfidf_vec = pickle.load(f)

# load all models and lstm
model_names = [
  "model_mnb",
  "model_lr",
  "model_rf",
  "model_gbc",
  "model_xgb",
]

models = []
for model in model_names:
  with open(f'models/{model}.pkl','rb') as f:
    models.append(pickle.load(f))

# import tensorflow as tf
# model_dir = 'models/emoji_lstm_glove_pretrain_freeze'
# model_reload = tf.keras.models.load_model(model_dir)

# ==============test set sample prediction==============
# def lstm_pred(test_idx):
#   probs = model_reload.predict(np.reshape(test_x_ids[test_idx], [1, -1]),verbose=0).ravel()
#   pred = np.argmax(probs)
#   return pred
def lstm_pred(test_idx):
  return 0

test_idx = choice(test_x_ids.shape[0])
print("Text: ",test_x[test_idx])
print("True Value: ", utils.label_to_emoji(test_y[test_idx]))
for i,model in enumerate(models):
  pred = model.predict(test_features[test_idx])[0]
  print(model_names[i])
  print("Prediction: ", utils.label_to_emoji(pred))

# lstm
print("lstm")
print("Prediction: ", utils.label_to_emoji(lstm_pred(test_idx)))

# load for lstm algo
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

word_to_index, word_to_vec = utils.load_glove_vecs()


#============== your words ==============
# def lstm_pred_your_words(your_words):
#   MAX_SENTENCE_LEN = 10
#   your_words_id = sentences_to_indices(np.array([your_words]), word_to_index, max_len=MAX_SENTENCE_LEN)
#   probs = model_reload.predict(np.reshape(your_words_id, [1, -1]),verbose = 0).ravel()
#   pred = np.argmax(probs)
#   return pred

def lstm_pred_your_words(your_words):
  return 0


print(f"{'models':+^30}")
[print(str(i)) for i in models]
# print(str(model_reload))

print()
print(f"{'test_cases':+^30}")
your_words_test = ['bad day','good morning','i like you','nice ball','i need food']
labels = [3,2,0,1,4]
preds = []
for i,your_words in enumerate(your_words_test):
  preds = []
  print(f"{your_words:+^30}")
  print("Expected Value:", utils.label_to_emoji(labels[i]))
  for i,model in enumerate(models):
    your_words_vec = tfidf_vec.transform(np.array([your_words]))
    pred = model.predict(your_words_vec)[0]
    preds.append(pred)

  lstm_pred = lstm_pred_your_words(your_words)
  preds.append(lstm_pred)
  print([utils.label_to_emoji(i) for i in preds])

def emoji_prediction(your_words=None,prompt_mode=False):
  if prompt_mode:
    your_words = input("Enter your text:\n")
  if your_words is None or your_words=='':
    print('We need input text')
    return None

  preds = []
  print(f"{your_words:+^30}")
  for i,model in enumerate(models):
    your_words_vec = tfidf_vec.transform(np.array([your_words]))
    pred = model.predict(your_words_vec)[0]
    preds.append(pred)

  lstm_pred = lstm_pred_your_words(your_words)
  preds.append(lstm_pred)
  print([utils.label_to_emoji(i) for i in preds])
  return preds

emoji_prediction()
emoji_prediction()
emoji_prediction()

# print([utils.label_to_emoji(i) for i in preds])
# (emoji_prediction('love u'))
# (emoji_prediction('lets play baseball'))
# (emoji_prediction('sorry to hear that'))
# (emoji_prediction('im so sad'))
# (emoji_prediction('i order some chicken'))