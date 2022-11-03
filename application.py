# from app import route
from flask import Flask, render_template

# import utils and set plt settings
import nlp_proj_utils as utils
import numpy as np
import sklearn as sk
from numpy.random import choice
import pickle
from flask import request
import pandas as pd

#==========load model==========
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

# ============== real time prediction ==============
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

# def lstm_pred_text(your_words):
# 	MAX_SENTENCE_LEN = 10
# 	if(len(your_words.split(' '))>10):
# 		return None
# 	your_words_id = sentences_to_indices(np.array([your_words]), word_to_index, max_len=MAX_SENTENCE_LEN)
# 	probs = model_reload.predict(np.reshape(your_words_id, [1, -1]),verbose = 0).ravel()
# 	pred = np.argmax(probs)
# 	return pred


def predict_text(text):
	preds = []
	preds.append(text)
	preds.append("Unknown")
	for i,model in enumerate(models):
		vec_text = tfidf_vec.transform(np.array([text]))
		pred = model.predict(vec_text)[0]
		pred = utils.label_to_emoji(pred)
		preds.append(pred)
	#lstm
	# pred=lstm_pred_text(text)
	pred=0
	pred=utils.label_to_emoji(pred)

	# text, true value, models
	preds.append(pred)
	df = pd.DataFrame(data=[preds],columns=["Text","True emoji"]+model_names+["lstm"],index=["Emoji"])
	return df


# ==============test set sample==============
df = pd.DataFrame()
with open("models/df_testset.pkl","rb") as f:
	df = pickle.load(f)

# # load from df prediction
def prediction_testset():
	global pred_text,pred_true_value,mnb_pred,lr_pred,rf_pred,gbc_pred,xgb_pred,lstm_pred
	res = df.sample(1).iloc[0].to_list()
	lst_pred_test = []
	lst_pred_test.append(res.pop(0))
	for i in res:
		lst_pred_test.append(utils.label_to_emoji(i))

	# dump to global with emoji
	pred_text,pred_true_value,mnb_pred,lr_pred,rf_pred,gbc_pred,xgb_pred,lstm_pred = lst_pred_test
	
	df_res = pd.DataFrame(data=[lst_pred_test],columns=["Text","True emoji"]+model_names+["lstm"],index=["Emoji"])
	print("Finished prediction")
	# return str/int form
	return df_res

# must run to initiate global variables
prediction_testset()

# ==============render options==============
def render_default():
	return render_template('index.html',
		pred_text=pred_text,
		pred_true_value=pred_true_value,
		mnb_pred=mnb_pred,
		lr_pred=lr_pred,
		rf_pred=rf_pred,
		gbc_pred=gbc_pred,
		xgb_pred=xgb_pred,
		lstm_pred=lstm_pred,
		)

def render_t1(text="good to see you"):
	from pandas.io.formats.style import Styler
	
	# sample from test set
	res = prediction_testset()
	df_show = res.T
	s = Styler(df_show)
	t_pred = s.set_table_attributes('class="tg"').to_html(classes='data')

	# predict_text
	res = predict_text(text)
	df_text = res.T
	s2 = Styler(df_text)
	t_pred_text = s2.set_table_attributes('class="tg"').to_html(classes='data')

	return render_template('index2.html',
		tables=[t_pred], 
		titles=df_show.columns.values,

		tables_text=[t_pred_text],
		titles_text=df_text.columns.values,
		)

#==========render web site==========
app = Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
	# return render_t1()
	if request.method == 'POST':
		if request.form.get('action') == 'Randomize':
			return render_t1()

		elif request.form.get('action2') != None:
			text = request.form.get('action2')
			return render_t1(text)

	elif request.method == 'GET':
		return render_t1()

	return "error"

if __name__ == '__main__':
	app.run(host='0,0,0,0')