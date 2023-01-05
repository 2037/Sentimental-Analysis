# README
# Introduction
This project is to test machine learning algorithms on prediction of emoji based on input text, like product reviews, or the sentiment of the financial news. 

For word embedding, I used Word2Vec method based on Glove (glove.6B.50d.txt). Then I treat it as a supervised multiclass classification problem.
The following algorithms are used to be trained and predict: 
* Multinomial Naïve Bayes
* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LSTM

The best predictor is Gradient Boosting with 62.5% accuracy. Ideally the LSTM will perform the best since it consider the context of text, which is more accurate than tokenization and word-embedding. We need more dataset to testify the performance of models. 

The application is deployed on http://jackwang2037.com/projects/.
