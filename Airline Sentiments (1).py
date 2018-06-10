
# coding: utf-8

# In[ ]:

## Imports
import sklearn
import csv
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sklearn.feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
import nltk
import re
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

#%% import and describe the data
tweets=pd.read_csv("C:/Users/zamey/Documents/DataSets/Tweets.csv")
print(tweets.head())
tweets.info()


# In[ ]:


# convert sentiment attributes to 1,0,-1
#tweets.loc[tweets['airline_sentiment']=='positive','airline_sentiment']=1
#tweets.loc[tweets['airline_sentiment']=='neutral','airline_sentiment']=0
#tweets.loc[tweets['airline_sentiment']=='negative','airline_sentiment']=2

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

norm_tweets=normalize_corpus(tweets['text'])
#%%
print(norm_tweets)
print(tweets.head(100))
# instantiate countVectorizer() object
vec = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
airlineList=tweets['airline'].unique()
#print(airlineList)
#%%
for AL in airlineList:
    airTweet=tweets.loc[(tweets['airline']==AL) & (tweets['airline_sentiment']=='negative')]
    BOW=vec.fit_transform(airTweet['text'])
    print(AL)
    #print(BOW)
    idf=vec.idf_
    a=dict(zip(vec.get_feature_names(), idf))
    aSort=sorted(a.items(), key=lambda x: x[1], reverse=True)
    bSort=list(x[0] for x in aSort[:100])
    print(bSort)
    
    # print wordcloud for airline
    
    wordcloud = WordCloud(max_font_size=30).generate(" ".join(bSort))
    
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

baggedTweets=vec.fit_transform(norm_tweets)
print(baggedTweets)
baggedTweetsMat = baggedTweets.toarray()
print(baggedTweetsMat)
vocab = vec.get_feature_names()
data=pd.DataFrame(baggedTweetsMat, columns=vocab)
print(tweets['airline_sentiment'])
X_train,X_test,y_train,y_test=train_test_split(baggedTweetsMat,tweets['airline_sentiment'],test_size=0.3,random_state=0)
print(X_train)
print(y_train)


#%%Define learning curve generation function

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


#%%%%% Random Forest Classifier #######################################################################

## Please apply the Random Forest Method to this dataset and find the feature_importance coefficient and try to visualize the feature importance matrix.
#fig, ax =plt.subplots(figsize=(10,10))

rfc=RandomForestClassifier(random_state=0)

rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)

print('Precision, recall, f1 score, and support are: \n\n'+classification_report(y_pred,y_test))

cm=confusion_matrix(y_pred,y_test)
print('Confusion Matrix:')
print(cm)
   

title = "Learning Curves (Random Forest)"
# Cross validation with 10 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=3
                  , test_size=0.2, random_state=0)

estimator = RandomForestClassifier()
plot_learning_curve(estimator, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.show()

#%%


title = "Learning Curves (Logistic Regression)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.

#from sklearn.model_selection import learning_curve
cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)

estimator = LogisticRegression(C=100,penalty='l1',tol=.01)
plot_learning_curve(estimator, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.show()

#%% logisticregression model




# Set regularization parameter

    # turn down tolerance for short training time
    clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
    clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
    clf_l1_LR.fit(X_train, y_train)
    clf_l2_LR.fit(X_train, y_train)

    coef_l1_LR = clf_l1_LR.coef_.ravel()
    coef_l2_LR = clf_l2_LR.coef_.ravel()

    # coef_l1_LR contains zeros due to the
    # L1 sparsity inducing norm

    sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
    sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100

    print("C=%.2f" % C)
    print("Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR)
    print("score with L1 penalty: %.4f" % clf_l1_LR.score(X_train, y_train))
    print("Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR)
    print("score with L2 penalty: %.4f" % clf_l2_LR.score(X_train, y_train))

#%% Boosted Tree Model
    


# gridsearch (pieces taken from kaggle "xgboost with GridSearchCV" module)
# Optimal parameters found to be md=8,mcw=1,gamma=0.5; other param options removed to reduce run time
parameters = {'max_depth':[8], 'min_child_weight':[1],'gamma':[.5]}
boozt=XGBClassifier()
clfbt=GridSearchCV(boozt,parameters,n_jobs=4,scoring='roc_auc')
clfbt.fit(X_train,y_train)
y_boozt_pred=clfbt.predict(X_test)
y_score_boozt = clfbt.predict_proba(X_test)[:,1]
# uncomment for gridsearch results
best_parameters, score, _ = max(clfbt.grid_scores_, key=lambda x: x[1])
print(score)

for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))


# print the ROC curve
#ROC(y_test,y_score_boozt)


# print stats
print('Precision, recall, f1 score, and support for boosted tree model are: \n\n'+classification_report(y_boozt_pred,y_test))

cmxg=confusion_matrix(y_boozt_pred,y_test)
print('Confusion Matrix:')
print(cmxg)


#%% Multi-layer Perceptron Model

n = 1000  #chunk row size
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

for i,chunk in enumerate(pd.read_csv("C:/Users/zamey/Documents/DataSets/Tweets.csv", chunksize=1000,header=None)):
    vec = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    Xprime=chunk.iloc[:,10:11]
    #print(X)
    Y=chunk.iloc[:,1].droplabels
    #print(Y)
    norm_X=normalize_corpus(Xprime)
    
    baggedTweets=vec.fit_transform(norm_X.ravel())
    #print(baggedTweets)
    baggedTweetsMat = baggedTweets.toarray()
    #print(baggedTweetsMat)
    vocab = vec.get_feature_names()
    data=pd.DataFrame(baggedTweetsMat, columns=vocab)
    #print(tweets['airline_sentiment'])
    X_train,X_test,y_train,y_test=train_test_split(baggedTweetsMat,Y,test_size=0.3,random_state=0)
    print(X_train)
    print(y_train)
    clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                        solver='sgd', verbose=10,  random_state=21,tol=0.000000001,warm_start=True)
    clf.partial_fit(X_train,y_train,['positive','negative','neutral'])
    pred=clf.predict(X_test)
    cmx=confusion_matrix(pred,y_test)
    print(cmx)
    
    
#trainChunks = [X_train[i:i+n] for i in range(0,X_train.shape[0],n)]
##print(trainChunks)
#trainResponseChunks=[y_train[i:i+n] for i in range(0,y_train.shape[0],n)]
#print(trainResponseChunks)
#
#for i in range(0,len(trainChunks)):

    #print(trainResponseChunks[i])
    

    