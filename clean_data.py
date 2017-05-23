import pandas as pd
from sklearn.cross_validation import train_test_split
from statsmodels.tools import add_constant
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import NMF

class DataCleaning(object):

    def __init__(self,path,timestamp=None):
        self.df = pd.read_csv(path, parse_dates= timestamp)



    # Make dummy variables
    def create_dummies(self,dummy_column):
        self.df = pd.get_dummies(self.df, prefix=dummy_column, columns=[dummy_column])

    #Combine Dummy Variables
    def combine_dummies(self,new_column_name,dummy_columns_to_combine):
        self.df[new_column_name] = pd.DataFrame(np.zeros((len(self.df),1)))
        for dummy in dummy_columns_to_combine:
            self.df[new_column_name] = self.df[new_column_name] + self.df[dummy]
        # self.df = self.drop_columns(dummy_columns_to_combine) #this is returning nothing


    # Test improvement with times as columns from timestamp
    def time_column(self,frequency):
        self.df[frequency] = self.df['timestamp'].dt.__getattribute__(frequency)


    def convert_time_to_total_minutes(self,beginning_of_year):
    # gets the total seconds since the beginning of the desired year (use 2003 for total dataframe, if interested in specific years make sure to filter timesamp by desired year)
        self.df['timestamp'] = ((self.df.timestamp - dt.datetime(beginning_of_year,1,1)).dt.total_seconds())/(60*30)
        # divide by 60*30 to get half hours

    # Test if numberic is better than dummy?
    def replace_categorical_with_numeric(self,column):
        # enumerate instead
        cat_list = self.df[column].unique()
        replace_dict = {}
        for i in xrange(len(cat_list)):
            replace_dict[cat_list[i]] = i
        self.df = self.df.replace({column:replace_dict})

    def replace_all_categoricals(self,columns):
        for column in columns:
            self.replace_categorical_with_numeric(column)

    def delete_rowvalues(self,column,values_to_delete):
        self.df =  self.df[~self.df[column].isin(values_to_delete)]

    #make resolution binary
    def make_binary(self,column,neg_values):
        #would isin be better? scould pass it a list for the negative?
        # def make_resolution_binary(df):
        #     return [0 if crime == "NONE" else 1 for crime in df.resolution]

        self.df[column] = [0 if any(neg_value in row for neg_value in neg_values) else 1 for row in self.df[column]]

    # drop any columns converted or useless columns
    def drop_columns(self,columns_to_drop):
        self.df = self.df.drop(columns_to_drop,axis = 1)



    def count_vectorize(self,column_to_vectorize):
        countvec = CountVectorizer(stop_words = 'english')
        return pd.DataFrame(countvec.fit_transform(self.df[column_to_vectorize]).toarray(), columns=countvec.get_feature_names())

    def add_vectorized_matrix_to_df(self,vector_matrix):
        return pd.concat([self.df,vector_matrix],axis=1,join='inner')

# Don't really need this
    # def save_df(self,path):
    #     self.df.to_csv(path,index=False)

    def my_tokenizer(self,doc, lemmatizer=WordNetLemmatizer(), stopwords=stopwords):
        tokens = word_tokenize(doc.decode('utf-8'))
        tokens = [t.lower() for t in tokens if t not in string.punctuation]
        tokens = [lemmatizer.lemmatize(t) if type(t) == str else t for t in tokens]
        return ' '.join(tokens)

    def vectorize(self,series_to_vectorize):
        documents = self.df[series_to_vectorize].tolist()
        new_docs = []
        for doc in documents:
            new_docs.append(self.my_tokenizer(doc))
        tfidf = TfidfVectorizer(stop_words='english')

        tf = pd.DataFrame(tfidf.fit_transform(documents).toarray(),columns = tfidf.get_feature_names())
        features = tfidf.get_feature_names()

        # Cosine Similarity using TF-IDF

        # 1. Compute cosine similarity
        # cosine_similarities = linear_kernel(tf, tf) #squareform(pdist(vec_mat, metric='cosine'))?
        return features, tf#, cosine_similarities


    def reduce_with_NMF(self,tf,features, n_topics,n_top_words=15):
        nmf = NMF(n_components=n_topics)
        W = nmf.fit_transform(tf)
        H = nmf.components_
        print("Reconstruction error: %f") % (np.array(tf - W.dot(H))**2).mean()
        topics = {}
        for topic_num, topic in enumerate(H):
            print("Topic %d:" % topic_num)
            print(" ".join([features[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            topics[topic_num] = topic
        return W, topics



    def make_x_y(self, y_column, dummy_columns_to_drop = None, logreg = False):
        y = self.df.pop(y_column).values
        if logreg:
            self.df = self.drop_columns(dummy_columns_to_drop)
            x = self.df.values
        else:
            x = self.df.values
        return x,y


    def train_test(self,x,y):
        return train_test_split(x,y)
