from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np

from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import csv
import collections





class Models(object):

    def __init__(self,path=False,y_column=False,timestamp=None):

        self.df = pd.read_csv(path)
        if y_column:
            self.y = self.df.pop(y_column).values
        self.x = self.df.values
        self.feature_names = self.df.columns

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x,self.y,random_state=42)
        self.modelLR = LogisticRegression(C=.01)
        self.modelRF = RandomForestClassifier()
        self.modelAGA = AdaBoostClassifier()
        self.modelGB = GradientBoostingClassifier()
        self.models = [self.modelLR,self.modelRF,self.modelAGA,self.modelGB]
        self.predictions = {}
        self.scores = {}
        self.importances_dictionary = {}

    def fit_models(self):
        for model in self.models:
            model.fit(self.X_train,self.y_train)
            self.predictions[str(model)[:10]] = model.predict(self.X_test)

    def model_scores(self):

        feature_models = ['RandomForest','AdaBoost','GradientBoost']

        self.importances_dictionary = {}
        for model in self.models:
            self.scores[str(model)[:10]] = [accuracy_score(self.y_test,self.predictions[str(model)[:10]]),precision_score(self.y_test,self.predictions[str(model)[:10]]),recall_score(self.y_test,self.predictions[str(model)[:10]])]
            print '--------------'
            print str(model)[:10], ':'
            print 'accuracy:', self.scores[str(model)[:10]][0]
            print 'precision:', self.scores[str(model)[:10]][1]
            print 'recall:', self.scores[str(model)[:10]][2], '\n\n'
            if any(feat_model in str(model) for feat_model in feature_models):
                self.importances = pd.DataFrame({'feature':self.df.columns,'importance':np.round(model.feature_importances_,5)})
                print "\n\n"
                self.importances = self.importances.sort('importance',ascending=False)
            else:
                print "\n"
            self.importances_dictionary[str(model)[:10]] = self.importances



    # dont the numeric representations mean nothing for this?
    def plot_top_feature_importances(self,number_of_features,show=False):

        self.top_features = self.importances[:number_of_features]
        self.top_features.importance = self.top_features.importance/self.top_features.importance.max()

        self.top_features.plot(x='feature',y='importance',kind='bar')
        if show:
            plt.show




    def grid_search_models(self):
        for model in self.models:

                #tuning_params = [{'C': [1]}]

            if "Gradient" in str(model)[:10]:
                #tuning_params = [{'max_depth': [2, 3, 5], 'learning_rate': [.001, .01, .1, .2]}]
                tuning_params = [{'learning_rate':np.logspace(-2,2,num=4),
                      'n_estimators':np.logspace(1,2,num=4).astype(int),
                      'max_depth':np.linspace(1,75, num=4).astype(int)}]
            elif "Ada" in str(model)[:5]:
                tuning_params = [{'n_estimators':[100,1000,2000],'learning_rate':np.logspace(-2,2,num=4)}]
            elif "Logistic" in str(model)[:10]:
                tuning_params = [{'C': [.001,.01,1]}]
            # elif "SVM" in str(model):
                #tuning_params = [{'kernel':['linear', 'rbf'], 'gamma':[0.0001, 0.001, .01],'C':[1, 10, 100, 1000]}]
                # tuning_params = [{'kernel':['rbf'], 'gamma':[0.0001,.001,.01],'C':[1,10,100,1000]}]
        #These params are a coarse search.
            elif "RandomForest" in str(model):
            #  #tuning_params = [{'max_depth': [2, 3, 5], 'n_estimators': [1000]}]
                tuning_params = [{'max_depth': np.linspace(1,75, num=4).astype(int), 'n_estimators': [100,1000,2000]}]

            coarse_search = GridSearchCV(model, tuning_params, n_jobs=-1)  #refit=False,
            coarse_search.fit(self.X_train, self.y_train)

            print str(model)[:10]
            print "Coarse search best params: %s " % coarse_search.best_params_
            print "Coarse search best score: %s \n\n" % coarse_search.best_score_
