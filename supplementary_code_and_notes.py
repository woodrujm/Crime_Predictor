<<<<<<< HEAD
=======
# Get the best features
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
model = RandomForestClassifier()
#
y = basic_df.pop('resolution').values
x_feat = tf.values
X_train_feat, X_test_feat, y_train_feat, y_test_feat = train_test_split(x_feat,y, random_state=42)
model.fit(X_train_feat,y_train_feat)
importances = {}
scores = {}
importance_df = pd.DataFrame({'feature':tf.columns,'importance':np.round(model.feature_importances_,5)})
importance_df = importance_df.sort('importance', ascending=False)
# #
for i in xrange(1,500,50):
    df = dc.add_vectorized_matrix_to_df(tf[importance_df.feature[:i].values.tolist()])
    x = df.values
    X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=42)
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    print i, 'features'
    print 'accuracy:', accuracy_score(y_test,prediction)
    print 'precision:', precision_score(y_test,prediction)
    print 'recall:', recall_score(y_test,prediction)
    print '\n\n'
    importances[i] = pd.DataFrame({'feature':df.columns,'importance':np.round(model.feature_importances_,5)})
    scores[i] =  [accuracy_score(y_test,prediction),precision_score(y_test,prediction),recall_score(y_test,prediction)]


import matplotlib.pyplot as plt
import pandas as pd
import csv
files = ['importances1.csv','importances2.csv','importances3.csv','importances4.csv','importances5.csv','importances6.csv','importances7.csv','importances8.csv','importances9.csv','importances10.csv']
for i, f in zip(importances.keys(),files):
     importances[i].to_csv(f,index=False)

import matplotlib.pyplot as plt
import pandas as pd
files = ['importances1.csv','importances2.csv','importances3.csv','importances4.csv','importances5.csv','importances6.csv','importances7.csv','importances8.csv','importances9.csv','importances10.csv']
importances_dictionary = {}
for name,path in zip([name.replace('.csv','') for name in files],files):
    importances_dictionary[name]= pd.read_csv(path)

for key,value in  importances_dictionary.iteritems():
    importances_dictionary[key] = value.sort_values('importance',ascending=False)

fig, axes = plt.subplots(5,2)
for ax, (key, value) in zip(axes.ravel(),importances_dictionary.iteritems()):
    labels = value.head(19).feature
    ax.set_xticklabels(xrange(1,20), rotation=90 )
    ax.bar(xrange(1,20),value.importance.head(19),tick_label = labels,align='center')

    # scores = {}
    # all_topics = {}
    # for i in [1,3,5,7,10]:
    #     w,topics = dc.reduce_with_NMF(tf,features,i*10)
    #     w = pd.DataFrame(w)
    #     df = dc.add_vectorized_matrix_to_df(w)
    #     x = dc.df.values
    #     X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=42)
    #     model.fit(X_train,y_train)
    #     prediction = model.predict(X_test)
    #     print 'accuracy:', accuracy_score(y_test,prediction)
    #     print 'precision:', precision_score(y_test,prediction)
    #     print 'recall:', recall_score(y_test,prediction)
    #     print '\n\n'
    #     scores[i] = [accuracy_score(y_test,prediction),precision_score(y_test,prediction),recall_score(y_test,prediction)]
    #     all_topics[i] = topics


>>>>>>> 5ad30a43a971440869354dd50ce069df501ee991


<<<<<<< HEAD
=======
        # gettint the best number of topics
    # scores = {}
    # all_topics = {}
    # for i in [1,3,5,7,10]:
    #     w,topics = dc.reduce_with_NMF(tf,features,i*10)
    #     w = pd.DataFrame(w)
    #     df = dc.add_vectorized_matrix_to_df(w)
    #     x = dc.df.values
    #     X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=42)
    #     model.fit(X_train,y_train)
    #     prediction = model.predict(X_test)
    #     print 'accuracy:', accuracy_score(y_test,prediction)
    #     print 'precision:', precision_score(y_test,prediction)
    #     print 'recall:', recall_score(y_test,prediction)
    #     print '\n\n'
    #     scores[i] = [accuracy_score(y_test,prediction),precision_score(y_test,prediction),recall_score(y_test,prediction)]
    #     all_topics[i] = topics

# Reconstruction error: 0.000877
# Topic 0:
# locked auto grand theft unlocked person pickpocket vehicle recovered petty
# Topic 1:
# malicious mischief vandalism vehicles graffiti breaking windows suspect tire slashing
# Topic 2:
# petty theft property building shoplifting prior bicycle lost unlocked receiving
# Topic 3:
# stolen automobile truck vehicle recovered plate lost receiving sf motorcycle
# Topic 4:
# battery sexual relationship dating spouse officer police injuries violation parole
# Topic 5:
# revoked drivers suspended license plate lost stolen peddling tab grand


# Reconstruction error: 0.000597
# Topic 0:
# locked auto grand theft unlocked pickpocket vehicle attempted strip bicycle recovered pursesnatch embezzlement employee possibly
# Topic 1:
# mischief malicious vandalism vehicles graffiti windows breaking suspect tire slashing buses cars adult bb street
# Topic 2:
# petty theft shoplifting auto locked prior unlocked bicycle property building strip attempted lost coin operated
# Topic 3:
# stolen automobile truck plate lost motorcycle receiving license knowledge sf recovered possession vehicle miscellaneous attempted
# Topic 4:
# battery sexual spouse relationship dating officer police injuries parole municipal code enroute assault child permit
# Topic 5:
# revoked drivers suspended license plate lost stolen peddling tab spouse relationship dating enroute explosive encouraging
# Topic 6:
# suspicious occurrence act female shots possible fired child possibly sex package solicits prostitution auto person
# Topic 7:
# violation traffic probation parole order restraining code municipal police permit officer stay away general park
# Topic 8:
# burglary entry unlawful forcible apartment residence house store construction hot prowl attempted flat tools att
# Topic 9:
# property theft grand petty receiving lost money knowledge personation receive false pickpocket stolen possession obtaining
# Topic 10:
# possession cocaine rock base sale paraphernalia narcotics marijuana sales amphetamine meth heroin substance controlled tools
# Topic 11:
# threats life danger leading immoral teachers school grounds incident disrupts activities disturbance public bring possess
# Topic 12:
# person grand theft suspicious shoplifting bicycle pickpocket ammunition prohibited poss attempted prior unlocked pursesnatch embezzlement
# Topic 13:
# vehicle recovered outside enroute jurisdiction sf auto stolen attempted department corrections officer parole robbery tampering
# Topic 14:
# building theft grand pickpocket petty bicycle attempted unlocked embezzlement pursesnatch employee robbery arson adult bodily
>>>>>>> 5ad30a43a971440869354dd50ce069df501ee991
<<<<<<< HEAD
=======


#WORK FLOW FOR GENERAL DATAFRAME
# resolutions_to_delete = ['UNFOUNDED','JUVENILE CITED','JUVENILE ADMONISHED','JUVENILE DIVERTED','CLEARED-CONTACT JUVENILE FOR MORE INFO','PROSECUTED FOR LESSER OFFENSE']
# dc.delete_rowvalues('resolution',resolutions_to_delete)
#     # -- Other values to delete:
# categories_to_delete =  ['NON-CRIMINAL','FRAUD','SECONDARY CODES', 'FORGERY/COUNTERFEITING','TRESPASS','DISORDERLY CONDUCT','DRUNKENNESS','DRIVING UNDER THE INFLUENCE','LIQUOR LAWS','LOITERING','BAD CHECKS','SEX OFFENSES, NON FORCIBLE','GAMBLING','PORNOGRAPHY/OBSCENE MAT','TREA']
# dc.delete_rowvalues('category',categories_to_delete)
# dc.make_binary('resolution', ['NONE'])
# # write down which numbers coorespond to which
# for column in ['pddistrict','category','dayofweek']:
#     dc.create_dummies(column)
# # dc.replace_all_categoricals(['pddistrict','category','dayofweek'])
# #     -- cant do this. NEED TO MAKE DUMMIES OUT OF THESE
#
# dc.drop_columns(['pdid','unique_key','address','location'])
#

# get arrest out
# dc.df = dc.df[dc.df.descript.str.contains('ARREST') == False]

 # dc.df.to_csv('/Users/hercules/Desktop/Galvanize/capstone/github_capstone/data/basic_df.csv',index=False)
#     - Remove these columns
#
# # ************LOGISTIC REGRESSION****************
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# model = LogisticRegression()
#
# # FOR BASE MODEL
# # df_lr = df_lr.drop('descript',axis=1)
# #convert datetime to total minutes since 2002
# # df_logreg = dc.convert_time_to_total_minutes(2003)
# # df_logreg = df_logred.drop(['longitude','latitude','timestamp'],axis=1)
#
# df_nonvect
# descriptcolumn = vec_df.descript
#
# # RUNNING BASEMODEL
# base_y = df_lr.pop('resolution').values
# base_x = df_lr.values
# bxtr, bxts, bytr, byts = train_test_split(base_x,base_y,test_size = .33,random_state=42)
# model.fit(bxtr,bytr)
# y_base_predict = model.predict(bxtr)
# print "accuracy:", accuracy_score(bytr,y_base_predict)
# print "precision:", precision_score(bytr,y_base_predict)
# print "recall:", recall_score(bytr,y_base_predict)

# accuracy: 0.615432053786
# //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
#   'precision', 'predicted', average, warn_for)
# precision: 0.0
# recall: 0.0



#Feature Engineering
# df with years months and days stripped out
# Getting years, months, days and hours to see if that helps
# for date in ['year','month','day','hour']:
#     dc.time_column(date)
# dc.drop_columns('timestamp')

# changing timestamp to year month day hour
# accuracy: 0.624886035051
# precision: 0.620364294603
# recall: 0.0633520652907
# dc.df.to_csv('/Users/hercules/desktop/galvanize/capstone/data/df_year_month_day_hour.csv')

#
# removing location
# accuracy: 0.617009346459
# precision: 0.649486272528
# recall: 0.00891000341981


# # with vectorized description

#
# --------------------------------------------------------------------------------
# LOG REG ON DESCRIPTIONS
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.metrics.pairwise import linear_kernel
# from sklearn.cluster import KMeans
# from nltk.stem.wordnet import WordNetLemmatizer
# import string
# import nltk
# import re
#
#
# features, tf, cosine_similarities = vectorize('descript')



# y  = dc.df.resolution.values
# descript_x = tf
# bxtr, bxts, bytr, byts = train_test_split(tf,y,test_size = .33,random_state=42)
# model.fit(bxtr,bytr)
# y_base_predict = model.predict(bxtr)
# print "accuracy:", accuracy_score(bytr,y_base_predict)
# print "precision:", precision_score(bytr,y_base_predict)
# print "recall:", recall_score(bytr,y_base_predict)

#
# accuracy: 0.873401602707
# precision: 0.851948839183
# recall: 0.811894022719

# ---try with countvectorizer?
# countvec = CountVectorizer(stop_words = 'english')
#
# descript_vectors = pd.DataFrame(countvec.fit_transform(dc.df['descript'].todense(), columns=countvec.get_feature_names())
#



# make dummies out of categorical
# for column in ['pddistrict','category','dayofweek']:
#     dc.create_dummies(column)
#
#
# family_category = {'category_family':['category_KIDNAPPING','category_MISSING PERSON','category_RUNAWAY','category_FAMILY OFFENSES']}
# sex_nonrape = {'category_nonviolen_sexcrime':['category_PROSTITUTION']}
# week_days = {'week_days':['dayofweek_Monday','dayofweek_Tuesday','dayofweek_Wednesday','dayofweek_Thursday']}
# week_ends = {'week_ends':['dayofweek_Friday','dayofweek_Saturday','dayofweek_Sunday']}
# #
# for dct_of_columns_to_combine in [family_category,sex_nonrape,week_days,week_ends]:
#     for key,values in dct_of_columns_to_combine.iteritems():
#         dc.combine_dummies(key,values)

# dicts = [family_category,sex_nonrape,week_days,week_ends]
# for values in dicts:
#     dc.drop_columns(values)






# categorical_columns = ['pddistrict','category','dayofweek']
# self.df = make_dummies(self.df,categorical_columns)

# df['category_FAMILY'] = (df['category_KIDNAPPING'])+(df['category_MISSING PERSON'])+(df['category_RUNAWAY'])+(df['category_FAMILY OFFENSES'])

# df['FRAUD'] = df['category_BAD CHECKS'] + df['category_FRAUD']
# df['sexual_nonrape'] = df['category_PORNOGRAPHY/OBSCENE MAT'] + df['category_PROSTITUTION'] + df['category_SEX OFFENSES, NON FORCIBLE']


# columns_to_drop = ['pdid','unique_key','address','longitude','latitude','category_BAD CHECKS','category_FRAUD','category_PORNOGRAPHY/OBSCENE MAT','category_PROSTITUTION','category_SEX OFFENSES, NON FORCIBLE','category_KIDNAPPING','category_MISSING PERSON','category_RUNAWAY','category_FAMILY OFFENSES']
# df = drop_columns(df,columns_to_drop)


# RANDOM FORESTS
# from sklearn.ensemble import RandomForestClassifier
#
# modelRF = RandomForestClassifier()
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# modelRF.fit(bxtr,bytr)
# y_base_predict = modelRF.predict(bxtr)
# print "accuracy:", accuracy_score(bytr,y_base_predict)
# print "precision:", precision_score(bytr,y_base_predict)
# print "recall:", recall_score(bytr,y_base_predict)
# features = model.feature_importances_
# features
# important_names = feature_names[features > np.mean(features)]
# importances = pd.DataFrame({'feature':dc.df.columns,'importance':np.round(model.feature_importances_,5)})
# importances = importances.sort_values('importance',ascending=False).set_index('feature')
# print importances
# importances.plot.bar()
# plt.show()
# import matplotlib.pyplot as plt

# accuracy: 0.844903703035
# precision: 0.824544631948
# recall: 0.757461197079

# from sklearn.ensemble import AdaBoostClassifier
#
# modelADA = AdaBoostClassifier()
# modelADA.fit(bxtr,bytr)
#
# y_base_predict = modelADA.predict(bxts)
# print "accuracy:", accuracy_score(byts,y_base_predict)
# print "precision:", precision_score(byts,y_base_predict)
# print "recall:", recall_score(byts,y_base_predict)
#
# accuracy: 0.822535744892
# precision: 0.775544855916
# recall: 0.757202902294
#
#
# from sklearn.ensemble import GradientBoostingClassifier
# modelGB = GradientBoostingClassifier()
#
# y_base_predict = modelGB.predict(bxts)
# print "accuracy:", accuracy_score(byts,y_base_predict)
# print "precision:", precision_score(byts,y_base_predict)
# print "recall:", recall_score(byts,y_base_predict)

# accuracy: 0.828222452754
# precision: 0.795058423688
# recall: 0.74486110785

# -----------------------------------
# With vectorized descrptions:
#
# modelRF.fit(bxtr,bytr)
# y_base_predict = modelRF.predict(bxts)
# print "accuracy:", accuracy_score(byts,y_base_predict)
# print "precision:", precision_score(byts,y_base_predict)
# print "recall:", recall_score(byts,y_base_predict)
# features = model.feature_importances_
# # features
# important_names = feature_names[features > np.mean(features)]
# importances = pd.DataFrame({'feature':dc.df.columns,'importance':np.round(model.feature_importances_,5)})
#
#
#
# accuracy: 0.988569737663
# precision: 0.994588966391
# recall: 0.975585296646
#
#
# Scores of just the "arrest" and "day" vectors (arrest is the most signal for description, day is the least)
# accuracy: 0.663030747877
# precision: 0.943250086887
# recall: 0.131695211335
# # --------------------------------------------------------
#WORK FLOW FOR GENERAL DATAFRAME
# dc.delete_rowvalues('resolution',resolutions_to_delete)
#     - resolutions_to_delete = ['UNFOUNDED','JUVENILE CITED','JUVENILE ADMONISHED','JUVENILE DIVERTED','CLEARED-CONTACT JUVENILE FOR MORE INFO','PROSECUTED FOR LESSER OFFENSE']
    # -- Other values to delete: categories_to_delete =  ['NON-CRIMINAL','FRAUD','SECONDARY CODES', 'FORGERY/COUNTERFEITING','TRESPASS','DISORDERLY CONDUCT','DRUNKENNESS','DRIVING UNDER THE INFLUENCE','LIQUOR LAWS','LOITERING','BAD CHECKS','SEX OFFENSES, NON FORCIBLE','GAMBLING','PORNOGRAPHY/OBSCENE MAT','TREA']
#
# RangeIndex: 2060484 entries, 0 to 2060483
# Data columns (total 12 columns):
# unique_key    int64
# category      object
# descript      object
# dayofweek     object
# pddistrict    object
# resolution    object
# address       object
# longitude     float64
# latitude      float64
# location      object
# pdid          int64
# timestamp     datetime64[ns]
# dtypes: datetime64[ns](1), float64(2), int64(2), object(7)
# memory usage: 188.6+ MB





# df = dc.make_binary('resolution', ['NONE'])
# dc.replace_all_categoricals(['pddistrict','category','dayofweek'])
#write down which numbers coorespond to which
# df = dc.drop_columns(['pdid','unique_key','address','location'])
#
# Int64Index: 2026616 entries, 0 to 2060483
# Data columns (total 12 columns):
# unique_key    int64
# category      int64
# descript      object
# dayofweek     int64
# pddistrict    int64
# resolution    int64
# address       object
# longitude     float64
# latitude      float64
# location      object
# pdid          int64
# timestamp     datetime64[ns]
# dtypes: datetime64[ns](1), float64(2), int64(6), object(3)
# memory usage: 201.0+ MB

#
# basic df
#
# - vectorize descript
# - drop descript from df
# - add tf to df
# -
>>>>>>> 5ad30a43a971440869354dd50ce069df501ee991


# - need to delete one of each of the dummy variables. need to add constant??


# access aws
# ssh -i "~/.ssh/crime.pem" ec2-user@ec2-54-148-132-173.us-west-2.compute.amazonaws.com

# copy files into aws
# scp -i "~/.ssh/crime.pem" /Users/hercules/Desktop/Galvanize/capstone/data/basic_df.csv ec2-user@ec2-54-148-132-173.us-west-2.compute.amazonaws.com:
# /Users/hercules/Desktop/Galvanize/capstone/data/basic_df.csv
# /Users/hercules/Desktop/Galvanize/capstone/github_capstone/clean_data.py
# /Users/hercules/Desktop/Galvanize/capstone/github_capstone/models.py

# scp -i ~/.ssh/crime.pem ec2-user@ec2-34-210-242-59.us-west-2.compute.amazonaws.com:/home/ec2-user/df_with_vectors.csv .




            # #Use results of coarse search to do a more fine-grained search.
            # fine_params = {param:np.linspace(.75*coarse_params[tuning_params],
            #                                  1.25*coarse_params[tuning_params],
            #                                  5) for param in coarse_params
            # }
            # for int_params in ['n_estimators', 'max_depth']:
            #     #Set as integer, and make sure > 0
            #     s = fine_params[int_params]
            #     fine_params[int_params] = s.astype(int)[s>=1]
            #
            #     fine_search = GridSearchCV(model, fine_params)
            #     fine_search.fit(self.X_train, self.y_train)
            #
            #     print "Fine search best params: %s " % fine_search.best_params_
            #     print "Fine search best score: %s " % fine_search.best_score_
            #
            # return fine_search

# tuning_paramsGB = [{'learning_rate':np.logspace(-2,2,num=4),
#       'n_estimators':np.logspace(1,2,num=4).astype(int),
#       'max_depth':np.linspace(1,75, num=4).astype(int)}]
#
# tuning_paramsRF = [{'max_depth': np.linspace(1,75, num=4).astype(int), 'n_estimators': [100,1000,2000]}]
#
# tuning_paramsADA = [{'n_estimators':[100,1000,2000],'learning_rate':np.logspace(-2,2,num=4)}]
#
# tuning_paramsLR = [{'C': [.001,.01,1]}]


# on entire dataset LogisticRegression had ~87%

# For three year dataframe:
# --------------
# LogisticRe :
# accuracy: 0.884217790961
# precision: 0.844108139877
# recall: 0.665971135455
#
# --------------
# RandomFore :
# accuracy: 0.885304396347
# precision: 0.830694872513
# recall: 0.687358720223
#
# --------------
# AdaBoostCl :
# accuracy: 0.883865378403
# precision: 0.825076537712
# recall: 0.687300759288
#
# --------------
# GradientBo :
# accuracy: 0.885098822355
# precision: 0.876457434915
# recall: 0.636121254275
=======
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

        coarse_search = GridSearchCV(model, tuning_params)  #refit=False,
        coarse_search.fit(X_train, y_train)

        print str(model)[:10]
        print "Coarse search best params: %s " % coarse_search.best_params_
        print "Coarse search best score: %s \n\n" % coarse_search.best_score_


                # #Use results of coarse search to do a more fine-grained search.
                # fine_params = {param:np.linspace(.75*coarse_params[tuning_params],
                #                                  1.25*coarse_params[tuning_params],
                #                                  5) for param in coarse_params
                # }
                # for int_params in ['n_estimators', 'max_depth']:
                #     #Set as integer, and make sure > 0
                #     s = fine_params[int_params]
                #     fine_params[int_params] = s.astype(int)[s>=1]
                #
                #     fine_search = GridSearchCV(model, fine_params)
                #     fine_search.fit(X_train, self.y_train)
                #
                #     print "Fine search best params: %s " % fine_search.best_params_
                #     print "Fine search best score: %s " % fine_search.best_score_
                #
                # return fine_search
    #
    # tuning_paramsGB = [{'learning_rate':np.logspace(-2,2,num=4),
    #       'n_estimators':np.logspace(1,2,num=4).astype(int),
    #       'max_depth':np.linspace(1,75, num=4).astype(int)}]
    #
    # tuning_paramsRF = [{'max_depth': np.linspace(1,75, num=4).astype(int), 'n_estimators': [100,1000,2000]}]
    #
    # tuning_paramsADA = [{'n_estimators':[100,1000,2000],'learning_rate':np.logspace(-2,2,num=4)}]
    #
    # tuning_paramsLR = [{'C': [.001,.01,1]}]

    # on entire dataset LogisticRegression had ~87%

    # For three year dataframe:
    # --------------
    # LogisticRe :
    # accuracy: 0.884217790961
    # precision: 0.844108139877
    # recall: 0.665971135455
    #
    # --------------
    # RandomFore :
    # accuracy: 0.885304396347
    # precision: 0.830694872513
    # recall: 0.687358720223
    #
    # --------------
    # AdaBoostCl :
    # accuracy: 0.883865378403
    # precision: 0.825076537712
    # recall: 0.687300759288
    #
    # --------------
    # GradientBo :
    # accuracy: 0.885098822355
    # precision: 0.876457434915
    # recall: 0.636121254275
>>>>>>> 5ad30a43a971440869354dd50ce069df501ee991
