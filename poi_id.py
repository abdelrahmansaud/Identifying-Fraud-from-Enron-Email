#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import numpy as np
import pprint as pp
sys.path.append("../tools/")

from collections import defaultdict
from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from tester import dump_classifier_and_data
from sklearn.cross_validation import StratifiedShuffleSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm, grid_search
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
'''features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
	'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock','director_fees', 'to_messages', 'from_poi_to_this_person',
	'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'f1', 'f2'] '''
features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
#print "Features: ", features_list
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers


data_dict.pop( "TOTAL", 0 )


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

ms = defaultdict(int)

#Replacing NaN with 0 for financial features
for i in my_dataset:
    #print data_dict[i]
    for j in data_dict[i]:
        if not j in ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',\
            'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']\
            and	data_dict[i][j] == 'NaN':
            data_dict[i][j] = 0
            #ms[j] += 1

            
#print "Number of missing features: "
#pp.pprint(ms)
			

#New Features
for p in my_dataset:
    try:
        if not my_dataset[p]['restricted_stock_deferred'] == 'NaN' and not my_dataset[p]['exercised_stock_options'] == 'NaN' and not my_dataset[p]['exercised_stock_options'] == 0:
            my_dataset[p]['f1'] = my_dataset[p]['restricted_stock_deferred']/my_dataset[p]['exercised_stock_options']
        else:
            my_dataset[p]['f1'] = 0.0
        if not my_dataset[p]['restricted_stock'] == 'NaN' and not my_dataset[p]['exercised_stock_options'] == 'NaN' and not my_dataset[p]['exercised_stock_options'] == 0:
            my_dataset[p]['f2'] = my_dataset[p]['restricted_stock']/my_dataset[p]['exercised_stock_options']
        else:
            my_dataset[p]['f2'] = 0.0
    except:
        pass

### Extract features and labels from dataset for local testing



data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


#pp.pprint(features)
scaler = MinMaxScaler()#.fit_transform(features)
selector = SelectKBest(f_classif)#.fit_transform(features, labels)


#exit()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.



rf = RandomForestClassifier(class_weight={0:.3, 1:.7})
knn = KNeighborsClassifier()
ada = AdaBoostClassifier()
sv = SVC(class_weight={0:.1, 1:.9}, cache_size=1500)
nb = GaussianNB()

rf_pipe = Pipeline([('selector', selector), ('rf', rf)])
knn_pipe = Pipeline([('scaler', scaler), ('selector', selector), ('knn', knn)])
ada_pipe = Pipeline([('selector', selector), ('ada', ada)])
sv_pipe = Pipeline([('selector', selector), ('sv', sv)])
nb_pipe = Pipeline([('selector', selector), ('nb', nb)])


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3)   

cv = StratifiedShuffleSplit(labels,500, test_size=0.3)

scores = ['precision_weighted', 'recall_weighted', 'accuracy', 'f1_weighted']

def getRF():

    print "==============="
    print "RandomForests"
    print "==============="

    for score in scores:

        print score
        print

        #parameters = {'n_estimators':range(10, 150, 10), 'criterion':['gini', 'entropy'], 'min_samples_split':range(2, 8, 2)}
        parameters = {'rf__n_estimators':range(10, 150, 10), 'rf__criterion':['gini', 'entropy'], 'rf__min_samples_split':range(2, 8, 2), 
            'selector__k':range(3, 22, 1)}	

        gs = grid_search.GridSearchCV(rf_pipe, parameters, scoring=score, cv=cv)
            
        gs.fit(features, labels)

         #This is the model you pass to tester.py
        clf = gs.best_estimator_

        print " "
        print "Optimal Model - by Grid Search"
        print clf
        print " "

        best_parameters = gs.best_estimator_.get_params()

        print " "
        print "Best Parameters- by Grid Search"
        print best_parameters
        print " "

        labels_pred = gs.predict(features)

        # Print Results  (will print the Grid Search score)
        print "Grid Search Classification report:" 
        print " "
        print classification_report(labels, labels_pred)
        print ' ' 

        # Print Results  (will print the tester.py score)
        print "tester.py Classification report:" 
        print " "
        test_classifier(clf, my_dataset, features_list)
        print " "
        print

def getKNN():

    print "==============="
    print "KNeighborsClassifier"
    print "==============="

    for score in scores:

        print score
        print

        #parameters = {'n_neighbors':range(2, 10, 2), 'weights':['distance', 'uniform'], 'metric':['minkowski', 'euclidean']}
        parameters = {'knn__n_neighbors': range(2, 10, 2), 'knn__weights':['distance', 'uniform'], 'knn__metric':['minkowski', 'euclidean'], 
            'selector__k':range(3, 20, 1)}

        gs = grid_search.GridSearchCV(knn_pipe, parameters, scoring=score, cv=cv)

        gs.fit(features, labels)

         #This is the model you pass to tester.py
        clf = gs.best_estimator_

        print " "
        print "Optimal Model - by Grid Search"
        print clf
        print " "

        best_parameters = gs.best_estimator_.get_params()

        print " "
        print "Best Parameters- by Grid Search"
        print best_parameters
        print " "

        labels_pred = gs.predict(features)

        # Print Results  (will print the Grid Search score)
        print "Grid Search Classification report:" 
        print " "
        print classification_report(labels, labels_pred)
        print ' ' 

        # Print Results  (will print the tester.py score)
        print "tester.py Classification report:" 
        print " "
        test_classifier(clf, my_dataset, features_list)
        print " "
        print

def getAda():
		
	print "==============="
	print "AdaBoost"
	print "==============="

	for score in scores:

		print score
		print

		#parameters = {'n_estimators':range(50, 100, 1), 'learning_rate':[x * 0.01 for x in range(100, 160, 1)]}
		parameters = {'ada__n_estimators': range(1, 100, 20), 'ada__learning_rate':[x * 0.01 for x in range(100, 160, 10)],
			'selector__k':range(3, 22, 1)}

		gs = grid_search.GridSearchCV(ada_pipe, parameters, scoring=score, cv=cv)

		gs.fit(features, labels)

		 #This is the model you pass to tester.py
		clf = gs.best_estimator_

		print " "
		print "Optimal Model - by Grid Search"
		print clf
		print " "

		best_parameters = gs.best_estimator_.get_params()

		print " "
		print "Best Parameters- by Grid Search"
		print best_parameters
		print " "

		labels_pred = gs.predict(features)

		# Print Results  (will print the Grid Search score)
		print "Grid Search Classification report:" 
		print " "
		print classification_report(labels, labels_pred)
		print ' ' 

		# Print Results  (will print the tester.py score)
		print "tester.py Classification report:" 
		print " "
		test_classifier(clf, my_dataset, features_list)
		print " "
		print

def getSVC():
		
	print "==============="
	print "SVC"
	print "==============="

	for score in scores:

		print score
		print

		parameters = {'sv__C': [0.01, 0.1, 1, 500, 1000, 5000, 10000, 50000, 100000], 'sv__kernel':['linear'],
			'selector__k':range(3, 22, 1)} #'sv__gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1, 10, 100, 500, 1000], 

		gs = grid_search.GridSearchCV(sv_pipe, parameters, scoring=score, cv=cv)

		gs.fit(features, labels)

		 #This is the model you pass to tester.py
		clf = gs.best_estimator_

		print " "
		print "Optimal Model - by Grid Search"
		print clf
		print " "

		best_parameters = gs.best_estimator_.get_params()

		print " "
		print "Best Parameters- by Grid Search"
		print best_parameters
		print " "

		labels_pred = gs.predict(features)

		# Print Results  (will print the Grid Search score)
		print "Grid Search Classification report:" 
		print " "
		print classification_report(labels, labels_pred)
		print ' ' 

		# Print Results  (will print the tester.py score)
		print "tester.py Classification report:" 
		print " "
		test_classifier(clf, my_dataset, features_list)
		print " "
		print

def getNB():

	print "==============="
	print "GaussianNB"
	print "==============="

	for score in scores:

		print score
		print

		parameters = {'selector__k':range(3, 22, 1)}	

		gs = grid_search.GridSearchCV(nb_pipe, parameters, scoring=score, cv=cv)
			
		gs.fit(features, labels)

		 #This is the model you pass to tester.py
		clf = gs.best_estimator_

		print " "
		print "Optimal Model - by Grid Search"
		print clf
		print " "

		best_parameters = gs.best_estimator_.get_params()

		print " "
		print "Best Parameters- by Grid Search"
		print best_parameters
		print " "

		labels_pred = gs.predict(features)

		# Print Results  (will print the Grid Search score)
		print "Grid Search Classification report:" 
		print " "
		print classification_report(labels, labels_pred)
		print ' ' 

		# Print Results  (will print the tester.py score)
		print "tester.py Classification report:" 
		print " "
		test_classifier(clf, my_dataset, features_list)
		print " "
		print
'''
knn_pipe.set_params(knn__p= 2, knn__metric= 'minkowski', selector__k= 21, scaler__copy= True, scaler= MinMaxScaler(copy=True, feature_range=(0, 1)), knn__weights= 'distance', selector= SelectKBest(k=20, score_func=f_classif), knn__leaf_size= 30, knn__algorithm= 'auto', scaler__feature_range= (0, 1), knn__n_neighbors= 2, selector__score_func= f_classif, knn__metric_params= None)

clf = knn_pipe.fit(features, labels)

'''

#nb_pipe.set_params(selector__k= 5, selector__score_func= f_classif)

clf = nb.fit(features, labels)

#print "Feature Score:"
#print zip(clf.named_steps['selector'].scores_, features_list[1:])


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
'''
list = []
if not feat_new[0] == 'poi': 
	list.append('poi')
for i in feat_new:
	list.append(i)
print list
'''
dump_classifier_and_data(clf, my_dataset, features_list)