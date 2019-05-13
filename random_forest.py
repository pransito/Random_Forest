import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import ensemble

# building a model to predict unit cost

# Read in the data (assuming it has all predictors X, 
# and the label Y)
data = pd.read_csv('data.csv')

# make dummies of categorical variables
data = pd.get_dummies(data,drop_first=True)

# split into X and Y
Y = data["unit_cost"]
X = data.drop(labels = ["unit_cost"],axis = 1)

# Create Train & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

# prep a reporting function
def classificationReport(model,accs,prec_0,prec_1,auc):
	""" plots the AUC of CV runs and reports other CV performance measures"""
    plt.hist(auc)
    plt.title('ROC-AUC score for ' + type(model).__name__)
    plt.show()
    print('Accuracy: ' + str(np.mean(accs)))
    print('Balanced Accuracy: ' + str(np.mean(np.array([np.mean(prec_1),np.mean(prec_0)]))))
    print('Sensitivity (prec 1): ' + str(np.mean(prec_1)))
    print('Specificity (prec 0): ' + str(np.mean(prec_0)))
    print('ROC-AUC: ' + str(np.mean(auc)))

# prep a function to test stability of CV
def stabilityModelCv(model,runs,X,Y,test_size,do_report=True,eval_return='f2'):
    """Function to fit a model on training data and test on test data.
    multiple splits so stability of CV score is assessable"""
    accs = []
    prec_0 = []
    prec_1 = []
    auc = []
    f2 = []
    for ii in range(runs):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        
        # model fitting
        model.fit(X_train, y_train)
        prediction_test = model.predict(X_test)
        
        # for auc need proba
        predproba_test = model.predict_proba(X_test)
        probs = [p[1] for p in predproba_test]
           
        # get the metrics
        accs.append(metrics.accuracy_score(y_test, prediction_test))
        cur_prec_0 = metrics.precision_score(y_test, prediction_test,pos_label=False) # specificity
        prec_0.append(cur_prec_0)
        cur_prec_1 = metrics.precision_score(y_test, prediction_test,pos_label=True) # sensitivity
        prec_1.append(cur_prec_1)
        cur_recall = metrics.recall_score(y_test, prediction_test,pos_label=True)
        auc.append(metrics.roc_auc_score(y_true=y_test,y_score=probs))
        cur_f2 = 5*(cur_prec_1*cur_recall/(4*cur_prec_1+cur_recall))
        f2.append(cur_f2)
    
    # report
    if do_report:
        classificationReport(model,accs,prec_0,prec_1,auc)
    
    # return dictionary and return
    ret_dict = {'acc': accs,'auc': auc, 'f2': f2}
    return(ret_dict[eval_return])
	

# fit random forest
model = ensemble.RandomForestRegressor(n_estimators = 1000, random_state = 300)
model = model.fit(X_train, y_train)

# predict
prediction_test = model.predict(X_test)
print('Explained variance in test sample:')
print(metrics.r2_score(y_test, prediction_test))

# print a report on CV performance
# return here the f2 score: 5*(precision*recall/(4*precision+cur_recall))
# an plot the CV score for 30 runs 
f2 = stabilityModelCv(model,runs=30,X,Y,test_size,do_report=True,eval_return='f2')


