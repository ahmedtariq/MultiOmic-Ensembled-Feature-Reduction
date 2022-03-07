import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, auc, roc_auc_score, precision_score, recall_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV

random.seed = 0
np.random.seed = 0

def ML_Pipeline(X_train, y_train,tytle,test = False, X_test= None,y_test=None,used_models = ['SVC', 'RF'], cv_scoring='average_precision',num_folds = 10, plot=True):
    #This function apply the full ML pipeline on several models and returns the models and the results of each 
    seed = 7
    scoring = cv_scoring
    # adding mcc to available evaluation
    cv_scoring = matthews_corrcoef if cv_scoring == "mcc" else cv_scoring
    
    models = []
    models.append(('SVC', SVC( class_weight='balanced', max_iter=100, probability=True)))
    models.append(('ET', ExtraTreesClassifier(class_weight='balanced')))
    models.append(('LR', LogisticRegression(class_weight='balanced',max_iter=500)))
    models.append(('RF', RandomForestClassifier(class_weight='balanced')))
    models.append(('GBM', GradientBoostingClassifier(learning_rate=0.001)))

    hyperparameters = {\
                       'SVC':{'C': [0.01,0.1,1,10,100], 'kernel':['linear']},\
                       'ET':{'n_estimators': [50,100,150,200], 'max_depth':[10,50,100]},\
                       'LR':{'C': [0.01,0.1,1,10], 'penalty': ['l1','l2']},\
                       'RF':{'n_estimators': [50,100,150,200], 'max_depth':[10,50,100]},\
                       'GBM': {'n_estimators': [100,500,1000]}}

    # checking used models to make sure in available list
    for i in used_models:
        if i not in [i for i,v in models]:
            raise NameError(i +' is not in the set of allowed models')

    results = {}
    names = []
    msgs = []
    estimators = {}
    print('Model', 'val_mean_score', 'val_std_score')
    for name, model in models:
        if name not in used_models:
            continue
        GS = GridSearchCV(model,hyperparameters[name], scoring=scoring, cv=num_folds)
        GS_fit = GS.fit(X_train, y_train)
        best_index = GS_fit.best_index_
        results[name] = pd.DataFrame(GS_fit.cv_results_).drop('params',axis=1).loc[best_index]
        estimators[name] = GS_fit.best_estimator_
        names.append(name)
        msg = (name, pd.DataFrame(GS_fit.cv_results_).drop('params',axis=1).loc[best_index,'mean_test_score'],\
                             pd.DataFrame(GS_fit.cv_results_).loc[best_index,'std_test_score'])
        
        print(msg)
        msgs.append(msg)
        if test:
            print("test_score")
            print('recall', recall_score(y_test, GS_fit.best_estimator_.predict(X_test)))
            print('precision', precision_score(y_test, GS_fit.best_estimator_.predict(X_test)))
            print("f1_score", f1_score(y_test, GS_fit.best_estimator_.predict(X_test)))
            print("Avg_Percision_score", average_precision_score(y_test, GS_fit.best_estimator_.predict_proba(X_test)[:,1]))
            print("PR_AUC_score", PR_AUC(y_test, GS_fit.best_estimator_.predict_proba(X_test)))
            print("ROC_AUC_score", roc_auc_score(y_test, GS_fit.best_estimator_.predict_proba(X_test)[:,1]))
            print('')

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.boxplot(np.array([results[i][[i for i in results[i].index if ('_test_score' in i) & ('split' in i)]].values for i in names  ]).T)
        plt.ylabel(cv_scoring+"_score")
        ax.set_xticklabels(names)
        plt.ylim(-.05,1.05)
        plt.show()

    return pd.DataFrame(results), estimators

def random_feature_selector(X_train, X_test, k):
    # this function returns X_train_filtered, X_test_filtered when k number of features selected randomy 
    col_filt = np.random.choice(X_train.columns,replace=False,size= k)
    X_train_filt = X_train.loc[:, col_filt]
    X_test_filt = X_test.loc[:, col_filt]
    return X_train_filt, X_test_filt

def Recursive_feature_selector_evaluation(X_train,y_train, X_test,y_test,Model, hub):
    # this function returns np array with the number of features at each hub and their relative f1 score 
    X_train_filt = X_train
    X_test_filt = X_test
    scores = []
    for i in range(0,60,hub):
        Model_fit = Model.fit(X_train_filt,y_train)
        scores.append((60-i,f1_score(y_test, Model_fit.predict(X_test_filt))))
        try:
            imp = Model_fit.coef_.flatten()
        except:
            imp = Model_fit.feature_importances_
        X_train_filt = X_train_filt.iloc[:,pd.Series(imp).nlargest(60-i-hub).index]
        X_test_filt = X_test_filt.iloc[:,pd.Series(imp).nlargest(60-i-hub).index]
        sns.lineplot(scores[:,0],scores[:,1])
    return np.array(scores)

def PR_AUC (y_true, y_proba):
    # calculate the precision-recall auc
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:,1])
    return auc(recall, precision)

