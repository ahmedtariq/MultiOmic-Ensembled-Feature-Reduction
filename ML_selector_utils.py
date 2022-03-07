import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

def SVC_Selector(X,y):
    param_grid = {'C': [0.01,0.1,1,10,100]}
    SVC_CV = GridSearchCV(SVC( class_weight='balanced', kernel='linear',max_iter=100),param_grid,scoring='f1', cv=5)
    SVC_CV_fit = SVC_CV.fit(X,y)
    DF_sig = pd.DataFrame({'feature': X.columns, 'SVC_sig': SVC_CV_fit.best_estimator_.coef_.flatten()})
    DF_sig = DF_sig.sort_values(by='SVC_sig',ascending=False)
    return DF_sig
	
def ET_Selector(X,y):
    param_grid = {'n_estimators': [50,100,150,200], 'max_depth':[10,50,100],'class_weight':['balanced']}
    ET_CV = GridSearchCV(ExtraTreesClassifier(),param_grid, scoring='f1', cv=5)
    ET_CV_fit = ET_CV.fit(X,y)
    DF_sig = pd.DataFrame({'feature': X.columns, 'ET_sig': ET_CV_fit.best_estimator_.feature_importances_})
    DF_sig = DF_sig.sort_values(by='ET_sig',ascending=False)
    return DF_sig
	
def GBM_Selector(X,y):
    param_grid = {'n_estimators': [100,500,1000]}
    GBM_CV = GridSearchCV(GradientBoostingClassifier(),param_grid, scoring='f1', cv=5)
    GBM_CV_fit = GBM_CV.fit(X,y)
    DF_sig = pd.DataFrame({'feature': X.columns, 'GBM_sig': GBM_CV_fit.best_estimator_.feature_importances_})
    DF_sig = DF_sig.sort_values(by='GBM_sig',ascending=False)
    return DF_sig
	
def ML_Selector(X_train, y_train, k=100):
    Selectors = {"SVC" : SVC_Selector,"ET": ET_Selector,"GBM": GBM_Selector }

    DF_sig_dict = {}
    for name, selector in Selectors.items():
        DF_sig = selector(X_train, y_train)
        DF_sig_dict[name] = DF_sig

    DF_sig_all = pd.concat([v.set_index('feature') for k,v in DF_sig_dict.items()], axis=1)
    DF_sig_all = (DF_sig_all - DF_sig_all.min()) / (DF_sig_all.max() - DF_sig_all.min())
    mean_sig = DF_sig_all.mean(axis=1).sort_values(ascending = False).head(k)
    return mean_sig