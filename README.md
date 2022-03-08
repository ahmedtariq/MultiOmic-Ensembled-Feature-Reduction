# EMFR (Ensembled Model Feature Reduction)

## Package Description
This module provides tool to apply supervised feature reduction to MultiOmic data. the module is built with Python and can be imported and used directly. 

## Method
EMFR creates three models those mentioned above after hyperparameter grid search with five fold cross validation to make sure that the best hyperparameters are selected, then each feature is scored three times once per model. Once the weighted information-gain (feature importance) is obtained from both ET and GBM and model coefficient is obtained from SVC, min-max normalization is implemented to confirm that all models scores are on the same scale, then weighted average of three scores is made for each feature to get one score per feature.  

## Flow
![image](https://user-images.githubusercontent.com/32236950/157235707-b3e5bc5a-52c8-49b6-9879-7c5b7ea042bc.png)

## Assumption
The module uses ensemple of supervised machine learning models, hence label need to be provided e.g: case vs control.

## Prerequisites
- Manual  
Download and install the following from conda  
[pandas](https://anaconda.org/anaconda/pandas)  
[scikit-learn](https://anaconda.org/anaconda/scikit-learn)

## Demo
in this demo we select top 20 significant features for each omic data seprately using our EMFR method (ML_Selector) and concatinating significant features from each Omic together to get most significant MultiOmic features.  

```python
from ML_selector_utils import ML_Selector

type_DF_train = []
type_DF_test = []
sig = []
# creaate X DataFrame as all MultiOmics feature on columns and samples on rows. Create Y DataFrame as label, must be label encoded where categories encoded to 1,2,3,....
# Assuming feature names are made as 'FeatureType_feature' e.g: RNAseq_RPS5, ATAC_ES05, ....
for type_ in pd.Series([i.split('_')[0] for i in X_train.columns]).unique():
    column_type = [i for i in X_train.columns if type_ in i]
    X_type = X[column_type]
    mean_sig = ML_Selector(X_type, y_train, k=20)
    sig.append(mean_sig)
    type_DF.append(X_type[mean_sig.index])
X_filt = pd.concat(type_DF,axis=1)
```
