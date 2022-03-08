# EMFR (Ensembled Model Feature Reduction)

## Assumption
This module is intended to apply supervised feature reduction to MultiOmic data. Wich means your data should have a label for every instance, e.g: case vs control.  

## Method
EMFR creates three models those mentioned above after hyperparameter grid search with five fold cross validation to make sure that the best hyperparameters are selected, then each feature is scored three times once per model. Once the weighted information-gain (feature importance) is obtained from both ET and GBM and model coefficient is obtained from SVC, min-max normalization is implemented to confirm that all models scores are on the same scale, then weighted average of three scores is made for each feature to get one score per feature.  

