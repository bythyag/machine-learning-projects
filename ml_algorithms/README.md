### project notes

1. the fit_intercept parameter in linear regression should be tested with both true and false values. it allows gridsearchcv to find out whether including an intercept in the model improves performance.
2. lr__c: inverse of regularization strength for logistic regression. smaller values mean stronger regularization, helping to prevent overfitting.
3. lbfgs: an optimizer (solver) efficient for small to medium datasets, supporting only l2 regularization.
4. liblinear: a solver suitable for small datasets, supporting both l1 and l2 regularization.



