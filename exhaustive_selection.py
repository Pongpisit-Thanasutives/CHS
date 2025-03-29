from sklearn.linear_model import LinearRegression as LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.model_selection import LeaveOneOut

def best_subset(X, y, support_size=1, scoring='r2', cv=5, leave_one_out=False):
    if leave_one_out:
        cv = LeaveOneOut()
        scoring = 'neg_mean_squared_error'
    efs = EFS(LinearRegression(fit_intercept=False), min_features=support_size, max_features=support_size, scoring=scoring, cv=cv, print_progress=False)
    efs.fit(X, y)
    return efs.best_idx_

