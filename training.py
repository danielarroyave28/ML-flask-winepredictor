# import libraries

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Data manipulation
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# model selection
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict

# machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import xgboost as xgb

# metrics
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_squared_error, precision_score, \
    recall_score

# Save models
import joblib

# misc
from scipy import stats

from custom_transformer import CombinedAttributesAdder

df = pd.read_csv("WineQT.csv")
df = df.drop(['Id'], axis=1)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['quality'])

# prepare data now
df_train_final = df_train.drop("quality", axis=1)
y_train = df_train["quality"].copy()

num_pipeline = Pipeline([
    ('attribs_adder', CombinedAttributesAdder(add_ratio_density=False)),
    ('min_max_scaler', MinMaxScaler())
])

df_train_prepared = num_pipeline.fit_transform(df_train_final.values)
df_train_prepared = pd.DataFrame(df_train_prepared, columns=list(df_train_final.columns) + ["fixed_volatile"],
                                 index=df_train_final.index)


le = LabelEncoder()
y_train_labeled = le.fit_transform(y_train)

models = [RandomForestClassifier(), xgb.XGBClassifier()]

scores = dict()

for m in models:
    results = cross_val_score(m, df_train_prepared, y_train_labeled, cv=10, scoring="accuracy")

    print(f'model: {str(m)}')
    print(results)

    print('-------------')

param_grid = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'min_samples_split': [3, 5, 6, 7],
    'max_leaf_nodes': [3, 6, 9],
    'criterion': ['gini', 'entropy', 'log_loss']

}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring="accuracy")
grid_search.fit(df_train_prepared, y_train_labeled )

final_model = grid_search.best_estimator_

df_test_final =df_test.drop("quality", axis=1)
y_test = df_test["quality"].copy()

df_test_prepared = num_pipeline.transform(df_test_final.values)
df_test_prepared = pd.DataFrame(df_test_prepared, columns = list(df_test_final.columns) + ["fixed_volatile"], index=df_test_final.index)

y_test_labeled = le.transform(y_test)

final_pred = final_model.predict(df_test_prepared)
final_pred_2 = le.inverse_transform(final_pred)

production_model = Pipeline([
    ('preparation', num_pipeline),
    ('prediction', final_model)
])

joblib.dump(production_model, "final_prod_model.pkl")
joblib.dump(le,'labelEncoder.joblib',compress=9)

print("process ended")