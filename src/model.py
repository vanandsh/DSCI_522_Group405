# author: Ofer Mansour, Jacky Ho, Anand Vemparala
# date: 2020-01-23

'''
Train an average ensemble model based on a Random Forest regressor model,
a XGBoost regressor model and a LightGBM model with a set train and test data.
Then, output the results in charts and tables

Usage: model.py --source_file_location=<source_file_location> --target_location=<target_location>

source_file_location - a path/filename pointing to the data to be read in
target_location - a path/filename prefix where to write the figure(s)/table(s) to and what to call it

'''

import requests
from docopt import docopt
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import LabelEncoder
import altair as alt

from selenium import webdriver

alt.data_transformers.enable('json')

categorical_features = [
    'neighbourhood_group',
    'neighbourhood',
    'room_type'
]

tuning_parameter_map = {
    'random_forest': {
      'max_depth': [10, 50],
      'min_samples_split': [5, 20],
      'n_estimators': [600, 1500],
      'criterion': ['mse']
    },
    'xgboost': {
      'max_depth': [5, 7, 10],
      'colsample_bytree': [0.6, 0.7, 0.8],
      'n_estimators': [500, 1000]
    },
    'lightGBM': {
      'min_data_in_leaf': [100, 300, 500, 1000, 1500],
      'num_leaves': [15, 30, 40, 50, 60],
      'max_depth': [15, 30, 45]
    }
}

# Minimized hyperparameters for testing 
# 
# tuning_parameter_map = {
#     'random_forest': {
#       'max_depth': [10],
#       'criterion': ['mse']
#     },
#     'xgboost': {
#       'max_depth': [5]
#     },
#     'lightGBM': {
#       'max_depth': [15]
#     }
# }


model_map = {
    'random_forest': RandomForestRegressor(),
    'xgboost': XGBRegressor(),
    'lightGBM': LGBMRegressor()
}


opt = docopt(__doc__)

def preprocess(full_train, full_test):
    X_train = full_train.drop(['price'], axis=1)
    y_train = full_train['price']
    X_test = full_test.drop(['price'], axis=1)
    y_test = full_test['price']

    for feature in categorical_features:
      le = LabelEncoder()
      le.fit(X_train[feature])
      X_train[feature] = le.transform(X_train[feature])
      X_test[feature] = le.transform(X_test[feature])

    print('Data preprocessed!')

    return [X_train, y_train, X_test, y_test]

def train_base_models(X_train, y_train):
    models = []

    for model_name in model_map.keys():
        model = GridSearchCV(
          estimator=model_map[model_name],
          param_grid=tuning_parameter_map[model_name],
          # cv=4,
          cv=2,
          verbose=2,
          n_jobs=-1,
          scoring='neg_mean_absolute_error'
        )

        print('Training ' + model_name + ' regressor')

        if model_name == 'lightGBM':
            model.fit(X_train, y_train, eval_metric='l1')
        elif model_name == 'xgboost':
            model.fit(X_train, y_train, eval_metric='mae')
        else:
            model.fit(X_train, y_train)

        models.append(model)

    return models 
    
def average_ensemble_models(models, X):
    return np.average(
        list(map(lambda x: x.predict(X), models)),
        axis=0
    )
    
def save_ensemble_residual_graphs(save_to, models, X, y):
  
    ensemble_residual_df = pd.DataFrame({
      'true_price': y,
      'average_ensemble_residual': y - average_ensemble_models(models, X)
    })

    residual_chart = alt.Chart(
        ensemble_residual_df
    ).mark_circle(size=30, opacity=0.4).encode(
      x=alt.X('true_price', title='Price'),
      y=alt.Y('average_ensemble_residual', title='Average ensembling residual')
    ).properties(
      width=850,
      height=500
    ).properties(
      title='Average Ensembling Residuals on Test Data'
    )

    residual_dist_chart = alt.Chart(ensemble_residual_df).mark_bar().encode(
      x=alt.X(
        'average_ensemble_residual',
        title='Average ensembling residual',
        bin=alt.Bin(extent=[-1200, 2000], step=5)
      ),
      y='count()'
    ).properties(
      width=850,
      height=500
    ).properties(
      title='Ensembling Residual Distribution'
    )

    with alt.data_transformers.enable('default'):
        residual_chart.save(
            save_to + '/ensemble_residual_plot.png'
        )
        residual_dist_chart.save(
            save_to + '/ensemble_residual_distribution.png'
        )
def save_feature_importance_table(save_to, models, columns):
    feature_important_df = pd.DataFrame({
      'Random Forest':
          models[0].best_estimator_.feature_importances_,
      'XGBoost':
          models[1].best_estimator_.feature_importances_,
      'LightGBM': (
          models[2].best_estimator_.feature_importances_/sum(
              models[2].best_estimator_.feature_importances_
          )
       )
    })

    feature_important_df.index = columns

    feature_important_df.to_csv(
        save_to + '/feature_importance_table.csv'
    )


def save_model_performance_table(save_to, models, X, y):
    test_mean_absolute_error_df = pd.DataFrame({
      'mean_absolute_error': [
        mean_absolute_error(
            y,
            DummyRegressor(strategy='median').fit(X, y).predict(X)
        ),
        mean_absolute_error(y, models[0].predict(X)),
        mean_absolute_error(y, models[1].predict(X)),
        mean_absolute_error(y, models[2].predict(X)),
        mean_absolute_error(y, average_ensemble_models(models, X)),
      ]
    })

    test_mean_absolute_error_df.index = [
        'Median Null Model',
        'Random Forest',
        'XGBoost',
        'LightGBM',
        'Average Ensembling'
    ]

    test_mean_absolute_error_df.to_csv(
        save_to + '/mean_absolute_error_table.csv'
    )


def main(source_file_location, target_location):
  
  train_file = source_file_location + "/train.csv"
  test_file = source_file_location + "/test.csv"
  
  results_plots_folder = target_location + "/plots"
  results_tables_folder = target_location + "/tables"
  
  full_train = pd.read_csv(train_file)
  full_test = pd.read_csv(test_file, index_col=0)

  X_train, y_train, X_test, y_test = preprocess(full_train, full_test)
  models = train_base_models(X_train, y_train)
  save_ensemble_residual_graphs(results_plots_folder, models, X_test, y_test)
  save_feature_importance_table(results_tables_folder, models, X_test.columns)
  save_model_performance_table(results_tables_folder, models, X_test, y_test)
  

main(opt['--source_file_location'], opt['--target_location'])
