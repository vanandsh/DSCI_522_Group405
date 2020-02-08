# author: Ofer Mansour, Jacky Ho, Anand Vemparala
# date: 2020-01-23

'''
Train an average ensemble model based on a Random Forest regressor model,
a XGBoost regressor model and a LightGBM model with a set train and test data.
Then, output the results in charts and tables in terms of test dataset performance.

Usage: model.py --source_file_location=<source_file_location> --target_location=<target_location>

source_file_location - a path to a folder containing train.csv and test.csv
target_location - a path to a folder where the result charts and tables will reside

'''

import requests
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
import os

from docopt import docopt
try:
    from schema import Schema, And, Or, Use, SchemaError
except ImportError:
    exit('This example requires that `schema` data-validation library'
         ' is installed: \n    pip install schema\n'
         'https://github.com/halst/schema')

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
      'criterion': ['mse'],
      'random_state': [0]
    },
    'xgboost': {
      'max_depth': [5, 7, 10],
      'colsample_bytree': [0.6, 0.7, 0.8],
      'n_estimators': [500, 1000],
      'random_state': [0]
    },
    'lightGBM': {
      'min_data_in_leaf': [100, 300, 500, 1000, 1500],
      'num_leaves': [15, 30, 40, 50, 60],
      'max_depth': [15, 30, 45],
      'random_state': [0]
    }
}

model_map = {
    'random_forest': RandomForestRegressor(),
    'xgboost': XGBRegressor(),
    'lightGBM': LGBMRegressor()
}

opt = docopt(__doc__)

# Label-encode the categorical features and split the data
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

    assert len(X_train.columns) == (len(full_train.columns) - 1)

    return [X_train, y_train, X_test, y_test]

# Train the base models, which are Random Forest, XGBoost and LightGBM in that
# order, and return the GridSearchCV wrappers of them
def train_base_models(X_train, y_train):
    models = []

    for model_name in model_map.keys():
        model = GridSearchCV(
          estimator=model_map[model_name],
          param_grid=tuning_parameter_map[model_name],
          cv=4,
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

    assert len(models) == len(model_map.keys())

    return models

# Average the base models as an ensemble
def average_ensemble_models(models, X):
    predictions = list(map(lambda x: x.predict(X), models))

    assert len(predictions) == len(models)

    return np.average(predictions, axis=0)

# Save the residual graphs from the models
def save_ensemble_residual_graphs(save_to, models, X, y):
    assert isinstance(save_to, str) == True

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
      width=350,
      height=500
    ).properties(
      title='Average Ensembling Residuals on Test Data'
    )

    residual_dist_chart = alt.Chart(ensemble_residual_df).mark_bar().encode(
      x=alt.X(
        'average_ensemble_residual',
        title='Average ensembling residual',
        bin=alt.Bin(extent=[-500, 500], step=25)
      ),
      y='count()'
    ).properties(
      width=350,
      height=500
    ).properties(
      title='Ensembling Residual Distribution'
    )

    model_result_charts = (residual_chart | residual_dist_chart).configure_axis(
      labelFontSize=15,
      titleFontSize=15
    )

    with alt.data_transformers.enable('default'):
        model_result_charts.save(save_to + '/model_result_charts.png')

def save_feature_importance_table(save_to, models, columns):
    assert len(models) == 3

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

def get_model_performance(models, X, y):
    assert len(models) == 3

    return [
      mean_absolute_error(
        y,
        DummyRegressor(strategy='median').fit(X, y).predict(X)
      ),
      mean_absolute_error(y, models[0].predict(X)),
      mean_absolute_error(y, models[1].predict(X)),
      mean_absolute_error(y, models[2].predict(X)),
      mean_absolute_error(y, average_ensemble_models(models, X))
    ]

# Output the model performances in terms of mean absolute error on a table
def save_model_performance_table(save_to, models, X_train, y_train, X_test, y_test):
    mean_absolute_error_df = pd.DataFrame({
      'train_mean_absolute_error': get_model_performance(
        models, X_train, y_train
      ),
      'test_mean_absolute_error': get_model_performance(
        models, X_test, y_test
      )
    })

    assert len(mean_absolute_error_df.index) == 5

    mean_absolute_error_df.index = [
        'Median Null Model',
        'Random Forest',
        'XGBoost',
        'LightGBM',
        'Average Ensembling'
    ]

    mean_absolute_error_df.to_csv(
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
    save_model_performance_table(
        results_tables_folder, models, X_train, y_train, X_test, y_test
    )

if __name__ == "__main__":
    schema = Schema({
        '--source_file_location': And(
            os.path.exists, error='source path should exist'
        ),
        '--target_location': And(
            os.path.exists, error='target path should exist'
        )
    })
    try:
        args = schema.validate(opt)
    except SchemaError as e:
        exit(e)

    main(opt['--source_file_location'], opt['--target_location'])
