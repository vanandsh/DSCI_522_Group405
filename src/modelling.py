'''
Train an average ensemble model based on a Random Forest regressor model,
a XGBoost regressor model and a LightGBM model with a set train and test data.
Then, output the results in charts and tables

Usage: modelling.py --train_location=<train_path> --test_location=<test_path> --output_location=<out_path>

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

alt.data_transformers.enable('json')

opt = docopt(__doc__)

random_state = 0

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

model_map = {
    'random_forest': RandomForestRegressor(random_state=random_state),
    'xgboost': XGBRegressor(random_state=random_state),
    'lightGBM': LGBMRegressor(random_state=random_state)
}

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
          param_grid=tuning_parameter_map[model_map],
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

    return models

def average_ensemble_models(models, X):
    return np.average(
        map(lambda x: x.predict(X), models),
        axis=0
    )

def save_ensemble_residual_graphs(models, X, y):
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
            out_path + '/ensemble_residual_plot.png'
        )
        residual_dist_chart.save(
            out_path + '/ensemble_residual_distribution.png'
        )

def save_feature_importance_table(models, columns):
    feature_important_df = pd.DataFrame({
      'Random Forest': models[0].feature_importances_,
      'XGBoost': models[1].feature_importances_,
      'LightGBM': (
          models[2].best_estimator_.feature_importances_/sum(
              models[2].best_estimator_.feature_importances_
          )
       )
    })

    feature_important_df.index = columns

    feature_important_df.to_csv(
        out_path + '/feature_importance_table.csv'
    )

def save_model_performance_table(models, X, y):
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
        'Median Null Model'
        'Random Forest',
        'XGBoost',
        'LightGBM',
        'Average Ensembling'
    ]

    test_mean_absolute_error_df.to_csv(
        out_path + '/mean_absolute_error_table.csv'
    )

def main(train_path, test_path, out_path):
    full_train = pd.read_csv(train_path, index_col=0)
    full_test = pd.read_csv(test_path, index_col=0)

    X_train, y_train, X_test, y_test = preprocess(full_train, full_test)
    models = train_base_models(X_train, y_train)
    save_ensemble_residual_graphs(models, X_test, y_test)
    save_feature_importance_table(models, X_test.columns)
    save_model_performance_table(models, X_test, y_test)

if __name__ == "__main__":
    main(opt["--train_path"], opt["--test_path"], opt['--out_path'])
