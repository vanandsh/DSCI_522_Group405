# New York Airbnb Price Predictor

## Proposal

For this data analysis, we would like to look at the Airbnb listings of New York City in 2019. This data set can either be retrieved from this Kaggle [page](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) or directly from the Airbnb data [site](http://insideairbnb.com/get-the-data.html).

We aim to investigate what would be the appropriate or comparable price for a future Airbnb listing in the same city in 2020, given the same set of listing features from the dataset. We want to know the prediction interval of that price as well. Through this predictive investigation, we also expect to explore what are the strongest predictors for price, what features are irrelevant, and what pertinent new features we can derive from the given predictors. All questions are predictive.

We will analyze the data by visualizing, building various models, and doing ensembling. We will visualize the relationships different features' relationships with price and check their correlations as well as distributions. In terms of modelling, we aim to train and tune different models, such as linear regression, random forest, and boosted trees, on the dataset. We may do stacking with all the models at the end to get a more accurate prediction. During the modelling phase, we also want to understand and compare feature importance in different models.

In terms of visualization, we will do a correlation table, a correlation heatmap, and a pair-wise plot with all the features to understand their relationships with each other from different angles. Doing this will give us an idea of how the predictive models will end up looking in terms of how different predictors influence them. Since we may do linear regression, this process will allow us to investigate multicollinearity, predictor distribution, and linear relationships with price, which relate to assumptions of linear regression. For summarizing our results, we will make a table showing feature importance with different models as well as their residual plots.

We will present our model results in different ways. As mentioned above, we will do a residual plot comparison from our models and a table showing different feature importance. Furthermore, we will dive into the statistical significance of the feature coefficients for linear regression and display the prediction intervals.

## Exploratory Data Analysis
[EDA Report](https://github.com/UBC-MDS/DSCI_522_Group405/blob/master/doc/EDA_report.ipynb)