New York Airbnb Price Prediction 2019
================
Ofer Mansour, Jacky Ho, Anand Vemparala

# Summary

In this project, our team attempted to predict Airbnb prices in New York
by testing several tree-based algorithms and finding the best performing
model. The models used in this project were: Random Forest, XGBoost,
LightGBM and an average ensembling of the three algorithms, and
evaluated by Mean Absolute Error. The average ensembling of Random
Forest, XGBoost and LightGBM was the best performing model with a Mean
Absolute Error of $63.94.

# Introduction

We aim to investigate what would be the appropriate or comparable price
for a future Airbnb listing in New York in 2020, given the same set of
listing features from the dataset. New York is one of the most visited
cities in the world, with a predicted 67 million tourists visiting in
2019 (Mcgeehan 2019b) and 65 million tourists (Mcgeehan 2019a) visiting
in 2018. With Airbnb’s popularity continuing to rise and its disruption
of the hotel industry, more people are staying in Airbnb’s, as 500
million stays in an Airbnb have occurred since 2008 (Sherwood 2019), the
company’s founding. With the popularity of both New York as a tourist
attraction and Airbnb, being able to predict the price of an Airbnb in
New York, given the same set of listing features from the dataset, would
be extremely useful. An accurate and reasonable price prediction for a
listing would be helpful for New York hosts, especially new hosts, to
set their prices correctly.

# Data

The dataset used in this project is about Airbnb listings in New York in
2019. The Airbnb listing data is from[Inside
Airbnb](http://insideairbnb.com/get-the-data.html)(“Get the Data -
Inside Airbnb. Adding Data to the Debate.” n.d.), which is compiled from
Airbnb’s website. Each of the 48,895 rows represents an Airbnb listing,
which includes several features, such as the price of the listing,
neighbourhood, room type and the number of reviews.

# EDA

The Airbnb data has 16 variables, where 11 are numerical and 5 are
categorical.

Variables:

  - **id:** Integer representing listing identification number.  
  - **name:** The title of the listing.  
  - **availability\_365:** The number of days in the year that the
    listing is available to book.  
  - **calculated\_host\_listings\_count:** The total number of listings
    the host of the listing has.  
  - **host\_id:** Integer representing the host’s identification
    number.  
  - **host\_name:** First name(s) of host(s).  
  - **latitude:** The latitude of location of listing.  
  - **longitude:** The longitude of location of listing.  
  - **minimum\_nights:** The minimum nights a user must book the
    listing.  
  - **neighbourhood:** The neighbourhood the listing is in.  
  - **neighbourhood\_group:** The borough the listing is in. (The 5
    boroughs of New York are Brooklyn, Manhattan, the Bronx, Staten
    Island and Queens).  
  - **number\_of\_reviews:** The number of reviews the listing has.  
  - **price (response variable):** The per night price of the listing.  
  - **reviews\_per\_month:** The number of reviews the listing has per
    month.  
  - **room\_type:** The type of room of the listing. (The 3 options are:
    Entire home/apartment, Private room and Shared room).  
  - **last\_review:** The date of the last review posted of the listing.

We dropped ‘id’,‘name’, ‘host\_id’, ‘host\_name’,‘last\_review’ as they
do not provide any insight on the price of a listing. To standardize the
data, we imputed values by replacing NaNs with zeros wherever required.

To understand how strongly the remaining features are correlated to the
price of a listing, we generated a correlation plot for all numerical
features.

<img src="../results/plots/corr-plot.png" width="2100" /> Figure 1.
Correlation and distribution plots for numerical features

We observe that no individual predictor by itself is highly correlated
to price.

We look at the distribution of our target variable price to get a
general sense of the prices in all neighbourhoods of New York. There is
an outlier that stands at $10,000. Other than this outlier, all prices
are mostly less than $1000, with an average of $152 per night.

<img src="../results/plots/price-dist.png" width="2100" /> Figure 2.
Distribution of New York Airbnb prices per night in 2019

To understand the categorical features, we grouped the prices per night
of the Airbnb listings based on the neighbourhood and the room type.

<img src="../results/plots/categorical-plots.png" width="2700" /> Figure
3. Mean New York Airbnb prices per night based by room type and
neighbourhood group in 2019

The mean price per night per night based by room type is as we would
have expected, asprices of entire home/apts are higher than private
room, followed by shared accomodations.

From the different neighbourhoods, Manhattan is the most expensive,
while the Bronx is the cheapest. We expect that neighbourhood group and
type of room will be important features in predicting price.

# Our chosen models, metric and why

For our prediction, we have chosen Random Forest, XGBoost, and LightGBM
as our models; also, we have picked mean absolute error as our metric
and average ensembling as our final prediction. We have decided to pick
only ensemble tree-based algorithms because they generally have better
predictive performances than other regressors, such as linear
regression. Furthermore, the feature, neighbourhood, has over 200
categories, and the only models that can process it with labels are
tree-based. Otherwise, we would have to encode it and hence face high
cardinality problems, which cannot be handled well by linear regression
without further work such as feature selection adn regularization.
Moreover, based on our exploratory data analysis, we have decent
confidence that tree-based models can figure out prices based on the
existing features. The primary value of our models is to provide early
Airbnb users with reasonable prices on their listings so that they can
get them up and running quickly and effectively. To maximize this value,
we average the model predictions as an ensemble for the final predicted
prices. For the same reason, we use mean absolute error as our
performance metric instead of mean squared error because we want to
optimize more for median than mean. Mean is more impacted by outliers
than the median, and we think aiming for median rather than mean will
produce more of a reasonable price for potential customers, efficiently
paving the way to the new hosts’ first bookings.

# Our modelling result

We built three models, Random Forest, LightGBM, and XGBoost, and
averaged their predictions to produce the final productions. We derived
all models with some degree of hyperparameter tuning. The Random Forest
and XGBoost regressors both took considerably longer to train than the
LightGBM regressor, and that is why we limited their combinations of
hyperparameters for tuning due to our time constraint.

| train\_mean\_absolute\_error | test\_mean\_absolute\_error |
| ---------------------------: | --------------------------: |
|                        84.32 |                       83.20 |
|                        46.71 |                       65.35 |
|                        53.08 |                       65.45 |
|                        64.05 |                       65.79 |
|                        53.27 |                       63.94 |

Table 1. Mean Absolute Errors

Even though the models have imbalanced hyperparameter tuning, they have
comparable results and are all an obvious upgrade over the median null
model. Moreover, the ensemble has a decent performance improvement over
the base models.

We deem the potential of overfitting to be quite low on all the models
because their training and test errors are close enough to not arise the
suspicion of overfitting. Especially for the ensembled result due to not
relying on a single model, we are confident that our model should be
solid for predicting with new data.

Let us look at the feature importances of different models.

| Random.Forest | XGBoost | LightGBM |
| ------------: | ------: | -------: |
|         0.004 |   0.095 |    0.003 |
|         0.078 |   0.085 |    0.056 |
|         0.171 |   0.080 |    0.156 |
|         0.289 |   0.284 |    0.151 |
|         0.127 |   0.176 |    0.043 |
|         0.072 |   0.054 |    0.181 |
|         0.027 |   0.044 |    0.062 |
|         0.045 |   0.042 |    0.068 |
|         0.086 |   0.097 |    0.130 |
|         0.101 |   0.043 |    0.150 |

Table 2. Feature Importance

Except for longitude, no other feature seems to be consistently
important for all models. For instance, the top three features for the
Random Forest regressor also include latitude and room type, while for
the LightGBM regressor, they are the number of days available per year,
minimum nights and latitude. However, for our goal, interpreting feature
importance is not our focus.

Let us look at the ensemble residuals on the test set.

<img src="../results/plots/ensemble_residual_plot.png" width="916" />
Figure 4. Residuals for average
ensembling

<img src="../results/plots/ensemble_residual_distribution.png" width="900" />
Figure 5. Distribution of residuals average ensembling

<br><br> The residuals look to follow a normal distribution with a few
big residuals spread along the high prices. We do not quite care about
those big residuals over high prices because we intend our model to
estimate the median. In particular, we do not want our model to be
affected or swayed by high prices by too much since pulling the
predictions towards that direction will hinder the listings’ ability to
get bookings quickly and hence discourage new hosts from maintaining
those listings. In conclusion, we believe our ensemble has done its job
effectively.

# Limitations

Our project has a few limitations. First, the data we have does not
capture the season and months of the year. There could be a high
correlation between the prices and the season of the year. In particular
seasons and holidays, we can expect tourist inflows to be much higher
than normal thereby inflating the listing prices. As well, the prices of
Airbnb listings fluctuate throughout the year, which is not captured and
limits us from probing into this further. Finally, a rating feature
would have helped our models to predict the prices better. In the
dataset we are using, we do have reviews but it would require us to do
sentiment analysis on each review to get a pulse of how the Airbnb
listing compares to others.

There are some limitations to our models as well. Due to our time
constraint and long training time with random forest as well as xgboost,
we could not cover a more extended range of hyperparameters, which could
result in better models. Also, since interpretation is not an emphasis
in our result, we could go for a more complicated but powerful
ensembling technique, such as stacking. Moreover, we did not create any
new feature which could potentially improve predictions. For instance,
we could turn the last\_review feature into a new feature for days since
the last review.

# Potential future improvements

If this project were to continue, we had a few upgrades in mind in terms
of data analysis and modeling. Regarding data analysis, we would want to
look at some new features we can generate, such as days before the last
review and whether a listing has low availability. We will explore
whether they have distinct patterns on prices and form our expectations
about their respective importances. We could potentially take them into
account for our models and explore other ensembling techniques, such as
stacking, to see if we could arrive at more accurate predictions.

# References

<div id="refs" class="references">

<div id="ref-gridExtra">

Auguie, Baptiste. 2015. *GridExtra: Miscellaneous Functions for "Grid"
Graphics*. <http://CRAN.R-project.org/package=gridExtra>.

</div>

<div id="ref-chandra2015python">

Chandra, Rakesh Vidya, and Bala Subrahmanyam Varanasi. 2015. *Python
Requests Essentials*. Packt Publishing Ltd.

</div>

<div id="ref-Chen:2016:XST:2939672.2939785">

Chen, Tianqi, and Carlos Guestrin. 2016. “XGBoost: A Scalable Tree
Boosting System.” In *Proceedings of the 22nd Acm Sigkdd International
Conference on Knowledge Discovery and Data Mining*, 785–94. KDD ’16. New
York, NY, USA: ACM. <https://doi.org/10.1145/2939672.2939785>.

</div>

<div id="ref-docopt">

de Jonge, Edwin. 2018. *Docopt: Command-Line Interface Specification
Language*. <https://CRAN.R-project.org/package=docopt>.

</div>

<div id="ref-GettheDa10:online">

“Get the Data - Inside Airbnb. Adding Data to the Debate.” n.d.
<http://insideairbnb.com/get-the-data.html>.

</div>

<div id="ref-NIPS2017_6907">

Ke, Guolin, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma,
Qiwei Ye, and Tie-Yan Liu. 2017. “LightGBM: A Highly Efficient Gradient
Boosting Decision Tree.” In *Advances in Neural Information Processing
Systems 30*, edited by I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach,
R. Fergus, S. Vishwanathan, and R. Garnett, 3146–54. Curran Associates,
Inc.
<http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf>.

</div>

<div id="ref-docoptpython">

Keleshev, Vladimir. 2014. *Docopt: Command-Line Interface Description
Language*. <https://github.com/docopt/docopt>.

</div>

<div id="ref-schema">

———. 2019. *Schema: Validating Python Data Structures*.
<https://github.com/keleshev/schema>.

</div>

<div id="ref-mcgeehan_2018">

Mcgeehan, Patrick. 2019a. “N.Y. Draws a Record 65 Million Tourists (in
Spite of Trump’s Trade War, Many Were Chinese).” *The New York Times*.
The New York Times.
<https://www.nytimes.com/2019/01/16/nyregion/nyc-tourism-record.html>.

</div>

<div id="ref-mcgeehan_2019">

———. 2019b. “N.Y.C. Is on Pace to Draw a Record 67 Million Tourists This
Year.” *The New York Times*. The New York Times.
<https://www.nytimes.com/2019/08/19/nyregion/nyc-tourism.html>.

</div>

<div id="ref-mckinney-proc-scipy-2010">

McKinney, Wes. 2010. “Data Structures for Statistical Computing in
Python.” In *Proceedings of the 9th Python in Science Conference*,
edited by Stéfan van der Walt and Jarrod Millman, 51–56.

</div>

<div id="ref-selenium">

Muthukadan, Baiju. 2019. *Selenium with Python*.
<https://github.com/baijum/selenium-python>.

</div>

<div id="ref-oliphant2006guide">

Oliphant, Travis E. 2006. *A Guide to Numpy*. Vol. 1. Trelgol Publishing
USA.

</div>

<div id="ref-scikit-learn">

Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O.
Grisel, M. Blondel, et al. 2011. “Scikit-Learn: Machine Learning in
Python.” *Journal of Machine Learning Research* 12: 2825–30.

</div>

<div id="ref-R">

R Core Team. 2019. *R: A Language and Environment for Statistical
Computing*. Vienna, Austria: R Foundation for Statistical Computing.
<https://www.R-project.org/>.

</div>

<div id="ref-sherwood_2019">

Sherwood, Harriet. 2019. “How Airbnb Took over the World.” *The
Guardian*. Guardian News; Media.
<https://www.theguardian.com/technology/2019/may/05/airbnb-homelessness-renting-housing-accommodation-social-policy-cities-travel-leisure>.

</div>

<div id="ref-2018-altair">

Sievert, Jacob VanderPlas AND Brian E. Granger AND Jeffrey Heer AND
Dominik Moritz AND Kanit Wongsuphasawat AND Arvind Satyanarayan AND
Eitan Lees AND Ilia Timofeev AND Ben Welsh AND Scott. 2018. “Altair:
Interactive Statistical Visualizations for Python.” *The Journal of Open
Source Software* 3 (32). <http://idl.cs.washington.edu/papers/altair>.

</div>

<div id="ref-Python">

Van Rossum, Guido, and Fred L. Drake. 2009. *Python 3 Reference Manual*.
Scotts Valley, CA: CreateSpace.

</div>

<div id="ref-reshape2">

Wickham, Hadley. 2007. “Reshaping Data with the reshape Package.”
*Journal of Statistical Software* 21 (12): 1–20.
<http://www.jstatsoft.org/v21/i12/>.

</div>

<div id="ref-tidyverse">

———. 2017. *Tidyverse: Easily Install and Load the ’Tidyverse’*.
<https://CRAN.R-project.org/package=tidyverse>.

</div>

<div id="ref-knitr">

Xie, Yihui. 2014. “Knitr: A Comprehensive Tool for Reproducible Research
in R.” In *Implementing Reproducible Computational Research*, edited by
Victoria Stodden, Friedrich Leisch, and Roger D. Peng. Chapman;
Hall/CRC. <http://www.crcpress.com/product/isbn/9781466561595>.

</div>

</div>
