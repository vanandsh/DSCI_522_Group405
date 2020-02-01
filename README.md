
# New York Airbnb Price Predictor

  - authors: Ofer Mansour, Jacky Ho, Anand Vemparala

## About

In this project, our team attempted to predict Airbnb prices in New
York, using a [dataset](http://insideairbnb.com/get-the-data.html)(“Get
the Data - Inside Airbnb. Adding Data to the Debate.” n.d.) of all
listings in 2019. The dataset includes 48,895 listings and includes
several features, such as the price of the listing, neighbourhood, room
type and number of reviews. To find the best performing prediction
model, we tested several tree-based models: Random Forest, XGBoost,
LightGBM and an average ensembling of the three models.

## Report

The final report can be found
[here](https://github.com/UBC-MDS/DSCI_522_Group405/blob/master/doc/final_report.md).

## Usage

To replicate this project, clone this GitHub repository, install the
dependencies and run the command below at the command line or terminal
from the root directory of this project:

    make all

To clean this repository to have no result or intermediate files, run
the command below at the command line or terminal from the root
directory of this project:

    make clean

## Dependencies

  - Python 3.7.3 and Python Packages:
      - docopt==0.6.2  
      - requests==2.22.0  
      - pandas==0.24.2  
      - numpy==1.16.4  
      - sklearn==0.21.2  
      - lightgbm==2.3.1  
      - xgboost==0.90  
      - altair==3.2.0
      - os==3.7.3
      - schema==0.7.1
  - R version 3.6.2 and R packages:
      - docopt==0.6.1  
      - tidyverse==1.2.1  
      - GGally==1.4.0  
      - gridExtra==2.3  
      - knitr==1.27.2  
  - GNU Make 3.81

# References

<div id="refs" class="references">

<div id="ref-GettheDa10:online">

“Get the Data - Inside Airbnb. Adding Data to the Debate.” n.d.
<http://insideairbnb.com/get-the-data.html>.

</div>

</div>
