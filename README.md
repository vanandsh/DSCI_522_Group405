
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
dependencies and run the commands below at the command line or terminal
from the root directory of this project:

    # download dataset
    python src/data_download.py --url="https://raw.githubusercontent.com/vanandsh/datasets/master/AB_NYC_2019.csv" --file_location="./data/raw_data.csv"
    
    # wrangle data 
    python src/wrangle.py --source_file_location="./data/raw_data.csv" --target_location="./data"
    
    # run eda 
    Rscript src/eda_script.R --source_file="./data/train.csv" --target_location="./results"
    
    # run model
    python src/model.py --source_file_location="./data" --target_location="./results"
    
    # knit final report
    Rscript src/knit_rmd.R --source_file="./doc/final_report.Rmd"

## Dependencies

  - Python and Python Packages
      - docopt
      - requests
      - pandas
      - numpy
      - sklearn
      - lightgbm
      - xgboost
      - altair
      - os
  - R
      - docopt
      - tidyverse
      - GGally
      - gridExtra
      - knitr

# References

<div id="refs" class="references">

<div id="ref-GettheDa10:online">

“Get the Data - Inside Airbnb. Adding Data to the Debate.” n.d.
<http://insideairbnb.com/get-the-data.html>.

</div>

</div>
