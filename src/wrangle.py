# author: Ofer Mansour, Jacky Ho, Anand Vemparala
# date: 2020-01-23

'''
Split the data into train and test after removing columns not required and imputing data.

Usage: wrangle.py --source_file_location=<source_file_location> --target_location=<target_location>

source_file_location - a path/filename pointing to the data to be read in
target_location - a path/filename pointing to where the cleaned/processed/transformed/paritioned data should live

'''

import pandas as pd
import numpy as np
from docopt import docopt
from sklearn.model_selection import train_test_split
import os.path


opt = docopt(__doc__)

def main(source_file, target_location):
  # read data
  airbnb_ny = pd.read_csv(source_file)

  # drop features
  airbnb_ny.drop(['id','name', 'host_id', 'host_name','last_review'], axis=1, inplace=True)
  
  # fill nas in reviews per month
  airbnb_ny = airbnb_ny.fillna({'reviews_per_month':0})
  
  # split to X and Y
  X = airbnb_ny.drop(['price'], axis=1)
  y = airbnb_ny.price
  
  # split to test and train
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
  
  # combine X and y for test and train respectively
  full_train = pd.concat((X_train, y_train), axis= 1)
  full_test = pd.concat((X_test, y_test), axis= 1)
  
  
  train_file = opt["--target_location"] + "/train.csv"
  test_file = opt["--target_location"] + "/test.csv"
  
  full_train.to_csv(train_file,header=True, index=False)
  full_test.to_csv(test_file)
  
def are_files_created(filedirectory):
        
    train_file = filedirectory + "/train.csv"
    test_file = filedirectory + "/test.csv"
    
    if os.path.isfile(train_file) and os.path.isfile(test_file):
        return True
    else:
        return False
        
def test_file_created():
    filedirectory = opt['--target_location']
    assert are_files_created(filedirectory) == True, "Training and testing files were not generated, please try again"
       
  
if __name__ == "__main__":
  source_file = opt["--source_file_location"]
  target_location = opt["--target_location"]
  main(source_file, target_location)
  test_file_created()
