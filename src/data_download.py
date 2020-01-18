'''
Download and save a csv file to a target location

Usage: data_download.py --url=<url> --file_location=<file_location>
 
'''

import requests
from docopt import docopt

opt = docopt(__doc__)

def main(url, file_location):
    # download data from url and save to file_location
    r = requests.get(url)
    with open(file_location, "wb") as f:
        f.write(r.content) 

if __name__ == "__main__":
    main(opt["--url"], opt["--file_location"])
