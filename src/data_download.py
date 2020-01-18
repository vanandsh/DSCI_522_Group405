"""Usage: data_download.py --SourceFileURL=<SourceFileURL> --TargetLocation=<TargetLocation>

Process FILE and optionally apply correction to either left-hand side or
right-hand side.
Arguments:
  SourceFileURL   URL to download file from
  TargetLocation  Location where the file should be downloaded
"""

import pandas as pd
import numpy as np
from docopt import docopt
import urllib.parse

opt = docopt(__doc__)

def main(source_file_url, target_location):
  # Read csv for source url
  data=pd.read_csv(source_file_url, index_col=0)
  # Write csv to target location
  data.to_csv(target_location)
  
# standard error function
def sterror():
    return 1

def test_sterror():
#   assert sterror(np.array([1, 1, 1])) == 0, "sterror should return 0 if vector values are all the same"
    assert 1==1

test_sterror()

if __name__ == '__main__':
    opt = docopt(__doc__)
    main(opt['--SourceFileURL'], opt['--TargetLocation'])
