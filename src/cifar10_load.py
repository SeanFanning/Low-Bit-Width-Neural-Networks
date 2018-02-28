from __future__ import division, print_function, absolute_import
#mechanism to dynamically include the relative path where utils.ipynb is housed to the module search path.
from inspect import getsourcefile
import os
import os.path
import sys
import time
import re
import pickle
import urllib, tarfile
current_path = os.path.abspath(getsourcefile(lambda:0))
parent_dir = os.path.split(os.path.dirname(current_path))[0]
sys.path.insert(0, parent_dir)


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def maybe_download_and_extract(dest_directory):
  """Download and extract the tarball from Alex's website."""
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.urlretrieve(DATA_URL, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  with tarfile.open(filepath, 'r:gz') as t:
    dataset_dir = os.path.join(dest_directory, t.getmembers()[0].name)
    t.extractall(dest_directory)

  return dataset_dir


dataset_dir = maybe_download_and_extract('/tmp/cifar_data/data')
print(dataset_dir)