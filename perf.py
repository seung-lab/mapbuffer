import pytest
import numpy as np
from mapbuffer import MapBuffer, HEADER_LENGTH
import random
import time

import pickle

pf = open("pkl.tsv", "a")
mf = open("mb.tsv", "a")

def mkdataset(datasize):
  return { 
    random.randint(0, 1000000000): np.random.bytes(random.randint(0,50000)) 
    for _ in range(datasize) 
  }

def test_pkl(data):
  datasize = len(data)
  labels = list(data.keys())
  random.shuffle(labels)
  labels = labels[:datasize//10]

  pkl = pickle.dumps(data)
  s = time.time()
  dat = pickle.loads(pkl)
  for label in labels:
    dat[label]
  t = time.time() - s

  pf.write(f"{datasize}\t{t*1000:.5f}\n")
  pf.flush()


def test_mb(data):
  datasize = len(data)
  labels = list(data.keys())
  random.shuffle(labels)
  labels = labels[:datasize//10]

  mbuf = MapBuffer(data)
  buf = mbuf.tobytes()

  s = time.time()
  mbuf = MapBuffer(buf)
  for label in labels:
    mbuf[label]
  t = time.time() - s 
  mf.write(f"{datasize}\t{t*1000:.5f}\n")
  mf.flush()

sz = 1
while sz < 1000000:
  data = mkdataset(sz)
  # test_pkl(data)
  test_mb(data)
  sz *= 2

mf.close()
pf.close()