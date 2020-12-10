import pytest
import numpy as np
from mapbuffer import MapBuffer, HEADER_LENGTH
import random
import time

import pickle

pf = open("pkl.tsv", "w+")
mf = open("mb.tsv", "w+")

def test(datasize):
  data = { 
    random.randint(0, 1000000000): np.random.bytes(random.randint(0,100000)) 
    for _ in range(datasize) 
  }

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

  mbuf = MapBuffer(data)
  buf = mbuf.tobytes()

  s = time.time()
  mbuf = MapBuffer(buf)
  for label in labels:
    mbuf[label]
  t = time.time() - s 
  mf.write(f"{datasize}\t{t*1000:.5f}\n")
  mf.flush()

for i in range(1,10000,50):
  test(i)

mf.close()
pf.close()