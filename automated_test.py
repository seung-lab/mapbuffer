import pytest
import numpy as np
from mapbuffer import MapBuffer, HEADER_LENGTH
import random

@pytest.mark.parametrize("compress", (None, "gzip", "br", "zstd", "lzma"))
def test_empty(compress):
  mbuf = MapBuffer({}, compress=compress)
  assert len(mbuf) == 0
  assert list(mbuf) == []

  assert mbuf.validate()
  assert mbuf.compress == compress

@pytest.mark.parametrize("compress", (None, "gzip", "br", "zstd"))
def test_full(compress):
  data = { 
    random.randint(0, 1000000000): bytes([ 
      random.randint(0,255) for __ in range(random.randint(0,50)) 
    ]) for _ in range(10000) 
  }
  mbuf = MapBuffer(data, compress=compress)
  assert set(data.keys()) == set(mbuf.keys())
  assert set(data) == set(mbuf)
  assert set(data.values()) == set(mbuf.values())

  for key in data:
    assert data[key] == mbuf[key]

  assert data == mbuf.todict()

  # mbuf.validate()

  assert len(mbuf.buffer) > HEADER_LENGTH