import pytest
import numpy as np
from mapbuffer import MapBuffer, MapInt
import random

@pytest.mark.parametrize("compress", (None, "gzip", "br", "zstd", "lzma"))
def test_empty(compress):
  mbuf = MapBuffer({}, compress=compress)
  assert len(mbuf) == 0
  assert list(mbuf) == []

  assert mbuf.validate()
  assert mbuf.compress == compress

  try:
    mbuf[1000]
    assert False
  except KeyError:
    pass


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
    assert data[key] == mbuf.get(key)
    assert key in mbuf

  assert data == mbuf.todict()

  for i in range(2000):
    if i not in data:
      assert i not in mbuf
      try:
        mbuf[i]
        assert False
      except KeyError:
        pass

  mbuf.validate()

  assert len(mbuf.buffer) > MapBuffer.HEADER_LENGTH

def test_mapint():
  N = 20
  data = { i:i+1 for i in range(N) }

  mi = MapInt(data)

  for i in range(N):
    assert mi[i] == data[i]

  try:
    mi[N]
    assert False
  except KeyError:
    pass





