import pytest
import numpy as np
from mapbuffer import MapBuffer, HEADER_LENGTH
import random
import mmap

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

  assert len(mbuf.buffer) > HEADER_LENGTH

@pytest.mark.parametrize("compress", (None, "gzip", "br", "zstd"))
def test_mmap_access(compress):
  data = { 
    1: b"hello",
    2: b"world",
  }
  mbuf = MapBuffer(data, compress=compress)
  with open("test_mmap.mb", "wb") as f:
    f.write(mbuf.tobytes())

  with open("test_mmap.mb", "rb") as f:
    mb = MapBuffer(f)

    assert mb[1] == b"hello"
    assert mb[2] == b"world"

@pytest.mark.parametrize("compress", (None, "gzip", "br", "zstd"))
def test_object_access(compress):
  data = { 
    1: b"hello",
    2: b"world",
  }
  mbuf = MapBuffer(data, compress=compress)

  class Reader:
    def __init__(self):
      self.lst = mbuf.tobytes()
    def __getitem__(self, slc):
      return self.lst[slc]

  mbuf2 = MapBuffer(Reader())
  assert mbuf2[1] == b"hello"
  assert mbuf2[2] == b"world"








