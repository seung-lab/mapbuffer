import pytest

import mmap
import os
import random

import numpy as np

from mapbuffer import ValidationError, IntMap, MapBuffer, HEADER_LENGTH

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
def test_crc32c(compress):
  data = { 
    1: b"hello",
    2: b"world",
  }
  mbuf = MapBuffer(data, compress=compress)

  idx = mbuf.buffer.index(b"hello")
  buf = list(mbuf.buffer)
  buf[idx] = ord(b'H')
  mbuf.buffer = bytes(buf)

  try:
    mbuf[1]
    assert False
  except ValidationError:
    pass

@pytest.mark.parametrize("compress", (None, "gzip", "br", "zstd"))
def test_mmap_access(compress):
  data = { 
    1: b"hello",
    2: b"world",
  }
  mbuf = MapBuffer(data, compress=compress)

  fileno = random.randint(0,2**32)
  filename = f"test_mmap-{fileno}.mb"

  with open(filename, "wb") as f:
    f.write(mbuf.tobytes())

  with open(filename, "rb") as f:
    mb = MapBuffer(f)

    assert mb[1] == b"hello"
    assert mb[2] == b"world"

  try:
    os.remove(filename)
  except (PermissionError, FileNotFoundError):
    pass

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

# def test_set_object():
#   data = { 
#     1: b"hello",
#     2: b"world",
#   }
#   mbuf = MapBuffer(data, compress=None, check_crc=False)
#   mbuf = MapBuffer(bytearray(mbuf.buffer), compress=None, check_crc=False)

#   assert mbuf[1] == b"hello"

#   mbuf[1] = b"abcde"
#   assert mbuf[1] == b"abcde"

#   try:
#     mbuf[2] = b'abcdefg'
#     assert False
#   except ValueError:
#     pass

#   try:
#     mbuf[9] = b'123'
#   except KeyError:
#     pass

def test_empty_intmap():
  im = MapBuffer({})
  assert len(im) == 0
  assert list(im) == []

  assert im.validate()

  try:
    im[1000]
    assert False
  except KeyError:
    pass

def test_full_intmap():
  data = { 
    random.randint(0, 1000000000): random.randint(0,1000000000) for _ in range(10000) 
  }

  im = IntMap(data)
  assert set(data.keys()) == set(im.keys())
  assert set(data) == set(im)
  assert set(data.values()) == set(im.values())

  for key in data:
    assert data[key] == im[key]
    assert data[key] == im.get(key)
    assert key in im

  assert data == im.todict()

  for i in range(2000):
    if i not in data:
      assert i not in im
      try:
        im[i]
        assert False
      except KeyError:
        pass

  im.validate()

  assert len(im.buffer) > HEADER_LENGTH

def test_mmap_access_intmap():
  data = { 
    1: 3,
    2: 4,
  }
  im = IntMap(data)

  fileno = random.randint(0,2**32)
  filename = f"test_mmap-{fileno}.im"

  with open(filename, "wb") as f:
    f.write(im.tobytes())

  with open(filename, "rb") as f:
    im = IntMap(f)

    assert im[1] == 3
    assert im[2] == 4

  try:
    os.remove(filename)
  except (PermissionError, FileNotFoundError):
    pass

def test_object_access_intmap():
  data = { 
    1: 3,
    2: 4,
  }
  mbuf = IntMap(data)

  class Reader:
    def __init__(self):
      self.lst = mbuf.tobytes()
    def __getitem__(self, slc):
      return self.lst[slc]

  mbuf2 = IntMap(Reader())
  assert mbuf2[1] == 3
  assert mbuf2[2] == 4

def test_set_object_intmap():
  data = { 
    1: 5,
    2: 7,
  }
  mbuf = IntMap(data)
  mbuf.buffer = bytearray(mbuf.buffer)

  assert mbuf[1] == 5

  mbuf[1] = 8
  assert mbuf[1] == 8

  try:
    mbuf[9] = b'123'
  except KeyError:
    pass




