import pytest

import mmap
import os
import random
from unittest.mock import patch

import numpy as np

from mapbuffer import ValidationError, IntMap, MapBuffer, HEADER_LENGTH

CACHE_PATH = "./test_index_cache.mbi"

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
@pytest.mark.parametrize("compute_crc", [True, False])
def test_full(compress, compute_crc):
  data = { 
    random.randint(0, 1000000000): bytes([ 
      random.randint(0,255) for __ in range(random.randint(0,50)) 
    ]) for _ in range(10000) 
  }
  mbuf = MapBuffer(data, compress=compress, compute_crc=compute_crc)
  assert set(data.keys()) == set(mbuf.keys())
  assert set(data) == set(mbuf)
  assert set(data.values()) == set(( bytes(x) for x in mbuf.values()))

  for key in data:
    assert data[key] == mbuf[key]
    assert data[key] == mbuf.get(key)
    assert key in mbuf

    if compress is None:
      assert len(data[key]) == mbuf.size(key)

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

@pytest.fixture(autouse=True)
def cleanup_cache():
    """Ensure cache file is removed before and after each test."""
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
    yield
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)


def make_mapbuffer(data=None, **kwargs):
    data = data or {1: b"hello", 2: b"world"}
    return MapBuffer(data, index_cache=CACHE_PATH, **kwargs)


def test_index_cache_file_is_created():
    """Cache file should be written after first access."""
    mbuf = make_mapbuffer()
    mbuf.index()
    assert os.path.exists(CACHE_PATH)


def test_index_cache_header_and_index_written():
    """Cache file should contain header + full index bytes."""
    mbuf = make_mapbuffer()
    index = mbuf.index()
    
    with open(CACHE_PATH, "rb") as f:
        cached = f.read()

    assert len(cached) == HEADER_LENGTH + index.nbytes


def test_index_cache_is_loaded_from_disk():
    """Second MapBuffer with same cache should read index from disk, not buffer."""
    mbuf = make_mapbuffer()
    original_index = mbuf.index().copy()

    # Reload — this time the cache exists, so index should come from disk
    mbuf2 = make_mapbuffer()
    mbuf2._index = None  # ensure not inherited

    with patch.object(np, "frombuffer", wraps=np.frombuffer) as mock_frombuffer:
        loaded_index = mbuf2.index()
        # np.frombuffer should NOT be called on the main buffer for the index
        for call in mock_frombuffer.call_args_list:
            args, kwargs = call
            # Ensure we're not reading index from the primary buffer
            assert kwargs.get("offset") != HEADER_LENGTH, \
                "Index was re-read from buffer instead of cache"

    np.testing.assert_array_equal(loaded_index, original_index)


def test_index_cache_values_correct():
    """Values retrieved using cache should match those from a non-cached buffer."""
    mbuf_cached = make_mapbuffer()
    mbuf_plain = MapBuffer({1: b"hello", 2: b"world"})

    for key in [1, 2]:
        assert mbuf_cached[key] == mbuf_plain[key]


def test_crc_error_raised_despite_cache():
    """CRC validation should still catch corruption even when cache exists."""
    data = {1: b"hello", 2: b"world"}
    mbuf = make_mapbuffer(data)
    mbuf.index()  # populate cache

    # Corrupt the data region in the buffer
    buf = bytearray(mbuf.buffer)
    idx = bytes(buf).index(b"hello")
    buf[idx] = ord(b"H")
    mbuf.buffer = bytes(buf)
    mbuf._index = None  # force re-read so cache is used but data is still corrupt

    with pytest.raises(ValidationError):
        mbuf[1]


def test_index_cache_not_rewritten_if_already_complete():
    """Cache file should not be overwritten on second load."""
    mbuf = make_mapbuffer()
    mbuf.index()

    mtime_after_first = os.path.getmtime(CACHE_PATH)

    mbuf2 = make_mapbuffer()
    mbuf2.index()

    mtime_after_second = os.path.getmtime(CACHE_PATH)
    assert mtime_after_first == mtime_after_second, \
        "Cache file was unexpectedly rewritten on second access"