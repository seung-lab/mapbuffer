import mmap 
import io
import itertools

from .exceptions import ValidationError
from .lib import eytzinger_sort

import numpy as np

import mapbufferaccel

class IntMap:
  """Represents a usable int->int dictionary."""
  MAGIC_NUMBER = b"intm"
  FORMAT_VERSION = 0
  HEADER_LENGTH = 10
  def __init__(self, data):
    """
    data: dict (int->byte serializable object) or bytes 
      (representing an IntMap)
    """
    self.dtype = np.uint64
    self.buffer = None

    self._header = None
    self._index = None

    if isinstance(data, dict):
      self.buffer = self.dict2buf(data)
    elif isinstance(data, io.IOBase):
      self.buffer = mmap.mmap(data.fileno(), 0, access=mmap.ACCESS_READ)
    elif isinstance(data, (bytes, bytearray, mmap.mmap)):
      self.buffer = data
    elif hasattr(data, "__getitem__"):
      self.buffer = data
    else:
      raise TypeError(
        f"data must be a dict, bytes, file, mmap, or otherwise support "
        f"__getitem__ with slice support for byte ranges. Got: {type(data)}"
      )

  def __len__(self):
    """Returns number of keys."""
    return int.from_bytes(self.header[6:10], byteorder="little", signed=False)

  @property
  def format_version(self):
    return self.header[len(self.MAGIC_NUMBER)]

  def __iter__(self):
    yield from self.keys()

  @property
  def header(self):
    """Get the header bytes."""
    if self._header is not None:
      return self._header

    # seems dumb, buf if self.buffer is an object that
    # requires network access, this is a valuable cache
    self._header = self.buffer[:self.HEADER_LENGTH]
    return self._header

  def index(self):
    """Get an Nx2 numpy array representing the index."""
    if self._index is not None:
      return self._index

    N = len(self)
    index_length = 2 * N * 8
    index = self.buffer[self.HEADER_LENGTH:index_length+self.HEADER_LENGTH]
    self._index = np.frombuffer(index, dtype=np.uint64).reshape((N,2))
    return self._index

  def keys(self):
    for label, offset in self.index():
      yield label

  def values(self):
    for label, value in self.items():
      yield value

  def items(self):
    N = len(self)
    index = self.index()
    for i in range(N):
      label = index[i,0]
      value = index[i,1]
      yield (label, value)

  def getindex(self, i):
    index = self.index()
    return index[i,1]

  def setindex(self, i, ct):
    index = self.index()
    N = index.shape[0]
    index[i,1] = ct

  def find_index_position(self, label):
    index = self.index()
    N = len(index)
    if N == 0:
      return None

    k = mapbufferaccel.eytzinger_binary_search(label, index)
    if k >= 0 and k < N:
      return k

    return None

  def get(self, label, *args, **kwargs):
    pos = self.find_index_position(label)
    if pos is None: # try to get default argument
      try:
        return args[0]
      except IndexError:
        try:
          return kwargs.get("default")
        except KeyError:
          raise KeyError("{} was not found.".format(label))
    
    return self.getindex(pos)

  def __contains__(self, label):
    pos = self.find_index_position(label)
    return pos is not None

  def __getitem__(self, label):
    pos = self.find_index_position(label)
    if pos is not None:
      return self.getindex(pos)
    else:
      raise KeyError("{} was not found.".format(label))

  def __setitem__(self, label, ct):
    pos = self.find_index_position(label)
    if pos is not None:
      return self.setindex(pos, ct)
    else:
      raise KeyError("{} was not found.".format(label))    

  def dict2buf(self, data):
    """Structure [ index length, sorted index, data ]"""
    labels = np.fromiter(
      ( 
        int(lbl) for lbl in itertools.chain.from_iterable(data.items()) 
      ), 
      count=len(data) * 2, 
      dtype=self.dtype
    )
    labels = labels.reshape((len(data),2), order="C")
    labels.view("<u8,<u8").sort(order=["f0"], axis=0)

    layout = mapbufferaccel.eytzinger_sort_indices(len(data))
    labels = labels[layout]

    N = len(data)
    N_region = N.to_bytes(4, byteorder="little", signed=False)

    header = (
      self.MAGIC_NUMBER 
      + bytes([ self.FORMAT_VERSION ])
      + int(3).to_bytes(1, byteorder="little", signed=False) # uint64
      + N_region
    )

    if N == 0:
      return header
    
    return b"".join([ header, labels.tobytes('C') ])

  def todict(self):
    return { label: val for label, val in self.items() }

  def tobytes(self):
    return self.buffer

  def validate(self):
    header = self.header
    index = self.index()
    if len(index) != len(self):
      raise ValidationError(f"Index size doesn't match. len(self): {len(self)}")

    magic = header[:len(self.MAGIC_NUMBER)]
    if magic != self.MAGIC_NUMBER:
      raise ValidationError(f"Magic number mismatch. Expected: {self.MAGIC_NUMBER} Got: {magic}")

    if self.format_version not in (0,):
      raise ValidationError(f"Unsupported format version. Got: {self.format_version}")

    return True
