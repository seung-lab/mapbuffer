"""
Serializable map of integers to bytes with near zero parsing.

MapBuffer is designed to allow you to store dictionaries mapping 
integers to binary buffers in a serialized format and then read 
that back in and use it without requiring an expensive parse of 
the entire dictionary. Instead, if you have a dictionary containing 
thousands of keys, but only need a few items from it, you can 
extract them rapidly.

Simple Example:

  from mapbuffer import MapBuffer

  data = { 2848: b'abc', 12939: b'123' }
  mb = MapBuffer(data)

  with open("data.mb", "wb") as f:
    f.write(mb.tobytes())

  with open("data.mb", "rb") as f:
    binary = f.read()

  mb = MapBuffer(binary)
  print(mb[2848]) # fast: almost zero parsing required

  >>> b'abc'
"""

import numpy as np

class MapBuffer:
  """Represents a usable int->bytes dictionary as a byte string."""
  def __init__(
    self, data=None, dtype=np.int64, 
    tobytesfn=None, frombytesfn=None
  ):
    """
    data: dict (int->byte serializable object) or bytes 
      (representing a MapBuffer)
    tobytesfn: function for converting dict values to byte strings
      if they are not already.
        e.g. lambda mystr: mystr.encode("utf8")
    frombytesfn: function for converting serialized byte 
      representations of Python objects back into a Python 
      object to simplify accessing values. 
        e.g. lambda mystr: mystr.decode("utf8")
    """
    self.tobytesfn = tobytesfn
    self.frombytesfn = frombytesfn
    self.dtype = dtype
    self.buffer = None

    if isinstance(data, dict):
      self.buffer = self.dict2buf(data)
    elif isinstance(data, bytes):
      self.buffer = data
    else:
      raise TypeError("data must be a dict or bytes. Got: " + str(type(dict)))

  def __len__(self):
    """Returns number of keys."""
    return int.from_bytes(self.buffer[:4], byteorder="little", signed=False)

  def __iter__(self):
    yield from self.keys()

  def datasize(self):
    """Returns size of data region in bytes."""
    return len(self.buffer) - 4 - len(self) * 2 * 8

  def index(self):
    """Get an Nx2 numpy array representing the index."""
    N = len(self)
    index_length = 2 * N * 8
    index = self.buffer[4:index_length+4]
    return np.frombuffer(index, dtype=np.uint64).reshape((N,2))

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
      label = index[2*i]
      value = self.getindex(i)
      yield (label, value)

  def getindex(self, i):
    index = self.index()
    N = index.shape[0]
    offset = index[i,1]
    if i < N - 1:
      next_offset = index[i+1,1]
      value = self.buffer[offset:next_offset]
    else:
      value = self.buffer[offset:]

    if self.frombytesfn:
      value = self.frombytesfn(value)

    return value  

  def __getitem__(self, label):
    index = self.index()
    N = len(index)
    if N == 0:
      return None

    first, last = 0, N
    count = N
    while count > 0:
      i = first
      step = count // 2
      i += step
      if index[i,0] < label:
        i += 1
        first = i
        count -= step + 1
      else:
        count = step

    if first < N and label == index[first,0]:
      return self.getindex(first)
    
    raise KeyError("{} was not found.".format(label))

  def dict2buf(self, data, tobytesfn=None):
    """Structure [ index length, sorted index, data ]"""
    labels = np.array([ int(lbl) for lbl in data.keys() ], dtype=self.dtype)
    labels.sort()
    N = len(labels)
    N_region = N.to_bytes(4, byteorder="little", signed=False)

    if N == 0:
      return N_region

    index_length = 2 * N
    index = np.zeros((index_length,), dtype=self.dtype)
    index[::2] = labels

    noop = lambda x: x
    tobytesfn = nvl(tobytesfn, self.tobytesfn, noop)
    
    data_region = b"".join(( tobytesfn(data[label]) for label in labels ))
    index[1] = 4 + index_length * 8
    for i, label in zip(range(1, len(labels)), labels):
      index[i*2 + 1] = index[(i-1)*2 + 1] + len(data[labels[i-1]])

    return N_region + index.tobytes() + data_region

  def todict(self):
    return { label: val for label, val in self.items() }

  def tobytes(self):
    return self.buffer

  def validate(self):
    return self.validate_buffer(self.buffer)

  @staticmethod
  def validate_buffer(buf):
    mapbuf = MapBuffer(buf)
    index = mapbuf.index()
    if len(index) != len(mapbuf):
      return False

    offsets = index[:,1].astype(np.int64)
    lengths = offsets[1:] - offsets[0:-1]
    if np.any(lengths < 0):
      return False
    if lengths.sum() + (len(buf) - offsets[-1]) != mapbuf.datasize():
      return False

    labels = index[:,0].astype(np.int64)
    labeldiff = labels[1:] - labels[0:-1]
    if np.any(labeldiff < 1):
      return False

    return True
    
def nvl(*args):
  """Return the leftmost argument that is not None."""
  if len(args) < 2:
    raise IndexError("nvl takes at least two arguments.")
  for arg in args:
    if arg is not None:
      return arg
  return args[-1]

# x = MapBuffer({ 1: b'123', 5: b'456', 4: b'789' }, frombytesfn=lambda x: int(x))
# print(x.buffer)
# print(list(x.keys()))
# print(x[5])

