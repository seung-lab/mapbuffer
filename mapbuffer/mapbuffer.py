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
from .exceptions import ValidationError
from .lib import nvl
from . import compression

import numpy as np

FORMAT_VERSION = 0
MAGIC_NUMBERS = b"mapbufr"
HEADER_LENGTH = 16

class MapBuffer:
  """Represents a usable int->bytes dictionary as a byte string."""
  __slots__ = (
    "data", "tobytesfn", "frombytesfn", 
    "dtype", "buffer", "_index", "_compress"
  )
  def __init__(
    self, data=None, compress=None,
    tobytesfn=None, frombytesfn=None
  ):
    """
    data: dict (int->byte serializable object) or bytes 
      (representing a MapBuffer)
    compress: string representing a valid compression type or None
      Valid: "gzip", "br", "zstd", "lzma"
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
    self.dtype = np.uint64
    self.buffer = None

    self._index = None
    self._compress = None

    if isinstance(data, dict):
      self.buffer = self.dict2buf(data, compress)
    elif isinstance(data, bytes):
      self.buffer = data
    else:
      raise TypeError("data must be a dict or bytes. Got: " + str(type(dict)))

  def __len__(self):
    """Returns number of keys."""
    return int.from_bytes(self.buffer[12:16], byteorder="little", signed=False)

  @property
  def compress(self):
    if self._compress:
      return self._compress

    self._compress = compression.normalize_encoding(
      self.buffer[8:12]
    )
    return self._compress

  @property
  def format_version(self):
    return self.buffer[len(MAGIC_NUMBERS)]

  def __iter__(self):
    yield from self.keys()

  def datasize(self):
    """Returns size of data region in bytes."""
    return len(self.buffer) - HEADER_LENGTH - len(self) * 2 * 8

  def index(self):
    """Get an Nx2 numpy array representing the index."""
    if self._index is not None:
      return self._index

    N = len(self)
    index_length = 2 * N * 8
    index = self.buffer[HEADER_LENGTH:index_length+HEADER_LENGTH]
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

    encoding = self.compress
    if encoding:
      value = compression.decompress(value, encoding, str(index[i,0]))

    if self.frombytesfn:
      value = self.frombytesfn(value)

    return value  

  def __getitem__(self, label):
    index = self.index()
    N = np.uint64(len(index))
    if N == 0:
      return None

    # Important for speed to ensure all types match
    # however the numpy type is fickle and keeps
    # converting itself to a float64 if not everything
    # is a matching type.
    label = np.uint64(label)

    # Cache aware Binary search using eytzinger ordering
    # not necessarily faster in Python (1.5x slower?), but
    # leaves the door open for C/C++ implementations.
    # Since this is a format, if we don't support it from
    # the start, it'll never happen without headaches.
    k = np.uint64(1)
    one = np.uint64(1)
    while k <= N:
      k = (k << one) + (index[(k-one),0] < label)
    k >>= np.uint64(ffs(~k))
    k -= one

    if k < N and label == index[k,0]:
      return self.getindex(int(k))
    
    raise KeyError("{} was not found.".format(label))

  def dict2buf(self, data, compress=None, tobytesfn=None):
    """Structure [ index length, sorted index, data ]"""
    labels = np.array([ int(lbl) for lbl in data.keys() ], dtype=self.dtype)
    labels.sort()

    out = np.zeros((len(labels),), dtype=np.uint64)
    eytzinger_sort(labels, out)
    labels = out

    N = len(labels)
    N_region = N.to_bytes(4, byteorder="little", signed=False)

    compress = compression.normalize_encoding(compress)
    compress_header = nvl(compress, "none")

    header = (
      MAGIC_NUMBERS + bytes([ FORMAT_VERSION ]) 
      + compress_header.zfill(4).encode("ascii") 
      + N_region
    )

    if N == 0:
      return header

    index_length = 2 * N
    index = np.zeros((index_length,), dtype=self.dtype)
    index[::2] = labels

    noop = lambda x: x
    tobytesfn = nvl(tobytesfn, self.tobytesfn, noop)

    bytes_data = { 
      label: compression.compress(tobytesfn(val), method=compress) 
      for label, val in data.items()
    }

    data_region = b"".join(
      ( bytes_data[label] for label in labels )
    )
    index[1] = HEADER_LENGTH + index_length * 8
    for i, label in zip(range(1, len(labels)), labels):
      index[i*2 + 1] = index[(i-1)*2 + 1] + len(bytes_data[labels[i-1]])

    return header + index.tobytes() + data_region

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
      raise ValidationError(f"Index size doesn't match. len(mapbuf): {len(mapbuf)}")

    magic = buf[:len(MAGIC_NUMBERS)]
    if magic != MAGIC_NUMBERS:
      raise ValidationError(f"Magic number mismatch. Expected: {MAGIC_NUMBERS} Got: {magic}")

    if mapbuf.format_version not in (0,):
      raise ValidationError(f"Unsupported format version. Got: {mapbuf.format_version}")

    if mapbuf.compress not in compression.COMPRESSION_TYPES:
      raise ValidationError(f"Unsupported compression format. Got: {mapbuf.compress}")

    if len(mapbuf) > 0:
      offsets = index[:,1].astype(np.int64)
      lengths = offsets[1:] - offsets[0:-1]
      if np.any(lengths < 0):
        raise ValidationError("Offsets are not sorted.")

      length = lengths.sum() + (len(buf) - offsets[-1])
      if length != mapbuf.datasize():
        raise ValidationError(f"Data length doesn't match offsets. Predicted: {length} Data Size: {mapbuf.datasize()}")

      # TODO: rewrite check to ensure eytzinger order
      # labels = index[:,0].astype(np.int64)
      # labeldiff = labels[1:] - labels[0:-1]
      # if np.any(labeldiff < 1):
      #   raise ValidationError("Labels aren't sorted.")
    elif len(buf) != HEADER_LENGTH:
      raise ValidationError("Format is longer than header for zero data.")

    return True

# TODO: rewrite as a stack to prevent possible stackoverflows
def eytzinger_sort(inpt, output, i = 0, k = 1):
  """
  Takes an ascendingly sorted input and 
  an equal sized output buffer into which to 
  rewrite the input in eytzinger order.

  Modified from:
  https://algorithmica.org/en/eytzinger
  """
  if k <= len(inpt):
    i = eytzinger_sort(inpt, output, i, 2 * k)
    output[k - 1] = inpt[i]
    i += 1
    i = eytzinger_sort(inpt, output,i, 2 * k + 1)
  return i

def ffs(x):
  """
  Returns the index, counting from 1, of the
  least significant set bit in `x`.

  Modified from: 
  https://stackoverflow.com/questions/5520655/return-index-of-least-significant-bit-in-python
  """
  return int(x & -x).bit_length()

