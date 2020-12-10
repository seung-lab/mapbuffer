import mmap 
import io

from .exceptions import ValidationError
from .lib import nvl
from . import compression

import numpy as np

import mapbufferaccel

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
    elif isinstance(data, io.IOBase):
      self.buffer = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    elif isinstance(data, (bytes, mmap.mmap)):
      self.buffer = data
    else:
      raise TypeError("data must be a dict, bytes, file, or mmap. Got: " + str(type(data)))

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

    return b"".join([ header, index.tobytes(), data_region ])

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

