import mmap 
import io

from .exceptions import ValidationError
from .lib import nvl, eytzinger_sort
from . import compression

import crc32c
import numpy as np

import mapbufferaccel

FORMAT_VERSION = 1
MAGIC_NUMBERS = b"mapbufr"
HEADER_LENGTH = 16

class MapBuffer:
  """Represents a usable int->bytes dictionary as a byte string."""
  __slots__ = (
    "data", "tobytesfn", "frombytesfn", 
    "dtype", "buffer", "check_crc", 
    "_header", "_index", "_compress"
  )
  def __init__(
    self, data=None, compress=None,
    tobytesfn=None, frombytesfn=None,
    check_crc=True
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
    self.check_crc = check_crc

    self._header = None
    self._index = None
    self._compress = None

    if isinstance(data, dict):
      self.buffer = self.dict2buf(data, compress)
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
    return int.from_bytes(self.header[12:16], byteorder="little", signed=False)

  @property
  def compress(self):
    if self._compress:
      return self._compress

    self._compress = compression.normalize_encoding(
      self.header[8:12]
    )
    return self._compress

  @property
  def format_version(self):
    return self.header[len(MAGIC_NUMBERS)]

  def __iter__(self):
    yield from self.keys()

  def datasize(self):
    """Returns size of data region in bytes."""
    return len(self.buffer) - HEADER_LENGTH - len(self) * 2 * 8

  @property
  def header(self):
    """Get the header bytes."""
    if self._header is not None:
      return self._header

    # seems dumb, buf if self.buffer is an object that
    # requires network access, this is a valuable cache
    self._header = self.buffer[:HEADER_LENGTH]
    return self._header

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

    if self.format_version == 1:
      stored_check_value = int.from_bytes(value[-4:], byteorder='little')
      value = value[:-4]
      if self.check_crc:
        retrieved_check_value = crc32c.crc32c(value)
        if retrieved_check_value != stored_check_value:
          raise ValidationError(
            f"Label {i} failed its crc32c check. Stored: {stored_check_value} Computed: {retrieved_check_value}"
          )

    encoding = self.compress
    if encoding:
      value = compression.decompress(value, encoding, str(index[i,0]))

    if self.frombytesfn:
      value = self.frombytesfn(value)

    return value  

  def setindex(self, i, data):
    index = self.index()
    N = index.shape[0]
    offset = index[i,1]
    if i < N - 1:
      next_offset = index[i+1,1]
      existing_length = int(next_offset - offset) 
    else:
      existing_length = int(len(self.buffer) - offset)
      next_offset = int(len(self.buffer))

    if self.tobytesfn:
      data = self.tobytesfn(data)

    if self.compress:
      data = compression.compress(data, method=self.compress) 

    check_length = len(data)
    if self.format_version == 1:
      check_length += 4

    if check_length != existing_length:
      raise ValueError(
        f"Can only overwrite data of exactly the same length. "
        f"Expected: {existing_length} bytes, Got: {check_length} bytes"
      )

    if self.format_version == 1:
      self.buffer[offset:next_offset] = data
    else:
      data += crc32c.crc32c(data).to_bytes(4, byteorder='little')
      self.buffer[offset:next_offset] = data

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

  def __setitem__(self, label, label_data):
    pos = self.find_index_position(label)
    if pos is not None:
      return self.setindex(pos, label_data)
    else:
      raise KeyError("{} was not found.".format(label))    

  def dict2buf(self, data, compress=None, tobytesfn=None):
    """Structure [ index length, sorted index, data ]"""
    labels = np.fromiter(
      ( int(lbl) for lbl in data.keys() ), 
      count=len(data), dtype=self.dtype
    )
    labels.sort()

    layout = mapbufferaccel.eytzinger_sort_indices(len(data))
    labels = labels[layout]

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
    for label in bytes_data:
      bytes_data[label] += crc32c.crc32c(bytes_data[label]).to_bytes(4, byteorder='little')

    data_region = b"".join(
      ( bytes_data[label] for label in labels )
    )
    index[1] = HEADER_LENGTH + index_length * 8
    for i, label in zip(range(1, len(labels)), labels):
      index[i*2 + 1] = index[(i-1)*2 + 1] + len(bytes_data[labels[i-1]])

    del labels
    
    return b"".join([ header, index.tobytes(), data_region ])

  def todict(self):
    return { label: val for label, val in self.items() }

  def tobytes(self):
    return self.buffer

  def validate(self):
    header = self.header
    index = self.index()
    if len(index) != len(self):
      raise ValidationError(f"Index size doesn't match. len(self): {len(self)}")

    magic = header[:len(MAGIC_NUMBERS)]
    if magic != MAGIC_NUMBERS:
      raise ValidationError(f"Magic number mismatch. Expected: {MAGIC_NUMBERS} Got: {magic}")

    if self.format_version not in (0,1):
      raise ValidationError(f"Unsupported format version. Got: {self.format_version}")

    if self.compress not in compression.COMPRESSION_TYPES:
      raise ValidationError(f"Unsupported compression format. Got: {self.compress}")

    if len(self) > 0:
      offsets = index[:,1].astype(np.int64)
      lengths = offsets[1:] - offsets[0:-1]
      if np.any(lengths < 0):
        raise ValidationError("Offsets are not sorted.")

      length = lengths.sum() + (len(self.buffer) - offsets[-1])
      if length != self.datasize():
        raise ValidationError(f"Data length doesn't match offsets. Predicted: {length} Data Size: {mapbuf.datasize()}")

      # TODO: rewrite check to ensure eytzinger order
      # labels = index[:,0].astype(np.int64)
      # labeldiff = labels[1:] - labels[0:-1]
      # if np.any(labeldiff < 1):
      #   raise ValidationError("Labels aren't sorted.")
    elif len(self.buffer) != HEADER_LENGTH:
      raise ValidationError("Format is longer than header for zero data.")

    return True

  @staticmethod
  def validate_buffer(buf):
    mapbuf = MapBuffer(buf)
    return mapbuf.validate()
