import math
import mmap 
import io

from .exceptions import ValidationError
from .lib import nvl, eytzinger_sort
from . import compression

import numpy as np

import mapbufferaccel

DATA_TYPE = {
  np.unsignedinteger: 0,
  np.integer: 1,
  np.floating: 2,
  np.complex64: 3,
  0: np.unsignedinteger,
  1: np.integer,
  2: np.floating,
  3: np.complex64,
}

class MapInt:
  """Represents a usable int->int dictionary as a byte string."""
  FORMAT_VERSION = 0
  MAGIC_NUMBERS = b"mapint"
  HEADER_LENGTH = 12
  __slots__ = (
    "buffer", "index", "_key_type", "_value_type"
  )
  def __init__(
    self, data=None
    # key_type="uint64", value_type="uint64"
  ):
    """
    data: dict (number->number)
    """
    # key_type: numpy dtype
    # value_type: numpy dtype

    # key type and value type must be of the same byte width and class 
    # e.g. integer, float, or complex. Signed and unsigned can be
    # mixed. In the future, other type combinations may be supported.

    # Need to rewrite C code to accept multiple
    # types.
    key_type = np.uint64
    value_type = np.uint64

    self._key_type = None
    self._value_type = None

    if isinstance(data, dict):
      self.buffer = self.dict2buf(data, np.dtype(key_type), np.dtype(value_type))
    elif isinstance(data, io.IOBase):
      self.buffer = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    elif isinstance(data, (bytes, mmap.mmap)):
      self.buffer = data
    else:
      raise TypeError("data must be a dict, bytes, file, or mmap. Got: " + str(type(data)))

    self.index = self.decode_index()

  def decode_index(self):
    N = len(self)
    header_len = MapInt.HEADER_LENGTH
    index_length = 2 * N * np.dtype(self.key_type).itemsize
    index = self.buffer[header_len:index_length+header_len]
    return np.frombuffer(index, dtype=self.key_type).reshape((N,2))

  def __len__(self):
    """Returns number of keys."""
    return int.from_bytes(self.buffer[8:12], byteorder="little", signed=False)

  @property
  def format_version(self):
    return self.buffer[len(MapInt.MAGIC_NUMBERS)]

  @property
  def key_type(self):
    if self._key_type is not None:
      return self._key_type

    self._key_type = self.type_id_to_dtype(
      int(self.buffer[7] >> 5)
    )
    return self._key_type
  
  @property
  def value_type(self):
    if self._value_type is not None:
      return self._value_type

    self._value_type = self.type_id_to_dtype(
      int((self.buffer[7] >> 2) & 0b00000111)
    )
    return self._value_type 

  def type_id_to_dtype(self, type_id):
    dtype_class = DATA_TYPE[type_id]
    width = self.itemsize

    if dtype_class == np.unsignedinteger:
      opts = ( np.uint8, np.uint16, np.uint32, np.uint64 )
    elif dtype_class == np.integer:
      opts = ( np.int8, np.int16, np.int32, np.int64 )
    elif dtype_class == np.floating:
      opts = ( None, np.float16, np.float32, np.float64 )
    elif dtype_class == np.complex64:
      return np.complex64

    return opts[width]    

  @property
  def itemsize(self):
    return int(self.buffer[7] & 0b00000011)

  def __iter__(self):
    yield from self.keys()

  def keys(self):
    for key, value in self.index:
      yield key

  def values(self):
    for label, value in self.index:
      yield value

  def items(self):
    N = len(self)
    index = self.index
    for i in range(N):
      label = index[i,0]
      value = index[i,1].view(self.value_type)
      yield (label, value)

  def getindex(self, i):
    return self.index[i,1].view(self.value_type)

  def find_index_position(self, label):
    return mapbufferaccel.eytzinger_binary_search(label, self.index)

  def get(self, label, *args, **kwargs):
    pos = self.find_index_position(label)
    if pos == -1: # try to get default argument
      try:
        return args[0]
      except IndexError:
        try:
          return kwargs.get("default")
        except KeyError:
          raise KeyError("{} was not found.".format(label))
    
    return self.getindex(pos)

  def __contains__(self, label):
    return self.find_index_position(label) >= 0

  def __getitem__(self, label):
    pos = mapbufferaccel.eytzinger_binary_search(label, self.index)
    if pos >= 0:
      return self.index[pos,1].view(self.value_type)
    else:
      raise KeyError("{} was not found.".format(label))

  def dict2buf(self, data, key_type, value_type):
    """Structure [ index length, sorted index, data ]"""
    labels = np.array([ int(lbl) for lbl in data.keys() ], dtype=key_type)
    labels.sort()

    out = np.zeros((len(labels),), dtype=key_type)
    eytzinger_sort(labels, out)
    labels = out

    N = len(labels)
    N_region = N.to_bytes(4, byteorder="little", signed=False)

    key_width = np.dtype(key_type).itemsize
    val_width = np.dtype(value_type).itemsize

    if key_width != val_width:
      raise TypeError(f"The key and value types must have the same bit width. Key Type: {key_type} Value Type: {value_type}")

    def type_id(dtype):
      if np.issubdtype(dtype, np.unsignedinteger):
        return DATA_TYPE[np.unsignedinteger]
      elif np.issubdtype(dtype, np.integer):
        return DATA_TYPE[np.integer]
      elif np.issubdtype(dtype, np.floating):
        return DATA_TYPE[np.floating]
      elif np.issubdtype(dtype, np.complex64):
        return DATA_TYPE[np.complex64]
      else:
        raise TypeError(f"Type not supported: {dtype}")

    dtype_byte = (type_id(key_type) << 5) | (type_id(value_type) << 2) | int(math.log2(key_width))

    header = (
      MapInt.MAGIC_NUMBERS 
      + bytes([ MapInt.FORMAT_VERSION ]) 
      + bytes([ dtype_byte ])
      + N_region
    )

    if N == 0:
      return header

    index_length = 2 * N
    index = np.zeros((index_length,), dtype=key_type)
    index[::2] = labels

    for i in range(N):
      index[2*i+1] = data[index[2*i]]

    return b"".join([ header, index.tobytes() ])

  def todict(self):
    return { label: val for label, val in self.items() }

  def tobytes(self):
    return self.buffer

  def validate(self):
    return self.validate_buffer(self.buffer)

  @staticmethod
  def validate_buffer(buf):
    mapbuf = MapInt(buf)
    index = mapbuf.index()
    if len(index) != len(mapbuf):
      raise ValidationError(f"Index size doesn't match. len(mapbuf): {len(mapbuf)}")

    magic = buf[:len(MapInt.MAGIC_NUMBERS)]
    if magic != MapInt.MAGIC_NUMBERS:
      raise ValidationError(f"Magic number mismatch. Expected: {MapInt.MAGIC_NUMBERS} Got: {magic}")

    if mapbuf.format_version not in (0,):
      raise ValidationError(f"Unsupported format version. Got: {mapbuf.format_version}")

    if len(mapbuf) == 0 and len(buf) != MapInt.HEADER_LENGTH:
      raise ValidationError("Format is longer than header for zero data.")

    return True

  def __repr__(self):
    return str(self.todict())
