import numpy as np

class MapBuffer:
  def __init__(self, data=None, dtype=np.int64, factory=None):
    self.factory = factory
    self.dtype = dtype
    self.buffer = None

    if isinstance(data, dict):
      self.buffer = self.dict2buf(data)
    elif isinstance(data, bytes):
      self.buffer = data
    else:
      raise TypeError("data must be a dict or bytes. Got: " + str(type(dict)))

  def __len__(self):
    return int.from_bytes(self.buffer[:4], byteorder="little", signed=False)

  def __iter__(self):
    yield from self.keys()

  def datasize(self):
    return len(self.buffer) - 4 - len(self) * 2 * 8

  def index(self):
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

    if self.factory:
      value = self.factory(value)

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

  def dict2buf(self, data):
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
    
    data_region = b"".join(( data[label] for label in labels ))
    index[1] = 4 + index_length * 8
    for i, label in zip(range(1, len(labels)), labels):
      index[i*2 + 1] = index[(i-1)*2 + 1] + len(data[labels[i-1]])

    return N_region + index.tobytes() + data_region

  def todict(self):
    return { label: val for label, val in self.items() }

  def tobytes(self):
    return self.buffer

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
    

# x = MapBuffer({ 1: b'123', 5: b'456', 4: b'789' }, factory=lambda x: int(x))
# print(x.buffer)
# print(list(x.keys()))
# print(x[5])

