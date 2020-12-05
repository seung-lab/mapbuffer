class DecompressionError(Exception):
  """
  Decompression failed.
  """
  pass

class CompressionError(Exception):
  """
  Compression failed.
  """
  pass

class UnsupportedCompressionType(ValueError):
  """
  Raised when attempting to use a compression type which is unsupported
  by the storage interface.
  """
  pass