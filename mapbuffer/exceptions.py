class DecompressionError(BaseException):
  """
  Decompression failed.
  """
  pass

class CompressionError(BaseException):
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

class ValidationError(BaseException):
  pass