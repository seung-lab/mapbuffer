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

from .mapbuffer import MapBuffer, HEADER_LENGTH, MAGIC_NUMBERS, FORMAT_VERSION
from .exceptions import *