# mapbuffer

Map of integers to binary buffers requiring near zero parsing.

```python
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
```

MapBuffer is designed to allow you to store dictionaries mapping integers to binary buffers in a serialized format and then read that back in and use it without requiring an expensive parse of the entire dictionary. Instead, if you have a dictionary containing thousands of keys, but only need a few items from it you can extract them rapidly.  

This serialization format was designed to solve a performance problem with our pipeline for merging skeleton fragments from a large dense image segmentation. The 3D image was carved up into a grid and each gridpoint generated potentially thousands of skeletons which were written into a single pickle file. Since an individual segmentation could cross many gridpoints, fusion across many files is required, but each file contains many irrelevant skeleton fragments for a given operation. In one measurement, `pickle.loads` was taking 68% of the processing time for an operation that was taking two weeks to run on hundreds of cores. 

Therefore, this method was developed to skip parsing the dictionaries and rapidly extract skeleton fragments.

## Design

