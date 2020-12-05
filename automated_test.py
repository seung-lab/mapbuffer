import pytest
import numpy as np
from mapbuffer import MapBuffer


def test_empty():
  mbuf = MapBuffer({})
  assert len(mbuf) == 0
  assert list(mbuf) == []

  assert mbuf.validate()
