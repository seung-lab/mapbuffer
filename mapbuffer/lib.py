import os.path
import time
import types
import sys

COLORS = {
  'RESET': "\033[m",
  'YELLOW': "\033[1;93m",
  'RED': '\033[1;91m',
  'GREEN': '\033[1;92m',
}

def green(text):
  return colorize('green', text)

def yellow(text):
  return colorize('yellow', text)

def red(text):
  return colorize('red', text)

def colorize(color, text):
  color = color.upper()
  return COLORS[color] + text + COLORS['RESET']

def toabs(path):
  path = os.path.expanduser(path)
  return os.path.abspath(path)

def mkdir(path):
  path = toabs(path)

  try:
    if path != '' and not os.path.exists(path):
      os.makedirs(path)
  except OSError as e:
    if e.errno == 17: # File Exists
      time.sleep(0.1)
      return mkdir(path)
    else:
      raise

  return path

def touch(path):
  mkdir(os.path.dirname(path))
  open(path, 'a').close()

def nvl(*args):
  """Return the leftmost argument that is not None."""
  if len(args) < 2:
    raise IndexError("nvl takes at least two arguments.")
  for arg in args:
    if arg is not None:
      return arg
  return args[-1]

def sip(iterable, block_size):
  """Sips a fixed size from the iterable."""
  ct = 0
  block = []
  for x in iterable:
    ct += 1
    block.append(x)
    if ct == block_size:
      yield block
      ct = 0
      block = []

  if len(block) > 0:
    yield block

def first(lst):
  if isinstance(lst, types.GeneratorType):
    return next(lst)
  try:
    return lst[0]
  except TypeError:
    return next(iter(lst))

def toiter(obj, is_iter=False):
  if isinstance(obj, str) or isinstance(obj, dict):
    if is_iter:
      return [ obj ], False
    return [ obj ]

  try:
    iter(obj)
    if is_iter:
      return obj, True
    return obj 
  except TypeError:
    if is_iter:
      return [ obj ], False
    return [ obj ]

def duplicates(lst):
  dupes = []
  seen = set()
  for elem in lst:
    if elem in seen:
      dupes.append(elem)
    seen.add(elem)
  return set(dupes)

