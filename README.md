# align
Alignment python package. Not general but should allow some generalization with not too much effort.

# Installation

```python3 setup.py install --user```

```python3 -m pip install .``` In a virtual environment (```python3 -m venv "name"``` and ```source name/bin/activate```)

# Minimal example:

Load a list of files:

```
from glob import glob
from alignment import walign, talign
path = "./" # path to data
pattern = "" # Initial part (common) of the file names containing the data to align
files = sorted(glob("%s/%s*.fits" % (path,pattern)))
if (len(files)>0): files1 = walign(files)
if (len(files)>0): talign(files1)
