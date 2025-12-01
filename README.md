[![](https://github.com/jkrasting/notebook-template/actions/workflows/python-app.yml/badge.svg)](https://github.com/jkrasting/notebook-template/actions/workflows/python-app.yml)
[![PyPi](https://img.shields.io/pypi/v/esnb.svg)](https://pypi.python.org/pypi/esnb/)
[![Documentation Status](https://readthedocs.org/projects/esnb/badge/?version=latest)](https://esnb.readthedocs.io/en/latest/?badge=latest)

# ESNB: Earth System Notebook
#### <i>(formerly `notebook-template`)</i>
Repository for analysis notebook template development.

Link to work-in-progress [documentation](https://docs.google.com/document/d/1cY-yWoEOANqsDICZWNFNkxbwUHjEXBL63mL6aBbVyyM/edit?usp=sharing) on Google Docs. (You may need to request access.)


## Standards
The final converged notebook should make use of established standards and conventions as much as possible.

[Link to EMDS Standards](https://github.com/Earth-System-Diagnostics-Standards/EMDS/blob/main/standards.md)


## Quick Start

1. Set up a working directory
```
mkdir ~/esnb-test-dir && cd ~/esnb-test-dir
```

2. Create a test conda environment
```
conda create -y -n esnb_test jupyterlab pip
conda activate esnb_test
```

4. Install ESNB
```
pip install esnb
```

6. Create a new Pangeo demo notebook
```
nbinit pangeo
```

8. Launch Jupyter
```
jupyter lab
```
