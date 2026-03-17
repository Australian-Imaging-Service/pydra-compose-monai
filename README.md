# Pydra-compose-monai plugin

Pydra-compose-monai is a plugin package for the [Pydra dataflow engine](https://nipype.github.io/pydra),
which adds the feature to wrap up [BIDS Apps](https://bids-apps.neuroimaging.io/) into
Pydra task classes that take input files, stores them in a BIDS dataset created on the
fly, runs the BIDS App on them, then extracts the files into the tasks outputs. It can
be useful when running BIDS app on non-BIDS structured data (e.g. XNAT or LORIS repositories).

## For developers

Install repo in developer mode from the source directory. It is also useful to
install pre-commit to take care of styling via [black](https://black.readthedocs.io/):

```bash
python3 -m pip install -e '.[dev]'
pre-commit install
```
