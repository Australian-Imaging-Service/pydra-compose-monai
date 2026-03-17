# Pydra-compose-monai plugin

Pydra-compose-monai is a plugin package for the [Pydra dataflow engine](https://nipype.github.io/pydra),
which adds the feature to wrap up [MONAI Model Zoo]() into
Pydra task classes

## For developers

Install repo in developer mode from the source directory. It is also useful to
install pre-commit to take care of styling via [black](https://black.readthedocs.io/):

```bash
python3 -m pip install -e '.[dev]'
pre-commit install
```
