# Wave Documentation

This directory uses Sphinx to build documentation that is hosted at
https://wave-lang.readthedocs.io/.

## Publishing on Read the Docs

The project dashboard is here:
https://app.readthedocs.org/projects/wave-lang/.

A webhook is configured in the wave GitHub repository to notify
readthedocs when new changes are pushed, at which point it automatically starts
a website build and publishes the latest content.

## Building the documentation locally

### Setup virtual environment with requirements

From the project root directory:

```shell
python -m venv .venv
source .venv/bin/activate

# Install PyTorch (CPU version for docs)
python -m pip install -r pytorch-cpu-requirements.txt

# Install wave with docs dependencies
python -m pip install -e ".[docs]"
```

### Build docs

```shell
sphinx-build -b html . _build
```

### Serve locally locally with autoreload

```shell
# Default config - only watch for changes to this folder (.rst files).
sphinx-autobuild . _build

# Advanced config - also watch for changes to the entire project (.py files).
sphinx-autobuild . _build --watch ..
```

Then open http://127.0.0.1:8000 as instructed by the logs and make changes to
the files in this directory as needed to update the documentation.

## Authoring documentation

### Useful references

* https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
* https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
