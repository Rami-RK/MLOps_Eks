#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from pathlib import Path

# Load the package's VERSION file
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / 'requirements'
PACKAGE_DIR = ROOT_DIR / 'titanic_model'

with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()

def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return fd.read().splitlines()

# Call the setup function
setup(
    version=_version,  # Dynamically load the version
    install_requires=list_reqs(),  # Dynamically load the requirements
    include_package_data=True
)
