import os
from setuptools import find_packages, setup

NAME = "gunpowder-distances"
DESCRIPTION = "Nodes for distance transforms in gunpowder"
URL = "https://github.com/saalfeldlab/gunpowder-distances"
EMAIL = "heinrichl@janelia.hhmi.org"
AUTHOR = "Larissa Heinrich"
REQUIRES_PYTHON = ">=3.6"

REQUIRED = [
    "numpy",
    "scipy",
    "gunpowder",
]

EXTRAS = {}

DEPENDENCY_LINKS = []

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), "r") as f:
    LONG_DESCRIPTION = "\n" + f.read()
with open(os.path.join(here, "gpdist", "VERSION"), "r") as version_file:
    VERSION = version_file.read().strip()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    entry_points={},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    dependency_links=DEPENDENCY_LINKS,
    package_data={"gpdist": ["VERSION"]},
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
