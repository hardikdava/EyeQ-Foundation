import setuptools
from setuptools import find_packages
import re

with open("./eyeq/__init__.py", 'r') as f:
    content = f.read()
    # from https://www.py4u.net/discuss/139845
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EyeQ",
    version=version,
    author="hardikdava",
    author_email="hardik1901dava@gmail.com",
    description="Computer Vision Inference Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=[
        # list your requires
        # torch, opencv-python, numpy, supervision, segment-anything, grounding-dino
    ],
    packages=find_packages(exclude=("tests",)),
    extras_require={
        "dev": ["flake8", "black==22.3.0", "isort", "twine", "pytest", "wheel"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL-3.0 license",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
