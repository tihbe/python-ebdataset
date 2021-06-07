from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="ebdataset",
    version="0.0.2",
    author="Ismael Balafrej - NECOTIS",
    author_email="ismael.balafrej@usherbrooke.ca",
    description="An event based dataset loader under one common API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tihbe/python-ebdataset",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.14.3",
        "quantities>=0.12.4",
        "tqdm>=4.45.0",
        "torch>=1.4.0",
        "torchvision>=0.5.0",
        "h5py>=2.10.0",
    ],
    python_requires=">=3.5.2",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
