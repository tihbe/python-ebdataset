from distutils.core import setup

setup(
    name="ebdataset",
    version="0.0.1",
    description="An event based dataset loader under one common API.",
    author="Ismael Balafrej - NECOTIS",
    author_email="ismael dot balafrej at usherbrooke dot ca",
    url="http://www.gel.usherbrooke.ca/necotis/",
    packages=["ebdataset", "ebdataset.vision", "ebdataset.audio", "ebdataset.vision.parsers", "ebdataset.utils"],
    install_requires=[
        "numpy>=1.14.3",
        "opencv-python>=4.2.0",
        "quantities>=0.12.4",
        "tqdm>=4.45.0",
        "torch>=1.4.0",
        "torchvision>=0.5.0",
        "h5py>=2.10.0",
    ],
    python_requires=">=3.5.2",
)
