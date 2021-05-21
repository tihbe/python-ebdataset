ebdataset
=========

An event based dataset loader under one common python (>=3.5) API built on top of numpy record arrays for sparse representation and PyTorch for dense representation.

# Supported datasets

1. Neuromorphic Mnist dataset from
    Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.
    “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades",
    Frontiers in Neuroscience, vol.9, no.437, Oct. 2015. Available for download at https://www.garrickorchard.com/datasets/n-mnist

2. NCaltech101 dataset from
    Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.
    “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades",
    Frontiers in Neuroscience, vol.9, no.437, Oct. 2015. Available for download at https://www.garrickorchard.com/datasets/n-caltech101

3. IBM DVS128 Gesture dataset from
    A. Amir, B. Taba, D. Berg, T. Melano, J. McKinstry, C. Di Nolfo, T. Nayak, A. Andreopoulos, G. Garreau, M. Mendoza, J. Kusnitz, M. Debole, S. Esser, T. Delbruck, M. Flickner, and D. Modha,
    "A Low Power, Fully Event-Based Gesture Recognition System,"
    2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017. Available for download at http://research.ibm.com/dvsgesture/

4. INI Roshambo17 dataset from
    I.-A. Lungu, F. Corradi, and T. Delbruck,
    Live Demonstration: Convolutional Neural Network Driven by Dynamic Vision Sensor Playing RoShamBo,
    in 2017 IEEE Symposium on Circuits and Systems (ISCAS 2017), Baltimore, MD, USA, 2017. Available for download at https://drive.google.com/file/d/0BzvXOhBHjRheYjNWZGYtNFpVRkU/view?usp=sharing

5. INI UCF-50 dataset from:
    Hu, Y., Liu, H., Pfeiffer, M., and Delbruck, T. (2016).
    DVS Benchmark Datasets for Object Tracking, Action Recognition and Object Recognition.
    Front. Neurosci. 10, 405. doi:10.3389/fnins.2016.00405. Available for download at https://dgyblog.com/projects-term/dvs-dataset.html

6. NTidigits dataset from:
    Anumula, Jithendar, et al. “Feature Representations for Neuromorphic Audio Spike Streams.”
    Frontiers in Neuroscience, vol. 12, Feb. 2018, p. 23. DOI.org (Crossref), doi:10.3389/fnins.2018.00023. Available for download at https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M

7. Prophesee N-Cars dataset from:
    Amos Sironi, Manuele Brambilla, Nicolas Bourdis, Xavier Lagorce, Ryad Benosman
    “HATS: Histograms of Averaged Time Surfaces for Robust Event-based Object Classification”.
    To appear in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. Available for download at https://www.prophesee.ai/2018/03/13/dataset-n-cars/

# Installation
You can install the latest version of this package with:
```bash
pip install ebdataset
```

# Getting started

In the code:
```python
from ebdataset.vision import NMnist
from ebdataset.vision.transforms import ToDense
from quantities import ms

# With sparse representation:
for spike_train, label in NMnist(path):
    spike_train.x, spike_train.y, spike_train.p, spike_train.ts
    break

# Or use the pytorch transforms API for dense tensors
dt = 1*ms
loader = NMnist(path, is_train=True, transforms=ToDense(dt))
for spike_train, label in loader:
    spike_train.shape # => (34, 34, 2, duration of sample)
    break
```

Or with the visualization sub-package:
```bash
python -m ebdataset.visualization.spike_train_to_vid NMnist path
```

![](images/nmnist-2.gif) ![](images/nmnist-9.gif)

# Contributing

Feel free to create a pull request if you're interested in this project. 
