import os
import numpy as np
from torch.utils import data
from scipy.io import loadmat
from ..utils.units import Hz


class ECoGJoystickTracking(data.Dataset):
    """Helper class to read the ECoG Joystick Tracking dataset from
    Schalk, G., J. Kubanek, K. J. Miller, N. R. Anderson, E. C. Leuthardt, J. G. Ojemann, D. Limbrick, D. Moran, L. A. Gerhardt, and J. R. Wolpaw. "Decoding two-dimensional movement trajectories using electrocorticographic signals in humans." Journal of neural engineering 4, no. 3 (2007): 264

    Ethics statement: All patients participated in a purely voluntary manner, after providing informed written consent, under experimental protocols approved by the Institutional Review Board of the University of Washington (#12193). All patient data was anonymized according to IRB protocol, in accordance with HIPAA mandate. It was made available through the library described in “A Library of Human Electrocorticographic Data and Analyses” by Kai Miller [Reference], freely available at https://searchworks.stanford.edu/view/zk881ps0522
    """

    def __init__(self, path, transforms=None, users=["fp", "gf", "rh", "rr"]):
        assert os.path.exists(path), f"Data not found at '{path}'"
        self.path = path
        self.transforms = transforms
        self.fs = 1000 * Hz  # Sampling frequency
        self.users = users

    @property
    def nsfilt(self):
        # amplitude roll-off function used for filtering
        return loadmat(os.path.join(self.path, "ns_1k_1_300_filt.mat"))["nsfilt"]

    def _loadmat(self, index):
        return loadmat(os.path.join(self.path, "data", f"{self.users[index]}_joystick.mat"))

    def electrode_positions(self, index):
        # Talairach coordinate systems for the 60 electrodes
        return self._loadmat(index)["electrodes"]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        """Return electrode data (n_electrode x time) and labels (4 x time), with labels = (x, y, target_x, target_y)"""
        mat = self._loadmat(index)
        data = mat["data"] if self.transforms is None else self.transforms(mat["data"])
        return data.T, np.stack((mat["CursorPosX"], mat["CursorPosY"], mat["TargetPosX"], mat["TargetPosY"])).squeeze()
