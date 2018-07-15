import unittest
from mdfptools.Extractor import *
import pickle
import mdtraj as md
import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class ExtractorTest(unittest.TestCase):
    def _load(self):
        with open("{}/data/parmed.pickle".format(THIS_DIR), "rb") as f:
            self.parm = pickle.load(f)
        self.traj = md.load("{}/data/mdtraj.h5".format(THIS_DIR))
    def test_extract_dipole_moment(self):
        self._load()
        out = LiquidExtractor.extract_dipole_magnitude(self.traj, self.parm)
        print(out)
    def test_extract_energies(self):
        self._load()
        out = SolutionExtractor.extract_energies(self.traj, self.parm)
        print(out)
    def test_extract_rgyr(self):
        self._load()
        out = SolutionExtractor.extract_rgyr(self.traj)
        print(out)
    def test_extract_sasa(self):
        self._load()
        out = SolutionExtractor.extract_sasa(self.traj)
        print(out)
