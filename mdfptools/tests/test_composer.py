import unittest
from mdfptools.Composer import *
import pickle
import mdtraj as md
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class ComposerTest(unittest.TestCase):
    def _load(self):
        self.smiles = "Cl-C1:C:C:C:C2:C:C:C:C:C:1:2"
        with open("{}/data/parmed.pickle".format(THIS_DIR), "rb") as f:
            self.parm = pickle.load(f)
        self.traj = md.load("{}/data/mdtraj.h5".format(THIS_DIR))
    def test_mdfp_composer(self):
        self._load()
        out = MDFPComposer(self.smiles, self.traj, self.parm)
        print(out.get_mdfp())
    def test_liquid_composer(self):
        self._load()
        out = LiquidComposer(self.smiles, self.traj, self.parm)
        print(out.get_mdfp())
    def test_solution_liquid_composer(self):
        self._load()
        out = SolutionLiquidComposer(self.smiles, self.traj, self.parm, self.traj, self.parm)
        print(out.get_mdfp())
