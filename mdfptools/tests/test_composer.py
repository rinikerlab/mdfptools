import pickle
import pytest
import numpy as np 
import mdtraj as md
from mdfptools.Composer import *



#TODO test the MDFP class


# @pytest.fixture
# def load_solution_files():
#     [
#         {},
#         {},
#         {},
#     ]
#     pass
import os
print(os.getcwd())


def read_pickle(path): return pickle.load(open(path, "rb"))

#################
@pytest.mark.parametrize(["traj", "pmd", "smiles", "xvg", "ref"], [
    (md.load("./data/water_example_1.h5"), read_pickle("./data/water_example_1.pickle"), None, None, read_pickle("./data/fp_water_example_1.pickle")),

])
def test_SolutionComposer(traj, pmd, smiles, xvg, ref):
    assert np.allclose(SolutionComposer.run(traj, pmd, smiles, xvg).get_mdfp(), ref.get_mdfp(), rtol = 0.005)


#################
@pytest.mark.parametrize(["traj", "pmd", "smiles", "xvg", "ref"], [
    (md.load("./data/liquid_example_1.h5"), read_pickle("./data/liquid_example_1.pickle"), None, None, read_pickle("./data/fp_liquid_example_1.pickle")),

])
def test_LiquidComposer(traj, pmd, smiles, xvg, ref):
    assert np.allclose(LiquidComposer.run(traj, pmd, smiles, xvg).get_mdfp(), ref.get_mdfp(), rtol = 0.005)

#################
@pytest.mark.parametrize(["traj", "pmd", "smiles", "xvg", "ref"], [
    (md.load("./data/water_gmx_example_1.xtc", top = "./data/water_gmx_example_1.gro"), None, "CCCCC1C2(CCN(CC2)C3(CCN(CC3)C(=O)C4=C(C=CC=C4C)C)C)OC(=O)N1CC5CCCCC5", "./data/water_gmx_example_1.xvg", read_pickle("./data/fp_water_gmx_example_1.pickle"))
])
def test_Solution42BitsComposer(traj, pmd, smiles, xvg, ref):
    assert np.allclose(Solution42BitsComposer.run(traj, pmd, smiles, xvg).get_mdfp(), ref.get_mdfp(), rtol = 0.005)
