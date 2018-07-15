import unittest
from mdfptools.Parameteriser import *

class ParameteriserTest(unittest.TestCase):
    def test_solution_parameteriser_openeyy(self):
        parm = SolutionParameteriser().via_openeye("CC")
    def test_solution_parameteriser_rdkit_ddec(self):
        SolutionParameteriser.load_ddec_models()
        parm = SolutionParameteriser().via_rdkit("CC")
        SolutionParameteriser.unload_ddec_models()
    def test_liquid_parameteriser_openeye(self):
        from simtk import unit
        parm = LiquidParameteriser().run("CC", density = 12 * unit.gram / unit.liter)
