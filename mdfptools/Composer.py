from rdkit import Chem
from rdkit.Chem import AllChem
# from Per_Frame_Property_Extractor import *
from .Extractor import *


class BaseComposer():
    from numpy import mean, std, median
    import functools

    def __init__(self, smiles ):
        self.smiles = smiles
        self.fp = {}
        self._get_relevant_properties()

    def get_mdfp(self):
        #TODO # self.vector_indices = []
        return functools.reduce(lambda a, b : a + b, self.fp.values())

    def _get_relevant_properties(self):
        self.fp  = {**self.fp, **self._get_2d_descriptors()}

    def _get_2d_descriptors(self):
        m = Chem.MolFromSmiles(self.smiles, sanitize = True)
        if m is None:
            m = Chem.MolFromSmiles(self.smiles, sanitize = False)
            m.UpdatePropertyCache(strict=False)
        fp = []
        fp.append(m.GetNumHeavyAtoms())
        fp.append(AllChem.CalcNumRotatableBonds(m))
        fp.append(len(m.GetSubstructMatches(Chem.MolFromSmarts('[#7]')))) # nitrogens
        fp.append(len(m.GetSubstructMatches(Chem.MolFromSmarts('[#8]')))) # oxygens
        fp.append(len(m.GetSubstructMatches(Chem.MolFromSmarts('[#9]')))) # fluorines
        fp.append(len(m.GetSubstructMatches(Chem.MolFromSmarts('[#15]')))) # phosphorous
        fp.append(len(m.GetSubstructMatches(Chem.MolFromSmarts('[#16]')))) # sulfurs
        fp.append(len(m.GetSubstructMatches(Chem.MolFromSmarts('[#17]')))) # chlorines
        fp.append(len(m.GetSubstructMatches(Chem.MolFromSmarts('[#35]')))) # bromines
        fp.append(len(m.GetSubstructMatches(Chem.MolFromSmarts('[#53]')))) # iodines
        return {"2d_counts" : fp}


    def _get_statistical_moments(self, property_extractor, statistical_moments = [mean, std, median], **kwargs):
        self.statistical_moments = [i.__name__ for i in statistical_moments]
        fp = {}
        prop = property_extractor(**kwargs)
        for i in prop:
            fp[i] = []
            for func in statistical_moments:
                 fp[i].append(func(prop[i]))
        return fp

class MDFPComposer(BaseComposer):
    def __init__(self, smiles, mdtraj_obj, parmed_obj):
        self.kwargs = {"mdtraj_obj" : mdtraj_obj ,
                        "parmed_obj" : parmed_obj}
        super(MDFPComposer, self).__init__(smiles)

    def _get_relevant_properties(self):
        self.fp  = {**self.fp, **self._get_2d_descriptors()}
        self.fp  = {**self.fp, **self._get_statistical_moments(WaterExtractor.extract_energies, **self.kwargs)}
        self.fp  = {**self.fp, **self._get_statistical_moments(WaterExtractor.extract_rgyr, **self.kwargs)}
        self.fp  = {**self.fp, **self._get_statistical_moments(WaterExtractor.extract_sasa, **self.kwargs)}

        del self.kwargs

class LiquidComposer(BaseComposer):
    def __init__(self, smiles, mdtraj_obj, parmed_obj):
        self.kwargs = {"mdtraj_obj" : mdtraj_obj ,
                        "parmed_obj" : parmed_obj}
        super(LiquidComposer, self).__init__(smiles)

    def _get_relevant_properties(self):
        self.fp  = {**self.fp, **self._get_2d_descriptors()}
        self.fp  = {**self.fp, **self._get_statistical_moments(LiquidExtractor.extract_energies, **self.kwargs)}
        self.fp  = {**self.fp, **self._get_statistical_moments(LiquidExtractor.extract_rgyr, **self.kwargs)}
        self.fp  = {**self.fp, **self._get_statistical_moments(LiquidExtractor.extract_sasa, **self.kwargs)}
        self.fp  = {**self.fp, **self._get_statistical_moments(LiquidExtractor.extract_dipole_magnitude, **self.kwargs)}

        del self.kwargs

class SolutionLiquidComposer(BaseComposer):
    def __init__(self, smiles, solv_mdtraj_obj, solv_parmed_obj, liq_mdtraj_obj, liq_parmed_obj):
        self.kwargs_solv = {"mdtraj_obj" : solv_mdtraj_obj ,
                        "parmed_obj" : solv_parmed_obj}
        self.kwargs_liq = {"mdtraj_obj" : liq_mdtraj_obj ,
                        "parmed_obj" : liq_parmed_obj}
        super(SolutionLiquidComposer, self).__init__(smiles)

    def _get_relevant_properties(self):
        self.fp  = {**self.fp, **self._get_2d_descriptors()}

        self.fp  = {**self.fp, **self._get_statistical_moments(WaterExtractor.extract_energies, **self.kwargs_solv)}
        self.fp  = {**self.fp, **self._get_statistical_moments(WaterExtractor.extract_rgyr, **self.kwargs_solv)}
        self.fp  = {**self.fp, **self._get_statistical_moments(WaterExtractor.extract_sasa, **self.kwargs_solv)}

        self.fp  = {**self.fp, **self._get_statistical_moments(LiquidExtractor.extract_energies, **self.kwargs_liq)}
        self.fp  = {**self.fp, **self._get_statistical_moments(LiquidExtractor.extract_rgyr, **self.kwargs_liq)}
        self.fp  = {**self.fp, **self._get_statistical_moments(LiquidExtractor.extract_sasa, **self.kwargs_liq)}
        self.fp  = {**self.fp, **self._get_statistical_moments(LiquidExtractor.extract_dipole_magnitude, **self.kwargs_liq)}

        del self.kwargs_liq, self.kwargs_solv

"""
parm_path = '/home/shuwang/Documents/Modelling/MDFP/Codes/vapour_pressure/crc_handbook/corrupted/RU18.1_8645.pickle'
parm = pickle.load(open(parm_path,"rb"))
traj = md.load('/home/shuwang/Documents/Modelling/MDFP/Codes/vapour_pressure/crc_handbook/corrupted/RU18.1_8645.h5')[:10]
# print(Liquid_Extractor.extract_dipole_magnitude(traj, parm))
x = MDFPComposer("Cl-C1:C:C:C:C2:C:C:C:C:C:1:2", traj, parm)
# print(x._get_statistical_moments(Base_Extractor.extract_rgyr, **{"mdtraj_obj" : traj}))
# print(x._get_statistical_moments(Liquid_Extractor.extract_dipole_magnitude, **{"mdtraj_obj" : traj, "parmed_obj" : parm}))
# print(x._get_statistical_moments(Base_Extractor.extract_sasa, **{"mdtraj_obj" : traj, "parmed_obj" : parm}))
# print(x._get_statistical_moments(Liquid_Extractor.extract_energies, **{"mdtraj_obj" : traj, "parmed_obj" : parm , "platform" : "OpenCL"}))
print(x.fp)
print(x.__dict__)
print(x.get_mdfp())
pickle.dump(x, open("/home/shuwang/tmp.pickle", "wb"))
"""
