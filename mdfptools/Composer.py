from rdkit import Chem
from rdkit.Chem import AllChem
# from Per_Frame_Property_Extractor import *
from .Extractor import *

from numpy import mean, std, median, isnan, array

import functools

class MDFP():
    """
    A MDFP object contains a set of features for a molecule, obtaining from a simulation or a set of simulations.


    .. todo::
        - method to give back the keys
        - store some metdadata?
    """
    def __init__(self, mdfp_dict):
        """
        Parameters
        ----------
        fp_dict : dict
            Keys are each of the type features a given `Extractor` obtains, e.g. "2d_count" are the 2D topological features obtained from molecule SMILES, "intra_lj" are the intra-molecular LJ energies obtained from simulation.

            Values are the corresponding set of numerics, stored as lists.
        """
        self.mdfp = mdfp_dict

    def get_mdfp(self):
        """
        Returns
        ----------
        a list of floating values, i.e. the mdfp feature vector
        """
        return functools.reduce(lambda a, b : a + b, self.mdfp.values())

    def __str__(self):
        return str(self.mdfp)

class BaseComposer():
    """
    The BaseComposer class containing functions that can be used by different composers for different types of simulations

    """

    @classmethod
    def run(cls, smiles ):
        """
        Parameters
        ----------
        smiles : str
            SMILES string of the solute molecule
        """
        cls.smiles = smiles
        cls.fp = {}
        cls._get_relevant_properties()

        return MDFP(cls.fp)

    @classmethod
    def _get_relevant_properties(cls):
        """
        Where the set of features to be included in the final MDFP are defined
        """
        cls.fp  = {**cls.fp, **cls._get_2d_descriptors()}

    @classmethod
    def _get_2d_descriptors(cls):
        """
        Obtain those 2D topological features as described in the original publication.

        """
        m = Chem.MolFromSmiles(cls.smiles, sanitize = True)
        if m is None:
            m = Chem.MolFromSmiles(cls.smiles, sanitize = False)
            m.UpdatePropertyCache(strict=False)
            Chem.GetSSSR(m)

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


    @classmethod
    def _get_statistical_moments(cls, property_extractor, statistical_moments = [mean, std, median], **kwargs):
        """
        Performs statistical weighting of the numerical properties (e.g. LJ and electrostatics energies) obtained from each frame of simulation.

        Parameters
        ----------
        property_extractor : mdfptools.Extractor
            The particular type of Extractor methodclass used to obtain the various properties from simulation.
        statistical_moments : list
            The list of statistical weighting to be performed to each properties from all the frames. Default list of weighting are the mean, standard deviation and median.
        """
        cls.statistical_moments = [i.__name__ for i in statistical_moments]
        fp = {}
        prop = property_extractor(**kwargs)
        for i in prop:
            fp[i] = []
            tmp = prop[i]
            for func in statistical_moments:
                 fp[i].append(func(array(tmp)[~isnan(tmp)]))
        return fp

"""
class TrialSolutionComposer(BaseComposer):
    def __init__(cls, smiles, mdtraj_obj, parmed_obj, **kwargs):
        cls.kwargs = {"mdtraj_obj" : mdtraj_obj ,
                        "parmed_obj" : parmed_obj}
        cls.kwargs = {**cls.kwargs , **kwargs}
        super(TrialSolutionComposer, cls).__init__(smiles)
    def _get_relevant_properties(cls):
        cls.fp  = {**cls.fp, **cls._get_2d_descriptors()}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(TrialSolutionExtractor.extract_energies, **cls.kwargs)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_rgyr, **cls.kwargs)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_sasa, **cls.kwargs)}

        del cls.kwargs
"""

# class MDFPComposer(BaseComposer):
class SolutionComposer(BaseComposer):
    """
    Composer used to extract features from solution simulations, namely one copy of solute in water solvent. This generates fingerprint most akin to that from the original publication.
    """
    @classmethod
    def run(cls, mdtraj_obj, parmed_obj, smiles = None, **kwargs):
        """
        Parameters
        -----------
        mdtraj_obj : mdtraj.trajectory
            The simulated trajectory
        parmed_obj : parmed.structure
            Parmed object of the fully parameterised simulated system.
        smiles : str
            SMILES string of the solute. If mdfptools.Parameteriser was used during parameterisation, then smiles is automatically obtained from the parmed_obj.
        """
        cls.kwargs = {"mdtraj_obj" : mdtraj_obj ,
                        "parmed_obj" : parmed_obj}
        cls.kwargs = {**cls.kwargs , **kwargs}
        if smiles is None:
            if parmed_obj.title != '': #try to obtain it from `parmed_obj`
                smiles = parmed_obj.title
            else:
                raise ValueError("Input ParMed Object does not contain SMILES string, add SMILES as an additional variable")
        return super().run(smiles)

    @classmethod
    def _get_relevant_properties(cls):
        """
        Where the set of features to be included in the final MDFP are defined
        """
        cls.fp  = {**cls.fp, **cls._get_2d_descriptors()}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_energies, **cls.kwargs)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_rgyr, **cls.kwargs)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_sasa, **cls.kwargs)}
        del cls.kwargs

class LiquidComposer(BaseComposer):
    """
    Composer used to extract features from liquid simulations, namely a box containing replicates of the same molecule.
    """
    # def __init__(cls, smiles, mdtraj_obj, parmed_obj):
    @classmethod
    def run(cls, mdtraj_obj, parmed_obj, smiles = None, **kwargs):
        """
        Parameters
        -----------
        mdtraj_obj : mdtraj.trajectory
            The simulated trajectory
        parmed_obj : parmed.structure
            Parmed object of the fully parameterised simulated system.
        smiles : str
            SMILES string of one copy of the solute. If mdfptools.Parameteriser was used during parameterisation, then smiles is automatically obtained from the parmed_obj.
        """
        cls.kwargs = {"mdtraj_obj" : mdtraj_obj ,
                        "parmed_obj" : parmed_obj}
        cls.kwargs = {**cls.kwargs , **kwargs}
        if smiles is None:
            if parmed_obj.title != '': #try to obtain it from `parmed_obj`
                smiles = parmed_obj.title
            else:
                raise ValueError("Input ParMed Object does not contain SMILES string, add SMILES as an additional variable")
        return super().run(smiles)

    @classmethod
    def _get_relevant_properties(cls):
        """
        Where the set of features to be included in the final MDFP are defined
        """
        cls.fp  = {**cls.fp, **cls._get_2d_descriptors()}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(LiquidExtractor.extract_energies, **cls.kwargs)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(LiquidExtractor.extract_rgyr, **cls.kwargs)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(LiquidExtractor.extract_sasa, **cls.kwargs)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(LiquidExtractor.extract_dipole_magnitude, **cls.kwargs)}

        del cls.kwargs

class SolutionLiquidComposer(BaseComposer):
    """
    Composer used to extract features from pairs of solution and liquid simulations.
    """
    @classmethod
    def run(cls, solv_mdtraj_obj, solv_parmed_obj, liq_mdtraj_obj, liq_parmed_obj, smiles = None, **kwargs):
        """
        Parameters
        -----------
        solv_mdtraj_obj : mdtraj.trajectory
            The simulated solution trajectory
        solv_parmed_obj : parmed.structure
            Parmed object of the fully parameterised simulated solution system.
        liq_mdtraj_obj : mdtraj.trajectory
            The simulated liquid trajectory
        liq_parmed_obj : parmed.structure
            Parmed object of the fully parameterised simulated liquid system.
        smiles : str
            SMILES string of one copy of the solute. If mdfptools.Parameteriser was used during parameterisation, then smiles is automatically obtained from the parmed_obj.
        """
        cls.kwargs_solv = {"mdtraj_obj" : solv_mdtraj_obj ,
                        "parmed_obj" : solv_parmed_obj}
        cls.kwargs_liq = {"mdtraj_obj" : liq_mdtraj_obj ,
                        "parmed_obj" : liq_parmed_obj}
        cls.kwargs_liq = {**cls.kwargs_liq , **kwargs}
        cls.kwargs_solv = {**cls.kwargs_solv , **kwargs}

        if smiles is None:
            if solv_parmed_obj.title != '' and liq_parmed_obj.title != '': #try to obtain it from `parmed_obj`
                assert solv_parmed_obj.title == liq_parmed_obj.title, "Solution solute SMILES is not the same as the Liquid solute SMILES"
                smiles = solv_parmed_obj.title
            else:
                raise ValueError("Input ParMed Object does not contain SMILES string, add SMILES as an additional variable")
        return super().run(smiles)

    @classmethod
    def _get_relevant_properties(cls):
        """
        Where the set of features to be included in the final MDFP are defined
        """
        cls.fp  = {**cls.fp, **cls._get_2d_descriptors()}

        cls.fp  = {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_energies, **cls.kwargs_solv)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_rgyr, **cls.kwargs_solv)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_sasa, **cls.kwargs_solv)}

        cls.fp  = {**cls.fp, **cls._get_statistical_moments(LiquidExtractor.extract_energies, **cls.kwargs_liq)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(LiquidExtractor.extract_rgyr, **cls.kwargs_liq)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(LiquidExtractor.extract_sasa, **cls.kwargs_liq)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(LiquidExtractor.extract_dipole_magnitude, **cls.kwargs_liq)}

        del cls.kwargs_liq, cls.kwargs_solv

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
