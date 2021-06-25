from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
# from Per_Frame_Property_Extractor import *
from .Extractor import *

import numpy as np
from numpy import mean, std, median, isnan, array

import functools

"""
TODOs:
    - cls.smiles to cls.mol??? 
    - store a copy of smiles into the MDFP object?
    - lift substructural search in zwitterion out of function?
    - CustomComposer
    - how to customise which mean median std function to use per property?
"""
class MDFP():
    """
    A MDFP object contains a set of features for a molecule, obtaining from a simulation or a set of simulations.


    .. todo::
        - method to give back the keys
        - store some metdadata? e.g. which composer made it
        - Option to create a empty MDFP
        - load in experimental field
        - serialisation
    """
    def __init__(self, mdfp_dict):
        """
        Parameters
        ----------
        fp_dict : dict
            Keys are each of the type features a given `Extractor` obtains, e.g. "2d_count" are the 2D topological features obtained from molecule SMILES, "intra_lj" are the intra-molecular LJ energies obtained from simulation.

            Values are the corresponding set of numerics, stored as lists.
        """
        self.order = mdfp_dict.pop("__order__", None)
        self.mdfp = mdfp_dict
        self._metadata = {}

    @property
    def metadata(self):
        return self._metadata
    
    @metadata.setter
    def metadata(self, key, val):
        self.metadata[key] = val


    def get_mdfp(self, which_keys = None): #TODO update doc
        """
        Returns
        ----------
        a list of floating values, i.e. the mdfp feature vector
        """


        if which_keys:
            to_export = {i : self.mdfp[i]  for i in self.mdfp if i in which_keys}
        else:
            to_export = self.mdfp

        if hasattr(self, "order") and self.order:
            to_export = {i : self.mdfp[i] for i in self.order if i in self.mdfp}

        return functools.reduce(lambda a, b : a + b, to_export.values())

    def __str__(self):
        if hasattr(self, "order") and self.order: return str({i : self.mdfp[i] for i in self.order})
        else: return self.mdfp

class BaseComposer():
    """
    The BaseComposer class containing functions that can be used by different composers for different types of simulations

    """

    @classmethod
    def run(cls, smiles, order = None ):
        """
        Parameters
        ----------
        smiles : str
            SMILES string of the solute molecule
         order : #TODO
        """
        cls.smiles = smiles
        cls.fp = {"__order__" : order} if order else {}

        cls._get_relevant_properties()

        return MDFP(cls.fp)

    @classmethod
    def _get_relevant_properties(cls):
        """
        Where the set of features to be included in the final MDFP are defined
        """
        cls.fp  = {**cls.fp, **cls._get_2d_descriptors()}

    @classmethod
    def _calc_shape_from_smiles(cls):
        """ Calculate 2D-shape parameter of compounds from SMILES.
        The 2D-shape parameter describes the elongation of the molecule and is calculated as
        the ratio between the eigenvalues of the covariance matrix of the 2D coordinates. 
        (Help function for Properties2D_From_SMILES)


        Returns
        ----------
        2D-shape: float
           2D-shape parameter of the input molecule
        """
        mol = Chem.MolFromSmiles(cls.smiles)
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer( 0 )
        xyz_array = np.array(conf.GetPositions())
        xy_array = xyz_array[:,:2]
        xy_mean = np.mean(xy_array, axis=0)
        xy_array_rescale = np.subtract(xy_array, xy_mean)
        mean_vec = np.mean(xy_array_rescale, axis=0)
        cov_mat = (xy_array_rescale - mean_vec).T.dot((xy_array_rescale - mean_vec)) / (xy_array_rescale.shape[0]-1)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        ratio_eig_vals = eig_vals[0]/eig_vals[1]
        return ratio_eig_vals

    #XXX : is cmpd_name needed?
    @classmethod
    def _extract_zwitterionic_label(cls, dist_cutoff = None, cmpd_name = None):
        """
        It returns a dictionary containing a binary label for zwitterionic compounds.
        The label is "0" if the compound is not zwitterionic, while it is "1" otherwise.
        A compound is considered zwitterinic if it contains both a positively and a negatively charged groups and 
        if the groups are at a distance from each other smaller than the cutoff distance (dist_cutoff).
        If the cutoff distance is not specified, then every compound containing both a positively and a negatively charged groups
        is considered zwitterionic.
        If cmpd_name is specified, it is returned in the output dictionary.

        Parameters
        ---------- 
        Molecule: RDKit molecule
            Molecule read in using one of the RDKit functions. See RDKit documentation.
        dist_cutoff: float, optional
            Cutoff distance for the definition of a zwitterionic compound. See description above. (Default = None) 
        cmpd_name: str, optional
            Name of the compound. If specified, it is returned in the output dictionary. (Default = None) 

        Returns
        ----------
        dict_zwit: dict
            Dictionary key is "is_zwit". It contains a flag for zwitterionic compounds. See the description above.
            If cmpd_name is specified, it is returned in the dictionary.
        """

        mol = Chem.MolFromSmiles(cls.smiles)

        # SMARTS of charged groups
        NR4 = Chem.MolFromSmarts('[NX4+]')
        NX3_no_amide = Chem.MolFromSmarts('[NX3;!$([NX3][CX3](=[OX1])[#6])]')
        COO = Chem.MolFromSmarts('[CX3](=O)[O-]')
        SO3 = Chem.MolFromSmarts('[#16X4](=[OX1])(=[OX1])[O-]')
        nR3 = Chem.MolFromSmarts('[$([nX4+]),$([nX3]);!$(*=*)&!$(*:*)]')
        OR4 = Chem.MolFromSmarts('[O-]')
        SR4 = Chem.MolFromSmarts('[S-]')

        is_zwit = 0
        if mol:
            #check for carboxylate
            idx_COO = mol.GetSubstructMatches(COO)  #((19, 21, 20),)
            idx_COO_O = [id1[2] for id1 in idx_COO]
            # check for sulfonic acids
            idx_SO3 = mol.GetSubstructMatches(SO3)
            idx_SO3_O = [id1[3] for id1 in idx_SO3]
            # Thiols
            idx_SR4 = mol.GetSubstructMatches(SR4)
            idx_SR4_S = [id1[0] for id1 in idx_SR4]
            # Alchols
            idx_OR4 = mol.GetSubstructMatches(OR4)
            idx_OR4_O = [id1[0] for id1 in idx_OR4]
            negatively_charged_atoms = list(set(idx_COO_O + idx_SO3_O + idx_SR4_S + idx_OR4_O))
            if negatively_charged_atoms:
                #check for charged nitrogens
                idx_NR4 = mol.GetSubstructMatches(NR4)
                idx_NR4_n = [id1[0] for id1 in idx_NR4]
                idx_nR3 = mol.GetSubstructMatches(nR3)
                idx_nR3_n = [id1[0] for id1 in idx_nR3]
                idx_NR3 = mol.GetSubstructMatches(NX3_no_amide)
                idx_NR3_n = [id1[0] for id1 in idx_NR3]
                positively_charged_atoms = list(set(idx_NR4_n + idx_nR3_n + idx_NR3_n))
                #ionizable_atoms_tmp = idx_NR4_n + idx_nR3_n + idx_COO_O + idx_SO3_O + idx_SR4_S + idx_OR4_O + idx_NR3_n
                #ionizable_atoms = list(set(ionizable_atoms_tmp))
                if positively_charged_atoms: 
                    conf = mol.GetConformer( -1 )
                    dist = []
                    pts_pos =  np.array([ list( conf.GetAtomPosition(atmidx) ) for atmidx in positively_charged_atoms ])
                    pts_neg = np.array([ list( conf.GetAtomPosition(atmidx) ) for atmidx in negatively_charged_atoms ])
                    for pt1, pt2 in zip(pts_pos, pts_neg):
                        dist.append(np.linalg.norm(pt1 - pt2))

                    dist_min = min(dist)
                    if dist_cutoff != None:
                        if dist_min <= dist_cutoff:
                            is_zwit = 1
                        else:
                            is_zwit = 0
                    else:
                        is_zwit = 1

                else: 
                    is_zwit = 0
            else:
                is_zwit = 0
            
        # dict_zwit = {"is_zwit": is_zwit}  

        # if cmpd_name == None:
        #     return dict_zwit
        # else:
        #     dict_zwit.update({"cmpd_name": cmpd_name})
        #     return dict_zwit
        return is_zwit

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


    #XXX better way of including this
    @classmethod
    def _get_extended_2d_descriptors(cls, cmpd_name = None, includeSandP = True, **kwargs):
        """
        It returns a dictionary of 2D-counts and properties for the input molecule provided as SMILES. 
        These are: number of heavy atoms ("HA_count"), number of rotatable bonds ("RB_count"), number of N, O, F, P, S, Cl, Br, and I atoms ("ATOM_count" where ATOM is the element), molecular weight ("MW"), number of hydrogen bond donor ("HBD_count") and acceptor atoms ("HBA_count"), binary zwitterionic flag ("is_zwit"), 2D-shape parameter ("2d_shape"), and the topological polar surface area ("2d_psa"). 
        The 2D-shape parameter descibes the elongation of the molecule (see the function calc_shape_from_Mol for details).

        Parameters:
        ----------
        includeSandP: bool, optional
            Set to False to exclude the S and P atoms from the calculation of the TPSA. (Default = True) 
        dist_cutoff: float, optional
            Cutoff distance for the definition of a zwitterionic compound. (Default = None)  
            If the compound contains both a positively and a negatively charged chemical group and these are at a distance from each other smaller than the cutoff distance (dist_cutoff), 
            than the zwitterionic flag ("is_zwit") is set to 1. Otherwise the compound is considered not zwitterionic ("is_zwit" = 0).
            If the cutoff distance is not specified, then every compound containing both a positively and a negatively charged groups is considered zwitterionic. 
        cmpd_name: str, optional
            Name of the compound. If specified, it is returned in the output dictionary. (Default = None) 

        Returns
        ----------
        dict_2dcounts: dict
            Dictionary of 2D-counts and properties (see description above). 
            If cmpd_name is specified, it is returned in the dictionary.
        """

        molecule = Chem.MolFromSmiles(cls.smiles, sanitize = True)
        if molecule is None:
            molecule = Chem.MolFromSmiles(cls.smiles, sanitize = False)
            molecule.UpdatePropertyCache(strict=False)
            Chem.GetSSSR(molecule)

        # define SMARTS
        RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
        H_acceptor = Chem.MolFromSmarts('[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]')
        H_donor = Chem.MolFromSmarts('[!H0;#7,#8,#9,#16]')

        # Counts
        HA_count = molecule.GetNumHeavyAtoms()
        N_count  = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[#7]'))) # nitrogens
        O_count  = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[#8]'))) # oxygens
        F_count  = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[#9]'))) # fluorines
        P_count  = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[#15]'))) # phosphorous
        S_count  = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[#16]'))) # sulfurs
        Cl_count = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[#17]'))) # chlorines
        Br_count = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[#35]'))) # bromines
        I_count  = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[#53]'))) # iodines
        PSA2D = (Descriptors.TPSA(molecule, includeSandP=includeSandP))/100 #/100 to have nm instead of Angstrom
        N_donors = len(molecule.GetSubstructMatches(H_donor)) 
        N_acceptors = len(molecule.GetSubstructMatches(H_acceptor)) 
        RB_count  = len(molecule.GetSubstructMatches(RotatableBond))
        MW = Descriptors.ExactMolWt(molecule)
        shape_2d = cls._calc_shape_from_smiles()
        is_zwit = cls._extract_zwitterionic_label(**kwargs)
 
        #dict_2dcounts = {"MW": MW, "HA_count": HA_count, "RB_count": RB_count, "N_count": N_count, "O_count": O_count, "F_count": F_count, "P_count": P_count, "S_count": S_count, "Cl_count": Cl_count, "Br_count": Br_count, "I_count": I_count, "HBD_count": N_donors, "HBA_count": N_acceptors, "2d_shape": shape_2d, "2d_psa": PSA2D}
        dict_2dcounts = {"2d_counts" : [MW, HA_count, RB_count, N_count, O_count, F_count,P_count, S_count, Cl_count, Br_count, I_count, N_donors, N_acceptors, shape_2d, PSA2D, is_zwit]}

        return dict_2dcounts

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
            tmp = array(prop[i])
            for func in statistical_moments:
                 fp[i].append(func(tmp[~isnan(tmp)]))
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
    def run(cls, mdtraj_obj, parmed_obj, smiles = None, xvg_file_path = None, **kwargs):
        """
        Parameters
        -----------
        mdtraj_obj : mdtraj.trajectory
            The simulated trajectory
        parmed_obj : parmed.structure
            Parmed object of the fully parameterised simulated system.
        smiles : str
            SMILES string of the solute. If mdfptools.Parameteriser was used during parameterisation, then smiles is automatically obtained from the parmed_obj.
        xvg_file_path : #TODO
        """
        cls.kwargs = {"mdtraj_obj" : mdtraj_obj ,
                        "parmed_obj" : parmed_obj}
        
        if xvg_file_path is not None:
            cls.kwargs["energy_file_xvg"] = xvg_file_path

        cls.kwargs = {**cls.kwargs , **kwargs}
        if smiles is None:
            if parmed_obj.title != '': #try to obtain it from `parmed_obj`
                smiles = parmed_obj.title
            else:
                raise ValueError("Input ParMed Object {} does not contain SMILES string, add SMILES as an additional variable".format(parmed_obj))
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

WaterComposer = SolutionComposer

class LiquidComposer(BaseComposer):
    """
    Composer used to extract features from liquid simulations, namely a box containing replicates of the same molecule.
    """
    # def __init__(cls, smiles, mdtraj_obj, parmed_obj):
    @classmethod
    def run(cls, mdtraj_obj, parmed_obj, smiles = None, *args, **kwargs): #TODO better way of excluding unwanted arguments than *args
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
                raise ValueError("Input ParMed Object {} does not contain SMILES string, add SMILES as an additional variable".format(parmed_obj))
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
                raise ValueError("Input ParMed Object {} does not contain SMILES string, add SMILES as an additional variable".format(parmed_obj))
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

class Solution42BitsComposer(BaseComposer):
    """
    Composer used to extract features from solution simulations, namely one copy of solute in water solvent. This generates fingerprint most akin to that from the original publication.
    """
    @classmethod
    def run(cls, mdtraj_obj, parmed_obj = None, smiles = None, xvg_file_path = None, **kwargs):
        """
        Parameters
        -----------
        mdtraj_obj : mdtraj.trajectory
            The simulated trajectory
        parmed_obj : parmed.structure
            Parmed object of the fully parameterised simulated system.
        smiles : str
            SMILES string of the solute. If mdfptools.Parameteriser was used during parameterisation, then smiles is automatically obtained from the parmed_obj.
        xvg_file_path : #TODO
        """
        if parmed_obj is None and xvg_file_path is None:
            raise TypeError("At least one of parmed_obj or xvg_file_path should be supplied")

        if smiles is None:
            if parmed_obj.title != '': #try to obtain it from `parmed_obj`
                smiles = parmed_obj.title
            else:
                raise ValueError("Input ParMed Object {} does not contain SMILES string, add SMILES as an additional variable".format(parmed_obj))

        cls.kwargs = {"mdtraj_obj" : mdtraj_obj}

        if parmed_obj is not None:
            cls.kwargs["parmed_obj"] =  parmed_obj
        
        if xvg_file_path is not None:
            cls.kwargs["energy_file_xvg"] = xvg_file_path

        cls.kwargs = {**cls.kwargs , **kwargs}
        return super().run(smiles, order = ("2d_counts", "water_intra_crf", "water_intra_lj", "water_total_crf", "water_total_lj", "water_intra_ene", "water_total_ene", "water_rgyr", "water_sasa", "water_psa3d")) #TODO resolve the water/solution thing, maybe remove solutionextractor/solutioncomposer?

    @classmethod
    def _get_relevant_properties(cls):
        """
        Where the set of features to be included in the final MDFP are defined
        """
        cls.kwargs["solute_residue_name"] =  "LIG" 

        cls.fp  = {**cls.fp, **cls._get_extended_2d_descriptors()}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_energies_from_xvg, **cls.kwargs)} if "energy_file_xvg" in cls.kwargs else {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_energies, **cls.kwargs)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_rgyr, **cls.kwargs)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_sasa, **cls.kwargs)}
        cls.fp  = {**cls.fp, **cls._get_statistical_moments(WaterExtractor.extract_psa3d, **cls.kwargs)}
        del cls.kwargs

Water42BitsComposer = Solution42BitsComposer #TODO class factories????


class CustomComposer(BaseComposer):
    """
    handle to include arbitary properties given functions and extractors

    display all the properties that can be included
    """
    pass




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
