import mdtraj as md
from mdtraj.utils import ensure_type
from mdtraj.geometry import _geometry
from rdkit import Chem
import glob
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
import subprocess
import glob
from rdkit.Chem import AllChem
import numpy as np
from pymol import cmd
import os
from scipy.spatial import distance


class ComposerGMX:

    def __init__(self):
        pass


    @classmethod
    def gro2pdb(self, gro_file, pdb_output = None):
        """
        Converts a gro file into a pdb file. 

        Parameters
        ----------
        gro_file: str
            coordinates filename (.gro)
        pdb_output: str, optional
            name of the output pdb file. If not specified, then the pdb output file name be the same as the one of the gro_file.

        Returns
        ----------
        output.pdb: file
            pdb file
        """

        if pdb_output == None:
            pdb_output = os.path.splitext(gro_file)[0] + '.pdb'
        else:
            if str.split(pdb_output, '.')[-1] != "pdb":
                pdb_output = pdb_output + '.pdb'
         

        if os.path.isfile(pdb_output):
            return

        if os.path.isfile(gro_file):
            cmd.reinitialize()
            cmd.load(gro_file, 'test')
            cmd.save(pdb_output, 'test')
        
        if os.path.isfile(pdb_output):
            return
        else:
            try:
                args_subproc = "gmx editconf -f " + gro_file + " -o " + pdb_output + " -pbc no"
                subprocess.call(args_subproc, shell=True)
            except:
                print("Error: {} could not be converted to pdb".format(gro_file))
        

    @classmethod
    def solute_solvent_split(self, topology, solute_residue_name = None):
        """
        Obtain atom indexes of solute and solvent atoms from mdtraj topology file

        Parameters
        ----------
        topology: topology
            mdtraj topology file. It can be obtained from loaded structure or trajectory as object.topology
        solute_residue_name: str or list, optional
            name or list of names of the residue(s) to consider as solute (Default = 'LIG')

        Returns
        ----------
        solute_atoms: list
            List of indexes of the solute atoms
        solvent_atoms: list
            List of indexes of the solvent atoms
        """

        if solute_residue_name == None:
            solute_residue_name = 'LIG'

        if isinstance(solute_residue_name, list):
            solute_atoms = []
            for res_name in solute_residue_name:
                solute_atoms = solute_atoms + [atom.index for atom in topology.atoms if atom.residue.name == res_name]
        else:
            solute_atoms = [atom.index for atom in topology.atoms if atom.residue.name == solute_residue_name]
            if len(solute_atoms) == 0:
                print("The topology file does not containg any residue named '{}'. No solute atom extracted.".format(solute_residue_name))
        solvent_atoms = [atom.index for atom in topology.atoms if atom.residue.name != 'LIG']
        return solute_atoms, solvent_atoms



    @classmethod
    def get_stats(self, df):
        """Calcute mean, standard deviation, and median from list or array of numbers.

        Parameters
        ----------
        values: list or array
            list or array of numbers
        
        Returns
        ----------
        stats: list
            mean, standard deviation, and median of the provided input numbers
        """
        stats = [np.mean(df), np.std(df), np.median(df)]
        return stats


    @classmethod
    def shrake_rupley(self, traj, probe_radius=0.14, n_sphere_points=960, mode='residue', change_radii=None):
        """Compute the solvent accessible surface area of each atom or residue in each simulation frame. 
    
        Modified from MDTraj
    
        Parameters
        ----------
        traj : Trajectory
            An mtraj trajectory.
        probe_radius : float, optional
            The radius of the probe, in nm.
        n_sphere_points : int, optional
            The number of points representing the surface of each atom, higher
            values leads to more accuracy.
        mode : {'atom', 'residue'}
            In mode == 'atom', the extracted areas are resolved per-atom
            In mode == 'residue', this is consolidated down to the
            per-residue SASA by summing over the atoms in each residue.
        change_radii : dict, optional
            A partial or complete dict containing the radii to change from the 
            defaults. Should take the form {"Symbol" : radii_in_nm }, e.g. 
            {"Cl" : 0.181 } to change the radii of Chlorine to 181 pm for the ionic Cl-.
        Returns
        -------
        areas : np.array, shape=(n_frames, n_features)
            The accessible surface area of each atom or residue in every frame.
            If mode == 'atom', the second dimension will index the atoms in
            the trajectory, whereas if mode == 'residue', the second
            dimension will index the residues.
        """

        _ATOMIC_RADII = {'H'   : 0.120, 'He'  : 0.140, 'Li'  : 0.076, 'Be' : 0.059,
                         'B'   : 0.192, 'C'   : 0.170, 'N'   : 0.155, 'O'  : 0.152,
                         'F'   : 0.147, 'Ne'  : 0.154, 'Na'  : 0.102, 'Mg' : 0.086,
                         'Al'  : 0.184, 'Si'  : 0.210, 'P'   : 0.180, 'S'  : 0.180,
                         'Cl'  : 0.175, 'Ar'  : 0.188, 'K'   : 0.138, 'Ca' : 0.114,
                         'Sc'  : 0.211, 'Ti'  : 0.200, 'V'   : 0.200, 'Cr' : 0.200,
                         'Mn'  : 0.200, 'Fe'  : 0.200, 'Co'  : 0.200, 'Ni' : 0.163, 
                         'Cu'  : 0.140, 'Zn'  : 0.139, 'Ga'  : 0.187, 'Ge' : 0.211,
                         'As'  : 0.185, 'Se'  : 0.190, 'Br'  : 0.185, 'Kr' : 0.202, 
                         'Rb'  : 0.303, 'Sr'  : 0.249, 'Y'   : 0.200, 'Zr' : 0.200,
                         'Nb'  : 0.200, 'Mo'  : 0.200, 'Tc'  : 0.200, 'Ru' : 0.200,
                         'Rh'  : 0.200, 'Pd'  : 0.163, 'Ag'  : 0.172, 'Cd' : 0.158,  
                         'In'  : 0.193, 'Sn'  : 0.217, 'Sb'  : 0.206, 'Te' : 0.206,
                         'I'   : 0.198, 'Xe'  : 0.216, 'Cs'  : 0.167, 'Ba' : 0.149,
                         'La'  : 0.200, 'Ce'  : 0.200, 'Pr'  : 0.200, 'Nd' : 0.200,
                         'Pm'  : 0.200, 'Sm'  : 0.200, 'Eu'  : 0.200, 'Gd' : 0.200,
                         'Tb'  : 0.200, 'Dy'  : 0.200, 'Ho'  : 0.200, 'Er' : 0.200,
                         'Tm'  : 0.200, 'Yb'  : 0.200, 'Lu'  : 0.200, 'Hf' : 0.200,
                         'Ta'  : 0.200, 'W'   : 0.200, 'Re'  : 0.200, 'Os' : 0.200,
                         'Ir'  : 0.200, 'Pt'  : 0.175, 'Au'  : 0.166, 'Hg' : 0.155,
                         'Tl'  : 0.196, 'Pb'  : 0.202, 'Bi'  : 0.207, 'Po' : 0.197,
                         'At'  : 0.202, 'Rn'  : 0.220, 'Fr'  : 0.348, 'Ra' : 0.283,
                         'Ac'  : 0.200, 'Th'  : 0.200, 'Pa'  : 0.200, 'U'  : 0.186,
                         'Np'  : 0.200, 'Pu'  : 0.200, 'Am'  : 0.200, 'Cm' : 0.200,
                         'Bk'  : 0.200, 'Cf'  : 0.200, 'Es'  : 0.200, 'Fm' : 0.200,
                         'Md'  : 0.200, 'No'  : 0.200, 'Lr'  : 0.200, 'Rf' : 0.200,
                         'Db'  : 0.200, 'Sg'  : 0.200, 'Bh'  : 0.200, 'Hs' : 0.200,
                         'Mt'  : 0.200, 'Ds'  : 0.200, 'Rg'  : 0.200, 'Cn' : 0.200,
                         'Uut' : 0.200, 'Fl'  : 0.200, 'Uup' : 0.200, 'Lv' : 0.200,
                         'Uus' : 0.200, 'Uuo' : 0.200}


        xyz = ensure_type(traj.xyz, dtype=np.float32, ndim=3, name='traj.xyz', shape=(None, None, 3), warn_on_cast=False)

        if mode == 'atom':
            dim1 = xyz.shape[1]
            atom_mapping = np.arange(dim1, dtype=np.int32)

        elif mode == 'residue':
            dim1 = traj.n_residues
            if dim1 == 1:
                atom_mapping = np.array([0] * xyz.shape[1], dtype=np.int32)
            else:
                atom_mapping = np.array([a.residue.index for a in traj.top.atoms], dtype=np.int32)
                if not np.all(np.unique(atom_mapping) == np.arange(1 + np.max(atom_mapping))):
                    raise ValueError('residues must have contiguous integer indices starting from zero')
        else:
            raise ValueError('mode must be one of "residue", "atom". "{}" supplied'.format(mode))

    
        modified_radii = {}
        if change_radii is not None:
            # in case _ATOMIC_RADII is in use elsehwere...
            modified_radii = deepcopy(_ATOMIC_RADII)
            # Now, modify the values specified in 'change_radii'
            for k, v in change_radii.items(): modified_radii[k] = v


        out = np.zeros((xyz.shape[0], dim1), dtype=np.float32)
        atom_radii = []
        for atom in traj.topology.atoms: 
            atom_name = "{}".format(atom).split('-')[1]
            element = ''.join(i for i in atom_name if not i.isdigit())
            if bool(modified_radii):
                atom_radii.append(modified_radii[element])
            else:
                atom_radii.append(_ATOMIC_RADII[element])

        radii = np.array(atom_radii, np.float32) + probe_radius
    
        _geometry._sasa(xyz, radii, int(n_sphere_points), atom_mapping, out)
    
        return out


    @classmethod
    def calc_shape_from_SMILES(self, smiles):
        """ Calculate 2D-shape parameter of compounds from SMILES.
        The 2D-shape parameter describes the elongation of the molecule and is calculated as
        the ratio between the eigenvalues of the covariance matrix of the 2D coordinates. 
        (Help function for Properties2D_From_SMILES)

        Parameters
        ---------- 
        SMILES: SMILES
            SMILES of the molecule

        Returns
        ----------
        2D-shape: float
           2D-shape parameter of the input molecule
        """
        mol = Chem.MolFromSmiles(smiles)
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
        return(ratio_eig_vals)

    @classmethod
    def calc_shape_from_Mol(self, molecule):
        """ Calculate 2D-shape parameter of compounds from RDKit molecule.
        The 2D-shape parameter describes the elongation of the molecule and is calculated as
        the ratio between the eigenvalues of the covariance matrix of the 2D coordinates. 
        (Help function for Properties2D_From_Mol)

        Parameters
        ---------- 
        Molecule: RDKit molecule
            Molecule read in using one of the RDKit functions. See RDKit documentation.

        Returns
        ----------
        2D-shape: float
           2D-shape parameter of the input molecule
        """
        mol = molecule
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
        return(ratio_eig_vals)


    @classmethod
    def extract_Zwitterionic_Label_From_Mol(self, mol, dist_cutoff = None, cmpd_name = None):
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

        # SMARTS of charged groups
        NR4 = Chem.MolFromSmarts('[NX4+]')
        NX3_no_amide = Chem.MolFromSmarts('[NX3;!$([NX3][CX3](=[OX1])[#6])]')
        COO = Chem.MolFromSmarts('[CX3](=O)[O-]')
        SO3 = Chem.MolFromSmarts('[#16X4](=[OX1])(=[OX1])[O-]')
        nR3 = Chem.MolFromSmarts('[$([nX4+]),$([nX3]);!$(*=*)&!$(*:*)]')
        OR4 = Chem.MolFromSmarts('[O-]')
        SR4 = Chem.MolFromSmarts('[S-]')

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
                        dist.append(distance.euclidean(pt1, pt2))

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
            
        dict_zwit = {"is_zwit": is_zwit}  

        if cmpd_name == None:
            return(dict_zwit)
        else:
            dict_zwit.update({"cmpd_name": cmpd_name})
            return(dict_zwit)


    @classmethod
    def Properties2D_From_Mol(self, molecule, cmpd_name = None, includeSandP = None, dist_cutoff = None):
        """
        It returns a dictionary of 2D-counts and properties for the input RDKit molecule. 
        These are: number of heavy atoms ("HA_count"), number of rotatable bonds ("RB_count"), number of N, O, F, P, S, Cl, Br, and I atoms ("ATOM_count" where ATOM is the element), molecular weight ("MW"), number of hydrogen bond donor ("HBD_count") and acceptor atoms ("HBA_count"), binary zwitterionic flag ("is_zwit"), 2D-shape parameter ("2d_shape"), and the topological polar surface area ("2d_psa"). 
        The 2D-shape parameter descibes the elongation of the molecule (see the function calc_shape_from_Mol for details).

        Parameters:
        ----------
        Molecule: RDKit molecule
            Molecule read in using one of the RDKit functions. See RDKit documentation.
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

        if includeSandP == None:
            includeSandP = True

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
        MW = Chem.Descriptors.ExactMolWt(molecule)
        shape_2d = self.calc_shape_from_Mol(molecule)
        is_zwit = self.extract_Zwitterionic_Label_From_Mol(molecule, dist_cutoff=dist_cutoff)
        
        dict_2dcounts = {"MW": MW, "HA_count": HA_count, "RB_count": RB_count, "N_count": N_count, "O_count": O_count, "F_count": F_count, "P_count": P_count, "S_count": S_count, "Cl_count": Cl_count, "Br_count": Br_count, "I_count": I_count, "HBD_count": N_donors, "HBA_count": N_acceptors, "2d_shape": shape_2d, "2d_psa": PSA2D}

        dict_2dcounts.update(is_zwit)

        if cmpd_name == None:
            return(dict_2dcounts)
        else:
            dict_2dcounts.update({"cmpd_name": cmpd_name})
            return(dict_2dcounts)


    @classmethod
    def Properties2D_From_SMILES(self, smiles, cmpd_name = None, includeSandP = None):
        """
        It returns a dictionary of 2D-counts and properties for the input molecule provided as SMILES. 
        These are: number of heavy atoms ("HA_count"), number of rotatable bonds ("RB_count"), number of N, O, F, P, S, Cl, Br, and I atoms ("ATOM_count" where ATOM is the element), molecular weight ("MW"), number of hydrogen bond donor ("HBD_count") and acceptor atoms ("HBA_count"), binary zwitterionic flag ("is_zwit"), 2D-shape parameter ("2d_shape"), and the topological polar surface area ("2d_psa"). 
        The 2D-shape parameter descibes the elongation of the molecule (see the function calc_shape_from_Mol for details).

        Parameters:
        ----------
        SMILES: SMILES
            SMILES of the molecule
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

        if includeSandP == None:
            includeSandP = True

        molecule = Chem.MolFromSmiles(smiles)

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
        MW = Chem.Descriptors.ExactMolWt(molecule)
        shape_2d = self.calc_shape_from_Mol(molecule)
        is_zwit = self.extract_Zwitterionic_Label_From_Mol(molecule)
 
        dict_2dcounts = {"MW": MW, "HA_count": HA_count, "RB_count": RB_count, "N_count": N_count, "O_count": O_count, "F_count": F_count, "P_count": P_count, "S_count": S_count, "Cl_count": Cl_count, "Br_count": Br_count, "I_count": I_count, "HBD_count": N_donors, "HBA_count": N_acceptors, "2d_shape": shape_2d, "2d_psa": PSA2D}

        dict_2dcounts.update(is_zwit)

        if cmpd_name == None:
            return(dict_2dcounts)
        else:
            dict_2dcounts.update({"cmpd_name": cmpd_name})
            return(dict_2dcounts)


    @classmethod
    def extract_dipole_moments(self, solute_traj, prm_obj, solute_atoms, cmpd_name = None):
        """
        It returns Dipole Moment Terms.
        Especially, it returns a dictionary with mean, median and standard deviation of the dipole moment (mu)
        and of the dipole moment components along x-, y-, and z-axis (mu_x, mu_y, mu_z).
        It uses the dipole_moments function of MDTraj. 
        It requires as inputs the trajectory of the solute molecule, partial charges charges of the solute atoms, which are contained in the topology file, and the indices of the solute atoms.
        If cmpd_name is specified, it is returned in the dictionary.
        See also Tutorial2.ipynb.
 
        Parameters:
        ----------
        solute_traj: MDTraj trajectory object
            trajectory of the solute. The trajectory should be read in using the MDTraj functions. 
            Example: solute_traj = md.load(traj_file, top=pdb_file, atom_indices = solute_atoms)
        prm_obj: object containing structure's topology information
            This object can be obtained from the topology file (.top for GROMACS) using the parmed library. 
            For the GROMCS topology file, the command is: prm_obj = pmd.gromacs.GromacsTopologyFile(top_file).
        solute_atoms: list
            indeces of solute atoms. Needed to extract the partial charges of the solute atoms from the topology object.
        cmpd_name: str, optional
            Name of the compound. If specified, it is returned in the output dictionary. (Default = None) 

        Returns
        ----------
        dict_dip_mom: dict
            Dictionary containing mean, standard deviation, and median of the the dipole moment magnitude and components calculated over the simulation trajectory. 
            If cmpd_name is specified, it is returned in the dictionary.

        """

        solute_charges = [i.charge for idx,i in enumerate(prm_obj.atoms) if idx in solute_atoms ]
        dip_moments = md.dipole_moments(solute_traj, solute_charges)
        dipole_magnitude = np.sqrt(np.square(dip_moments).sum(axis=1))
        # <mu_x>, <mu_y>, <mu_z>, std_mu_x, std_mu_y, std_mu_z, med_mu_x, med_mu_y, med_mu_z, <mu>, std_mu, med_mu
        dip_components_av = list(dip_moments.mean(axis=0))  # <mu_x>, <mu_y>, <mu_z>
        dip_components_std = list(dip_moments.std(axis=0))  # std_mu_x, std_mu_y, std_mu_z
        dip_components_med = list(np.median(dip_moments, axis=0))  #med_mu_x, med_mu_y, med_mu_z
        dip_mom_stats = self.get_stats(dipole_magnitude)  #<mu>, std_mu, med_mu

        dict_dip_mom = {'av_mu_x': dip_components_av[0], 'av_mu_y': dip_components_av[1], 'av_mu_z': dip_components_av[2], 'std_mu_x': dip_components_std[0], 'std_mu_y': dip_components_std[1], 'std_mu_z': dip_components_std[2], 'med_mu_x': dip_components_med[0], 'med_mu_y': dip_components_med[1], 'med_mu_z': dip_components_med[2], 'av_mu': dip_mom_stats[0], 'std_mu': dip_mom_stats[1], 'med_mu': dip_mom_stats[2]}

        if cmpd_name == None:
            return(dict_dip_mom)
        else:
            dict_dip_mom.update({"cmpd_name": cmpd_name})
            return(dict_dip_mom)


    @classmethod
    def extract_rgyr(self, mdtraj_obj, cmpd_name = None):
        """
        It returns a dictionary containing Mean, Median and Standard Deviation of the Radius of Gyration (Rgyr).
        It uses the compute_rg function of MDTraj. 
        If cmpd_name is specified, it is returned in the dictionary.

        Parameters:
        ----------
        solute_traj: MDTraj trajectory object
            trajectory of the solute. The trajectory should be read in using the MDTraj functions. 
            Example: solute_traj = md.load(traj_file, top=pdb_file, atom_indices = solute_atoms)
        cmpd_name: str, optional
            Name of the compound. If specified, it is returned in the output dictionary. (Default = None) 

        Returns
        ----------
        dict_rgyr: dict
            Dictionary containing mean, standard deviation, and median of the Rgyr calculated over the simulation trajectory.
            If cmpd_name is specified, it is returned in the dictionary.
        """

        df = list(md.compute_rg(mdtraj_obj, masses = np.array([a.element.mass for a in mdtraj_obj.topology.atoms])))
        stats = list(self.get_stats(df))

        dict_rgyr = {'wat_rgyr_av': stats[0], 'wat_rgyr_std': stats[1], 'wat_rgyr_med': stats[2]}

        if cmpd_name == None:
            return(dict_rgyr)
        else:
            dict_rgyr.update({"cmpd_name": cmpd_name})
            return(dict_rgyr)
    

    @classmethod
    def extract_sasa(self, mdtraj_obj, cmpd_name = None):
        """
        It returns a dictionary containing Mean, Median and Standard Deviation of the Solvent Accessible Surface Area (SASA).
        It uses the shrake_rupley function of MDTraj. 
        If cmpd_name is specified, it is returned in the dictionary.

        Parameters:
        ----------
        solute_traj: MDTraj trajectory object
            trajectory of the solute. The trajectory should be read in using the MDTraj functions. 
            Example: solute_traj = md.load(traj_file, top=pdb_file, atom_indices = solute_atoms)
        cmpd_name: str, optional
            Name of the compound. If specified, it is returned in the output dictionary. (Default = None) 

        Returns
        ----------
        dict_rgyr: dict
            Dictionary containing mean, standard deviation, and median of the Rgyr calculated over the simulation trajectory.
            If cmpd_name is specified, it is returned in the dictionary.
        """

        df = list(self.shrake_rupley(mdtraj_obj, mode='residue'))
        stats = list(self.get_stats(df))

        dict_sasa = {'wat_sasa_av': stats[0], 'wat_sasa_std': stats[1], 'wat_sasa_med': stats[2]}
 
        if cmpd_name == None:
            return(dict_sasa)
        else:
            dict_sasa.update({"cmpd_name": cmpd_name})
            return(dict_sasa)
     
    
    @classmethod
    def read_xvg(self, energy_file_xvg):
        """
        It reads the GROMACS xvg file into a pandas dataframe.
        (Help function for EnergyTermsGmx).

        Parameters:
        ----------
        energy_file_xvg: str
            Name of the xvg file to read in

        Returns
        ----------
        energy_df: df
            Pandas dataframe of the energy terms and simulation parameters contained in the xvg file
        """
        if os.path.isfile(energy_file_xvg):
            data = pd.read_csv(energy_file_xvg, sep="\t", header = None)
            N_lines_header = data[data[0].str.contains('@')].shape[0] + data[data[0].str.contains('#')].shape[0]
            energy_df = pd.read_csv(energy_file_xvg, header = None, skiprows = N_lines_header, delim_whitespace=True)
            N_col = energy_df.shape[1]
            colnames = ["time"]
            for i in range(N_col-1):
                colnames.append(str.split(str(data[data[0].str.contains('s{}'.format(i))]), " ")[-1].replace('"',''))
            energy_df.columns = colnames
            return energy_df
        else:
            print("Error: Energy file {} does not exist".format(energy_file_xvg))
            return

    


    @classmethod
    def EnergyTermsGmx(self, energy_file_xvg, cmpd_name = None):
        """
        Energy Terms are extracted from the Gromacs xvg files. 
        It returns Energy Terms (as described in the original publication).
        Espacially, it returns a dictionary that contains as keys the mean, standard deviation and median of the energy terms.
        If cmpd_name is specified, it is returned in the dictionary.
    
        Parameters:
        --------------------
        energy_file_xvg: file
            Gromacs xvg file containing energy terms
        cmpd_name: str, optional
            Name of the compound. If specified, it is returned in the output dictionary.
        
        Returns
        ----------
        dict_ene: dict
            Keys are mean, standard deviation, and median of the energy terms. 
            If cmpd_name is specified, it is returned in the dictionary.
        """

        if os.path.isfile(energy_file_xvg):
            energy_df = self.read_xvg(energy_file_xvg)
        else:
            print("Error: Energy file {} does not exist".format(energy_file_xvg))
            return

        energy_array = np.array(energy_df)
        idx_coul_sr_lig = [i for i, s in enumerate(energy_df) if 'Coul-SR:LIG' in s][0] #5
        idx_coul_14_lig = [i for i, s in enumerate(energy_df) if 'Coul-14:LIG' in s][0] #7
        idx_lj_sr_lig = [i for i, s in enumerate(energy_df) if 'LJ-SR:LIG' in s][0] #6
        idx_lj_14_lig = [i for i, s in enumerate(energy_df) if 'LJ-14:LIG' in s][0] #8
        idx_coul_sr_wat_lig = [i for i, s in enumerate(energy_df) if 'Coul-SR:Water' in s][0] #1
        idx_coul_14_wat_lig = [i for i, s in enumerate(energy_df) if 'Coul-14:Water' in s][0] #3
        idx_lj_sr_wat_lig = [i for i, s in enumerate(energy_df) if 'LJ-SR:Water' in s][0] #2
        idx_lj_14_wat_lig = [i for i, s in enumerate(energy_df) if 'LJ-14:Water' in s][0] #4
        intra_crf_stats = self.get_stats(energy_array[:,idx_coul_sr_lig] + energy_array[:,idx_coul_14_lig]) 
        intra_lj_stats = self.get_stats(energy_array[:,idx_lj_sr_lig] + energy_array[:,idx_lj_14_lig]) 
        tot_crf_stats = self.get_stats(energy_array[:,idx_coul_sr_lig] + energy_array[:,idx_coul_14_lig] + energy_array[:,idx_coul_sr_wat_lig] + energy_array[:,idx_coul_14_wat_lig]) 
        tot_lj_stats = self.get_stats(energy_array[:,idx_lj_sr_lig] + energy_array[:,idx_lj_14_lig] + energy_array[:,idx_lj_sr_wat_lig] + energy_array[:,idx_lj_14_wat_lig])
        intra_ene_stats = self.get_stats(np.sum(energy_array[:,[idx_coul_sr_lig, idx_coul_14_lig, idx_lj_sr_lig, idx_lj_14_lig]], axis=1)) 
        tot_ene_stats = self.get_stats(np.sum(energy_array[:,[idx_coul_sr_lig, idx_coul_14_lig, idx_lj_sr_lig, idx_lj_14_lig, idx_coul_sr_wat_lig, idx_coul_14_wat_lig, idx_lj_sr_wat_lig, idx_lj_14_wat_lig]], axis=1))
        intra_crf_av = intra_crf_stats[0]
        intra_crf_std = intra_crf_stats[1]
        intra_crf_med = intra_crf_stats[2]
        intra_lj_av = intra_lj_stats[0]
        intra_lj_std = intra_lj_stats[1]
        intra_lj_med = intra_lj_stats[2]
        tot_crf_av = tot_crf_stats[0]
        tot_crf_std = tot_crf_stats[1]
        tot_crf_med = tot_crf_stats[2]
        tot_lj_av = tot_lj_stats[0]
        tot_lj_std = tot_lj_stats[1]
        tot_lj_med = tot_lj_stats[2]
        intra_ene_av = intra_ene_stats[0]
        intra_ene_std = intra_ene_stats[1]
        intra_ene_med = intra_ene_stats[2]
        tot_ene_av = tot_ene_stats[0]
        tot_ene_std = tot_ene_stats[1]
        tot_ene_med = tot_ene_stats[2]
        
        
        dict_ene = {'intra_crf_av_wat': intra_crf_av, 'intra_crf_std_wat': intra_crf_std, 'intra_crf_med_wat': intra_crf_med, 'intra_lj_av_wat': intra_lj_av, 'intra_lj_std_wat': intra_lj_std, 'intra_lj_med_wat': intra_lj_med, 'total_crf_av_wat': tot_crf_av, 'total_crf_std_wat': tot_crf_std, 'total_crf_med_wat': tot_crf_med, 'total_lj_av_wat': tot_lj_av, 'total_lj_std_wat': tot_lj_std, 'total_lj_med_wat': tot_lj_med, 'intra_ene_av_wat': intra_ene_av, 'intra_ene_std_wat': intra_ene_std, 'intra_ene_med_wat': intra_ene_med, 'total_ene_av_wat': tot_ene_av, 'total_ene_std_wat': tot_ene_std, 'total_ene_med_wat': tot_ene_med} 

        if cmpd_name != None:
            dict_ene.update({"cmpd_name": cmpd_name})

        return(dict_ene)
        
                
        
    @classmethod
    def calc_psa3d(self, obj_list=None, include_SandP: bool = True, atom_to_remove = None, solute_resname = 'LIG'):
        """
        Help function to calculate the 3d polar surface area (3D-PSA) of molecules in Interface_Pymol for all the snapshots in a MD trajectory.
        (Contribution by Benjamin Schroeder)
   
        Parameters
        ----------
        obj_list: list, optional
            list of pymol objects (Default = "cmpd1")
        include_SandP: bool, optional 
            Set to False to exclude the S and P atoms from the calculation of the 3D-PSA. (Default = True)
        atom_to_remove: str, optional
            Single atom name of the atom to remove from the selection (Default = None). 
            Useful if you want to include only S or only P in the calculation of the 3D-PSA.
        solute_resname: str, optional
            Residue name of the solute molecule. The 3D-PSA will be calculated only for this solute molecule (Default = 'LIG')

        Returns
        ---------- 
        obj_psa_dict: list
            Values correspond to mean, standard deviation, and median of the 3D-PSA calculated over the simulation time
        """

        # IO
        if (obj_list is None):
            obj_list = cmd.get_names("objects")
        if len(obj_list) > 1:
            print("Warning: More than one object available. The 3D-PSA will be returned only for the first object {}".format(obj_list[0]))
        # Select first (and only) object, which contains the trajectory for the solute molecule
        obj = obj_list[0]
        cmd.frame(0)
        states = range(1, cmd.count_states(obj) + 1)  # get all states of the object
        ##Loop over all states
        psa = []
        for state in states:
            ###select all needed atoms by partialCSelection or element or H next to (O ,N)
            if atom_to_remove != None and isinstance(atom_to_remove, str):
                if include_SandP:
                    select_string = "resn {} and (elem N or elem O or elem S or elem P or (elem H and (neighbor elem N+O+S+P))) and ".format(solute_resname) + obj + " and not name {}".format(atom_to_remove)  #@carmen add: "or elem S"
                else:
                    select_string = "resn {} and (elem N or elem O or (elem H and (neighbor elem N+O))) and ".format(solute_resname) + obj  + " and not name {}".format(atom_to_remove)  #@carmen add: "or elem S"
            else:
                if include_SandP:
                    select_string = "resn {} and (elem N or elem O or elem S or elem P or (elem H and (neighbor elem N+O+S+P))) and ".format(solute_resname) + obj   #@carmen add: "or elem S"
                else:
                    select_string = "resn {} and (elem N or elem O or (elem H and (neighbor elem N+O))) and ".format(solute_resname) + obj  #@carmen add: "or elem S"
            cmd.select("noh", select_string)
            ###calc surface area
            psa.append(float(cmd.get_area("noh", state=state)))
            
        ### calculate mean, standard deviation, and median 
        obj_psa_dict = [np.mean(psa)/100, np.std(psa)/100, np.median(psa)/100]  #/100 to have nm instead of Angstrom

        return obj_psa_dict


    @classmethod
    def extract_psa3d(self, traj_file, gro_file, obj_list=None, include_SandP = None, cmpd_name = None, atom_to_remove = None, solute_resname = 'LIG'):
        """ 
        Calculates the 3d polar surface area (3D-PSA) of molecules in Interface_Pymol for all the snapshots in a MD trajectory.
        (Contribution by Benjamin Schroeder)
   
        Parameters:
        -------------
        traj_file: str
            trajectory filename
        gro_file: str 
            coordinates filename (.gro or .pdb)
        cmpd_name: str, optional
            Name of the compound used as object name (default is "cmpd1")
        include_SandP: bool, optional 
            Set to False to exclude the S and P atoms from the calculation of the 3D-PSA. (Default = True)
        atom_to_remove: str, optional
            Single atom name of the atom to remove from the selection (Default = None). 
            Useful if you want to include only S or only P in the calculation of the 3D-PSA.
        solute_resname: str, optional
            Residue name of the solute molecule. The 3D-PSA will be calculated only for this solute molecule (Default = 'LIG')

        Returns
        ----------
        dict_psa3d: dict
            Keys are mean (3d_psa_av), standard deviation (3d_psa_sd), and median (3d_psa_med) of the 3D-PSA calculated over the simulation time. 
            If cmpd_name is specified, it is returned in the dictionary.
        """

        if cmpd_name == None:
            cmpd_name = "cmpd1"

        # Load trajectory and remove solvent and salts
        obj1 = cmpd_name 
        cmd.reinitialize()
        cmd.load(gro_file, object=obj1)
        cmd.load_traj(traj_file, object=obj1)
        cmd.remove("solvent")
        cmd.remove("resn Cl-")
        cmd.remove("resn Na+")
        cmd.remove("resn K+")
        
        atom_names = []
        cmd.iterate_state(-1, selection=obj1 + " and not elem H", expression="atom_names.append(name)", space=locals())
        total_psa = self.calc_psa3d(include_SandP = include_SandP, solute_resname = solute_resname, atom_to_remove = atom_to_remove)

        dict_psa3d = {'3d_psa_av': total_psa[0], '3d_psa_sd': total_psa[1], '3d_psa_med': total_psa[2]} 

        if cmpd_name == "cmpd1":
            return(dict_psa3d)
        else:
            dict_psa3d.update({"cmpd_name": cmpd_name})
            return(dict_psa3d)
    



