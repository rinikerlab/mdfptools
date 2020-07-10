import glob
import subprocess
import pandas as pd
import numpy as np
import mdtraj as md
import os
from mdtraj.utils import ensure_type
from mdtraj.geometry import _geometry
from pymol import cmd
#import parmed as pmd
#from simtk.openmm import app


class Extractor_PLMDFP:

    def __init__(self):
        pass

    ######################################################
    #
    #    FUNCTIONS IN COMMON WITH ComposerGMX
    #
    ######################################################


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
    def shrake_rupley(self, traj, probe_radius=0.14, n_sphere_points=960, mode='residue', change_radii=None):
        """Compute the solvent accessible surface area of each atom or residue in each simulation frame.
            
        Modified from mdtraj to ensure that even if the atom symbol column in the PDB file is missing, one computes the SASA correctly.
        If the symbol column in the PDB file is missing, the original mdtraj function does not recognize correctly the elements. For example, "Cl" will be recognized as "C".
        In addition, for "Cl" the default radius was set to the bounded one and not ionic one.
    
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
                colnames.append(str.split(str(data[data[0].str.contains('s{} '.format(i))]), " ")[-1].replace('"',''))
            energy_df.columns = colnames
            return energy_df
        else:
            print("Error: Energy file {} does not exist".format(energy_file_xvg))
            return


    ######################################################
    #
    #    NEW FUNCTIONS FOR Extractor_PLMDFP
    #
    ######################################################

    @classmethod
    def extract_protein_ligang_energy_terms_single(self, xvg_file = None, df_ene = pd.DataFrame(), cmpd_name = None, N_equil = None, return_df = False, solute_name = "LIG"):
        """
        Extracts from a single csv output file and returns mean, median and standard deviation of the following energetic components:
        - Ligand - Ligand    Coulomb, Lennard-Jones and Total Intramolecular Energy
        - Ligand - Protein   Coulomb, Lennard-Jones and Total Intermolecular Energy
        - Ligand - Water     Coulomb, Lennard-Jones and Total Intermolecular Energy (Retrieved only Water-Lig energy contributions were calculated and stored in the xvg file)
        - Total Coulomb, Lennard-Jones and Total contributions: Sum of the previous terms (including, eventually the Water-Ligand contributions)
        
        Either xvg_file or df_ene must be specified as input.

        Parameters:
        --------------------
        xvg_file: str, optional
            xvg file containing energy terms extracted from simulations using the "gmx energy" function of GROMACS
        df_ene: dataframe, optional 
            dataframe containing energy terms
        cmpd_name: str, optional
            Name of the compound to be analyzed. If provided, it is returned in the dict_ProtLig_ene dictionary.       
        N_equil: int, optional 
            Number of Equilibration Steps to discard from the calculation of the energetic contributions
        return_df: bool
            True to return the dataframe with energy values for each frame. Default is False
        solute_name: str, optional
            name of the solute molecule. Default is "LIG"

        Returns
        ---------- 
        dict_ProtLig_ene: dict
            Keys describe each of the energetic contributions
        df_ene: dataframe, optional
            Dataframe of energy values for each frame in the trajectory. Returned as second value only if return_df is True
        """

        # check that the input was specified
        if xvg_file == None and df_ene.empty:
            return print("Error in extract_protein_ligang_energy_terms_single: Either xvg_file or  must be specified as input")

        # chech that the file exists 
        if df_ene.empty and xvg_file != None:
            if not os.path.isfile(xvg_file):
                return print("Error in extract_protein_ligang_energy_terms_single: File {} does not exists".format(xvg_file))
            else:
                # Read energy file
                df_ene = self.read_xvg(xvg_file)


        if N_equil != None:
            df_ene = df_ene.iloc[N_equil:]

        flag_water = 0    
        # Extract energy terms and statistics
        # lig - lig intramolecular
        idx_coul_lig = [i for i, s in enumerate(df_ene) if ('Coul' in s and '{}-{}'.format(solute_name, solute_name) in s)]
        idx_lj_lig = [i for i, s in enumerate(df_ene) if ('LJ' in s and '{}-{}'.format(solute_name, solute_name) in s)]
        idx_ene_lig = [i for i, s in enumerate(df_ene) if '{}-{}'.format(solute_name, solute_name) in s]
        intra_lig_crf = self.get_stats(df_ene.iloc[:,idx_coul_lig].sum(axis = 1))
        intra_lig_lj = self.get_stats(df_ene.iloc[:,idx_lj_lig].sum(axis = 1))
        intra_lig = self.get_stats(df_ene.iloc[:,idx_ene_lig].sum(axis = 1))
        # lig - protein
        idx_coul_pl = [i for i, s in enumerate(df_ene) if ('Coul' in s and '{}-Protein'.format(solute_name) in s)]
        idx_lj_pl = [i for i, s in enumerate(df_ene) if ('LJ' in s and '{}-Protein'.format(solute_name) in s)]
        idx_ene_pl = [i for i, s in enumerate(df_ene) if '{}-Protein'.format(solute_name) in s]
        inter_prot_lig_crf = self.get_stats(df_ene.iloc[:,idx_coul_pl].sum(axis = 1))
        inter_prot_lig_lj = self.get_stats(df_ene.iloc[:,idx_lj_pl].sum(axis = 1))
        inter_prot_lig = self.get_stats(df_ene.iloc[:,idx_ene_pl].sum(axis = 1))
        # lig - water
        if len([b for b in list(df_ene.columns) if 'Water-{}'.format(solute_name) in b]) > 0: 
            idx_coul_wl = [i for i, s in enumerate(df_ene) if ('Coul' in s and 'Water-{}'.format(solute_name) in s)]
            idx_lj_wl = [i for i, s in enumerate(df_ene) if ('LJ' in s and 'Water-{}'.format(solute_name) in s)]
            idx_ene_wl = [i for i, s in enumerate(df_ene) if 'Water-{}'.format(solute_name) in s]
            inter_wat_lig_crf = self.get_stats(df_ene.iloc[:,idx_coul_wl].sum(axis = 1))
            inter_wat_lig_lj = self.get_stats(df_ene.iloc[:,idx_lj_wl].sum(axis = 1))
            inter_wat_lig = self.get_stats(df_ene.iloc[:,idx_ene_wl].sum(axis = 1))
            flag_water = 1
        # total
        idx_coul_tot = [i for i, s in enumerate(df_ene) if 'Coul' in s]
        idx_lj_tot = [i for i, s in enumerate(df_ene) if 'LJ' in s]
        idx_ene_tot = [i for i, s in enumerate(df_ene) if ('Coul' in s or 'LJ' in s)]
        inter_tot_crf = self.get_stats(df_ene.iloc[:,idx_coul_tot].sum(axis = 1))
        inter_tot_lj = self.get_stats(df_ene.iloc[:,idx_lj_tot].sum(axis = 1))
        inter_tot = self.get_stats(df_ene.iloc[:,idx_ene_tot].sum(axis = 1))
            
        dict_ProtLig_ene = {'intra_crf_pl_av': intra_lig_crf[0], 'intra_crf_pl_std': intra_lig_crf[1], 'intra_crf_pl_med': intra_lig_crf[2], 'intra_lig_lj_pl_av': intra_lig_lj[0], 'intra_lig_lj_pl_std': intra_lig_lj[1], 'intra_lig_lj_pl_med': intra_lig_lj[2], 'intra_ene_pl_av': intra_lig[0], 'intra_ene_pl_std': intra_lig[1], 'intra_ene_pl_med': intra_lig[2], 'prot_lig_crf_pl_av': inter_prot_lig_crf[0], 'prot_lig_crf_pl_std':inter_prot_lig_crf[1], 'prot_lig_crf_pl_med':inter_prot_lig_crf[2], 'prot_lig_lj_pl_av':inter_prot_lig_lj[0], 'prot_lig_lj_pl_std':inter_prot_lig_lj[1], 'prot_lig_lj_pl_med':inter_prot_lig_lj[2], 'prot_lig_ene_pl_av':inter_prot_lig[0], 'prot_lig_ene_pl_std':inter_prot_lig[1], 'prot_lig_ene_pl_med':inter_prot_lig[2], 'tot_crf_pl_av': inter_tot_crf[0], 'tot_crf_pl_std':inter_tot_crf[1], 'tot_crf_pl_med':inter_tot_crf[2], 'tot_lj_pl_av':inter_tot_lj[0], 'tot_lj_pl_std':inter_tot_lj[1], 'tot_lj_pl_med':inter_tot_lj[2], 'tot_ene_pl_av':inter_tot[0], 'tot_ene_pl_std':inter_tot[1], 'tot_ene_pl_med':inter_tot[2]}   

        if flag_water == 1: 
            dict_ProtLig_ene.update({'wat_lig_crf_pl_av': inter_wat_lig_crf[0], 'wat_lig_crf_pl_std':inter_wat_lig_crf[1], 'wat_lig_crf_pl_med':inter_wat_lig_crf[2], 'wat_lig_lj_pl_av':inter_wat_lig_lj[0], 'wat_lig_lj_pl_std':inter_wat_lig_lj[1], 'wat_lig_lj_pl_med':inter_wat_lig_lj[2], 'wat_lig_ene_pl_av':inter_wat_lig[0], 'wat_lig_ene_pl_std':inter_wat_lig[1], 'wat_lig_ene_pl_med':inter_wat_lig[2]})

        if cmpd_name != None:
            dict_ProtLig_ene.update({'cmpd_name': cmpd_name})

        if return_df:
            return dict_ProtLig_ene, df_ene
        else:
            return dict_ProtLig_ene 
    
       
    @classmethod
    def extract_protein_ligang_energy_terms_multi(self, list_xvg_files, cmpd_name = None, N_equil = None, return_df = False, solute_name = "LIG"):
        """
        Extracts from multiple csv output files (one for each replica of the same compound) the following energetic components:
        - Ligand - Ligand    Coulomb, Lennard-Jones and Total Intramolecular Energy
        - Ligand - Protein   Coulomb, Lennard-Jones and Total Intermolecular Energy
        - Ligand - Water     Coulomb, Lennard-Jones and Total Intermolecular Energy (Retrieved only Water-Lig energy contributions were calculated and stored in the xvg file)
        - Total Coulomb, Lennard-Jones and Total contributions: Sum of the previous terms (including, eventually the Water-Ligand contributions)
        It returns:
        - mean, median and standard deviation of the energy terms for each of the replicas
        - mean, median and standard deviation of the energy terms obtained merging all the replicas 
 
        Parameters:
        --------------------
        list_xvg_files: list
            List of xvg file names 
        cmpd_name: str, optional
            Name of the compound to be analyzed. If provided, it is returned in the dict_ProtLig_ene dictionary.       
        N_equil: int, optional 
            Number of Equilibration Steps to discard from the calculation of the energetic contributions
        return_df: bool
            True to return the dataframe with energy values for each frame. Default is False
        solute_name: str, optional
            name of the solute molecule. Default is "LIG"

        Returns
        ---------- 
        dict_ProtLig_ene_multi: dict
            Keys describe each of the energetic contributions
        df_ene: dataframe, optional
            Dataframe of energy values for each frame in the trajectory (values from multiple trajectories are concatenated). Returned as second value only if return_df is True
        """

        list_xvg_files.sort()

        df_ene_all = pd.DataFrame()
        dict_ProtLig_ene_multi = {}
        for count, xvg1 in enumerate(list_xvg_files):
            dict_ene_PL_single = self.extract_protein_ligang_energy_terms_single(xvg_file = xvg1, N_equil = N_equil, solute_name = solute_name)
            dict_ene_PL_single = {k+'_{}'.format(count): v for k, v in dict_ene_PL_single.items()}
            dict_ProtLig_ene_multi.update(dict_ene_PL_single)
            # dataframe of all energies
            df_ene_tmp = self.read_xvg(xvg1)
            if N_equil != None:
                df_ene_tmp = df_ene_tmp.iloc[N_equil:] 
            df_ene_all = df_ene_all.append(df_ene_tmp)
        
        df_ene_all.reset_index(inplace = True, drop = True)

        dict_ene_PL_overall = self.extract_protein_ligang_energy_terms_single(df_ene = df_ene_all, solute_name = solute_name)
        dict_ene_PL_overall = {k+'_all_replicas': v for k, v in dict_ene_PL_overall.items()}
         
        # Read energy file

        dict_ProtLig_ene_multi.update(dict_ene_PL_overall)

        if cmpd_name != None:
            dict_ProtLig_ene_multi.update({'cmpd_name': cmpd_name})

        if return_df:
            return dict_ProtLig_ene_multi, df_ene_all
        else:
            return dict_ProtLig_ene_multi
           

    @classmethod
    def extract_min_and_mean_protein_ligang_energy(self, list_xvg_files, cmpd_name = None, N_equil = None, return_df = False, return_min_filename = False, solute_name = "LIG"):
        """
        Extracts from multiple csv output files (one for each replica of the same compound) the following energetic components:
        - Ligand - Ligand    Coulomb, Lennard-Jones and Total Intramolecular Energy
        - Ligand - Protein   Coulomb, Lennard-Jones and Total Intermolecular Energy
        - Ligand - Water     Coulomb, Lennard-Jones and Total Intermolecular Energy (Retrieved only Water-Lig energy contributions were calculated and stored in the xvg file)
        - Total Coulomb, Lennard-Jones and Total contributions: Sum of the previous terms (including, eventually the Water-Ligand contributions)
        It returns:
        - mean, median and standard deviation of the energy terms for the replica with the minimum total energy (intra_lig + prot_lig)
        - mean, median and standard deviation of the energy terms obtained merging all the replicas 
 
        Parameters:
        --------------------
        list_xvg_files: list
            List of xvg file names 
        cmpd_name: str, optional
            Name of the compound to be analyzed. If provided, it is returned in the dict_ProtLig_ene dictionary.       
        N_equil: int, optional 
            Number of Equilibration Steps to discard from the calculation of the energetic contributions
        return_df: bool, optional
            True to return the dataframe with energy values for each frame. Default is False
        return_min_filename: bool, optional
            True to return the name of the xvg file corresponding to the replica with the minimum total energy (intra_lig + prot_lig)
        solute_name: str, optional
            name of the solute molecule. Default is "LIG"

        Returns
        ---------- 
        dict_ProtLig_ene_multi: dict
            Keys describe each of the energetic contributions
        df_ene: dataframe, optional
            Dataframe of energy values for each frame in the trajectory (values from multiple trajectories are concatenated). Returned as second value only if return_df is True
        xvg_file: str, optional
            Name of the xvg file corresponding to the replica with the minimum total energy. Returned as second/third value only if return_min_filename is True. 
        """

        list_xvg_files.sort()

        dict_tot_ene_PL = {}
        df_ene_all = pd.DataFrame()
        dict_ProtLig_ene_multi = {}

        for count, xvg1 in enumerate(list_xvg_files):
            dict_ene_PL_single = self.extract_protein_ligang_energy_terms_single(xvg_file = xvg1, N_equil = N_equil, solute_name = solute_name)
            dict_tot_ene_PL[xvg1] = dict_ene_PL_single['intra_ene_pl_av'] + dict_ene_PL_single['prot_lig_ene_pl_av']
            #dict_ene_PL_single = {k+'_{}'.format(count): v for k, v in dict_ene_PL_single.items()}
            dict_ProtLig_ene_multi[xvg1] = dict_ene_PL_single
            # dataframe of all energies
            df_ene_tmp = self.read_xvg(xvg1)
            if N_equil != None:
                df_ene_tmp = df_ene_tmp.iloc[N_equil:] 
            df_ene_all = df_ene_all.append(df_ene_tmp)


        # get statistics overall replicas 
        df_ene_all.reset_index(inplace = True, drop = True)
        dict_ene_PL_overall = self.extract_protein_ligang_energy_terms_single(df_ene = df_ene_all, solute_name = solute_name)
        dict_ene_PL_overall = {k+'_all_replicas': v for k, v in dict_ene_PL_overall.items()}
         
        # get pose with minimum energy
        min_pose = min(dict_tot_ene_PL, key=dict_tot_ene_PL.get)
        dict_min = dict_ProtLig_ene_multi[min_pose]
        dict_min = {k+'_min_replica': v for k, v in dict_min.items()}
        
        dict_ene_PL_overall.update(dict_min)

        if cmpd_name != None:
            dict_ene_PL_overall.update({'cmpd_name': cmpd_name})

        if return_df:
            if return_min_filename:
                return dict_ene_PL_overall, df_ene_all, min_pose
            else:
                return dict_ene_PL_overall, df_ene_all
        else:
            if return_min_filename:
                return dict_ene_PL_overall, min_pose
            else:
                return dict_ene_PL_overall
           

    @classmethod
    def extract_residue_ligand_energy_terms(self, xvg_file = None , df_ene = pd.DataFrame(), cmpd_name = None, residue_number = None, N_equil = None, return_df = False):
        """
        Extract from simulations and returns mean, median and standard deviation of the energetic components of single residue - ligand interactions. 
        Those are Coulomb, Lennard-Jones and Total Intermolecular Energy.
    
        Either xvg_file or df_ene must be specified as input!!!

        Parameters:
        --------------------
        xvg_file: str, optional
            xvg file containing energy terms extracted from simulations using the gmx_energy function of GROMACS
        df_ene: dataframe, optional 
            dataframe containing energy terms
        cmpd_name: str, optional
            Name of the compound to be analyzed. If provided, it is returned in the dict_ResLig_ene dictionary in the "cmpd_name" key.       
        residue_number: int, str
            Number of the residue being analysed. If provided, it is returned in the dict_ResLig_ene dictionary in each of the keys as "resXXX".
        N_equil: int, optional 
            Number of Equilibration Steps to discard from the calculation of the energetic contributions
        return_df: bool
            True to return the dataframe with energy values for each frame. Default is False

        Returns
        ---------- 
        dict_ResLig_ene: dict
            Keys describe each of the energetic contributions
        df_ene: dataframe, optional
            Dataframe of energy values for each frame in the trajectory. Returned as second value only if return_df is True
        """

        # check that the input was specified
        if xvg_file == None and df_ene.empty:
            return print("Error in extract_protein_ligang_energy_terms_single: Either xvg_file or  must be specified as input")

        # chech that the file exists 
        if df_ene.empty and  xvg_file != None:
            if not os.path.isfile(xvg_file):
                return print("Error in extract_protein_ligang_energy_terms_single: File {} does not exists".format(xvg_file))
            else:
                # Read energy file
                df_ene = self.read_xvg(xvg_file)
        

        if N_equil != None:
            df_ene = df_ene.iloc[N_equil:]
            
        # Extract energy terms and statistics
        idx_coul_tot = [i for i, s in enumerate(df_ene) if 'Coul' in s]
        idx_lj_tot = [i for i, s in enumerate(df_ene) if 'LJ' in s]
        idx_ene_tot = [i for i, s in enumerate(df_ene) if ('Coul' in s or 'LJ' in s)]
        inter_res_crf = self.get_stats(df_ene.iloc[:,idx_coul_tot].sum(axis = 1))
        inter_res_lj = self.get_stats(df_ene.iloc[:,idx_lj_tot].sum(axis = 1))
        inter_tot = self.get_stats(df_ene.iloc[:,idx_ene_tot].sum(axis = 1))

        if residue_number == None and xvg_file != None:
            residue_number = self.guess_res_number(xvg_file)

        if residue_number != None:
            dict_ResLig_ene = {'res{}_crf_pl_av'.format(residue_number): inter_res_crf[0], 'res{}_crf_pl_std'.format(residue_number): inter_res_crf[1], 'res{}_crf_pl_med'.format(residue_number) :inter_res_crf[2], 'res{}_lj_pl_av'.format(residue_number): inter_res_lj[0], 'res{}_lj_pl_std'.format(residue_number): inter_res_lj[1], 'res{}_lj_pl_med'.format(residue_number): inter_res_lj[2], 'res{}_ene_pl_av'.format(residue_number): inter_tot[0], 'res{}_ene_pl_std'.format(residue_number): inter_tot[1], 'res{}_ene_pl_med'.format(residue_number): inter_tot[2]}
        else:
            dict_ResLig_ene = {'res_crf_pl_av': inter_res_crf[0], 'res_crf_std_pl':inter_res_crf[1], 'res_crf_med_pl':inter_res_crf[2], 'res_lj_pl_av':inter_res_lj[0], 'res_lj_std_pl':inter_res_lj[1], 'res_lj_med_pl':inter_res_lj[2], 'res_ene_pl_av':inter_tot[0], 'res_ene_std_pl':inter_tot[1], 'res_ene_med_pl':inter_tot[2]}


        if cmpd_name != None:
            dict_ResLig_ene.update({'cmpd_name': cmpd_name})

        if return_df:
            return dict_ResLig_ene, df_ene 
        else:
            return dict_ResLig_ene 


 
    @classmethod
    def extract_residue_ligand_energy_terms_multi_supplier(self, list_xvg_files, cmpd_name = None, list_residue_numbers = None, N_equil = None, return_df = False):
        """
        Extract from simulations and returns mean, median and standard deviation of the energetic components of single residue - ligand interactions
        for all files in list_xvg_files. If list_residue_numbers is specified it extracts the energy terms only for the residues in the list.
        If multiple files are available for each of the residues (e.g. multiple replicas), it also returns mean, median and standard deviation calculated over all replicas.
    
        Parameters:
        --------------------
        list_xvg_files: list
            List of xvg file names 
        cmpd_name: str
            Name of the compound to be analyzed. If provided, it is returned in the dict_ResLig_ene dictionary in the "cmpd_name" key.       
        list_residue_numbers: int, str
            Number of the residue being analysed. If provided, it is returned in the dict_ResLig_ene dictionary in each of the keys as "resXXX".
        N_equil: int, optional 
            Number of Equilibration Steps to discard from the calculation of the energetic contributions
        return_df: bool
            True to return the dataframe with energy values for each frame. Default is False

        Returns
        ---------- 
        dict_ResLig_ene: dict
            Keys describe each of the energetic contributions
        df_ene: dataframe, optional
            Dataframe of energy values for each frame in the trajectory (values from multiple trajectories are concatenated). Returned as second value only if return_df is True
        """

        list_xvg_files.sort()

        dict_ResLig_ene_multi = {}
        df_ResLig_ene_multi = pd.DataFrame()
        if list_residue_numbers != None:
            for res1 in list_residue_numbers:
                xvg_files_res1 = [x for x in list_xvg_files if "res_{}_".format(res1) in x]
                df_ene_all = pd.DataFrame()
                if len(xvg_files_res1) > 1:
                    # calculate energy contributions for each residue and each replica
                    for count, xvg1 in enumerate(xvg_files_res1):
                        dict_ResLig_ene = self.extract_residue_ligand_energy_terms(xvg_file = xvg1, residue_number = res1, N_equil = N_equil)
                        dict_ResLig_ene = {k+'_{}'.format(count): v for k, v in dict_ResLig_ene.items()}
                        dict_ResLig_ene_multi.update(dict_ResLig_ene)
                        # dataframe of all energies per residue
                        df_ene_all_tmp2 = self.read_xvg(xvg1)
                        df_ene_all_tmp2.reset_index(inplace = True, drop = True)
                        if N_equil != None:
                            df_ene_all_tmp2 = df_ene_all_tmp2.iloc[N_equil:]            
                        df_ene_all = df_ene_all.append(df_ene_all_tmp2) 
                    # calculate statistics for each residue overall replicas
                    df_ene_all.reset_index(inplace = True, drop = True) 
                    dict_ene_RL_overall = self.extract_residue_ligand_energy_terms(df_ene = df_ene_all, residue_number = res1)
                    dict_ene_RL_overall = {k+'_all_replicas'.format(count): v for k, v in dict_ene_RL_overall.items()}
                    dict_ResLig_ene_multi.update(dict_ene_RL_overall) 
                    df_ResLig_ene_multi = pd.concat([df_ResLig_ene_multi, df_ene_all], axis =1)
                                     
                elif len(xvg_files_res1) == 1:
                    # calculate energy contributions for each residue. Only one replica available
                    dict_ResLig_ene = self.extract_residue_ligand_energy_terms(xvg_file = xvg_files_res1[0], residue_number = res1, N_equil = N_equil)
                    #dict_ResLig_ene = {k+'_0': v for k, v in dict_ResLig_ene.items()}
                    dict_ResLig_ene_multi.update(dict_ResLig_ene)
                    df_ene_all = self.read_xvg(xvg_files_res1[0])
                    df_ene_all.reset_index(inplace = True, drop = True)
                    df_ResLig_ene_multi = pd.concat([df_ResLig_ene_multi, df_ene_all], axis =1)
                elif len(xvg_files_res1) == 0:
                    print("Error: No energy files in list_xvg_files for residue {}".format(res1))

            if cmpd_name != None:
                dict_ResLig_ene_multi.update({'cmpd_name': cmpd_name})

            if return_df and not df_ResLig_ene_multi.empty:
                return dict_ResLig_ene_multi, df_ResLig_ene_multi
            else:
                return dict_ResLig_ene_multi

        else:
            print("list_residue_numbers was not provided. Calculating energy contributions for each of the files in list_xvg_files.") 
            print("The function will try to guess the residue numner from the file name. To prevent that the output is not informative, a nested dictionary is returned with xvg file names as main keys") 
            dict_ResLig_ene_multi = {}
            for xvg1 in list_xvg_files:
                res1 = self.guess_res_number(xvg1)
                if isinstance(res1, int):
                    dict_ResLig_ene = self.extract_residue_ligand_energy_terms(xvg_file = xvg1, residue_number = res1, N_equil = N_equil)
                else: 
                    dict_ResLig_ene = self.extract_residue_ligand_energy_terms(xvg_file = xvg1, N_equil = N_equil)

                dict_ResLig_ene_multi[xvg1] = dict_ResLig_ene
                
            if return_df:
                print("Warning: the dataframe containing the energy contributuions is not returned. Not implemented yet.")
                
            return dict_ResLig_ene_multi 
           

    @classmethod
    def guess_res_number(self, xvg_file: str):
        x = xvg_file.split('_')
        if "res" not in x: 
            return None
        ns = [i for i,s in enumerate(x) if "res" in s]
        if x[ns[0]] == 'res':
            Nres = int(x[ns[0] +1])
            if 'Nres' not in locals():
                Nres = None
                print("Error: The residue number for {} could not be guessed".format(xvg_file))
        else:
            Nres = None
            print("Error: The residue number for {} could not be guessed".format(xvg_file))
        return(Nres)


    @classmethod
    def extract_rgyr(self, mdtraj_obj, cmpd_name = None, N_equil = None, return_df = False):
        """
        Extracts radius of gyration (rgyr) from each frame of the simulation and returns a dictionary containing Mean, Median and Standard Deviation of rgyr.

        Parameters:
        --------------------
        mdtraj_obj: obj
            trajectory of the solute 
        cmpd_name: str
            Name of the compound to be analyzed. If provided, it is returned in the dict_rgyr dictionary in the "cmpd_name" key.       
        N_equil: int, optional 
            Number of Equilibration Steps to discard from the calculation of the energetic contributions
        return_df: bool
            True to return the dataframe with the values of Rgyr for each frame. Default is False

        Returns:
        --------------------
        dict_rgyr: dict
        df: dataframe, optional
            Dataframe of Rgyr values for each frame in the trajectory. Returned as second value only if return_df is True
        """

        df = list(md.compute_rg(mdtraj_obj, masses = np.array([a.element.mass for a in mdtraj_obj.topology.atoms])))

        if N_equil != None:
            df = df[N_equil:]
 
        stats = list(self.get_stats(df))

        dict_rgyr = {'rgyr_pl_av': stats[0], 'rgyr_pl_std': stats[1], 'rgyr_pl_med': stats[2]}

        if cmpd_name != None:
            dict_rgyr.update({"cmpd_name": cmpd_name})

        if return_df:
            return(dict_rgyr, df)
        else:
            return(dict_rgyr)
    

    @classmethod
    def extract_rgyr_multi(self, list_mdtraj_obj, cmpd_name = None, N_equil = None, return_df = False):
        """
        Extracts radius of gyration (rgyr) from every frame of each trajectory.
        It returns a dictionary containing Mean, Median and Standard Deviation of rgyr for both the single trajectories and the concatenated one.

        Parameters:
        --------------------
        mdtraj_obji_list: list
            List of solute trajectory objects 
        cmpd_name: str
            Name of the compound to be analyzed. If provided, it is returned in the dict_rgyr dictionary in the "cmpd_name" key.       
        N_equil: int, optional 
            Number of Equilibration Steps to discard from the calculation of the energetic contributions
        return_df: bool
            True to return the dataframe with Rgyr values for each frame. Default is False

        Returns:
        --------------------
        dict_rgyr_multi: dict
        df: dataframe, optional
            Dataframe of Rgyr values for each frame in the trajectory (values from multiple trajectories are concatenated). Returned as second value only if return_df is True

        """

        dict_rgyr_multi = {}
        df_rgyr_multi = pd.DataFrame()
        for count, trj1 in enumerate(list_mdtraj_obj): 
            dict_rgyr, df_rgyr = self.extract_rgyr(trj1, N_equil = N_equil, return_df = True) 
            dict_rgyr = {k+'_{}'.format(count): v for k, v in dict_rgyr.items()}
            dict_rgyr_multi.update(dict_rgyr)
            df_rgyr_multi = df_rgyr_multi.append(df_rgyr)

        # calculate statistics over all replicas
        stats = list(self.get_stats(df_rgyr_multi.values.tolist()))
        dict_rgyr = {'rgyr_pl_av': stats[0], 'rgyr_pl_std': stats[1], 'rgyr_pl_med': stats[2]}
        dict_rgyr = {k+'_all_replicas': v for k, v in dict_rgyr.items()}
        dict_rgyr_multi.update(dict_rgyr)
 
        if cmpd_name != None:
            dict_rgyr_multi.update({'cmpd_name': cmpd_name})

        if return_df:
            df_rgyr_multi.reset_index(inplace = True, drop = True)
            df_rgyr_multi.columns = ['rgyr']
            return dict_rgyr_multi, df_rgyr_multi
        else:
            return dict_rgyr_multi
        


    @classmethod
    def extract_sasa(self, mdtraj_obj, cmpd_name = None, N_equil = None, return_df = False):
        """
        Extracts solvent accessible surface area (SASA) from each frame of the simulation and returns a dictionary containing Mean, Median and Standard Deviation of SASA.

        Parameters:
        --------------------
        mdtraj_obj: obj
            trajectory of the solute 
        cmpd_name: str
            Name of the compound to be analyzed. If provided, it is returned in the dict_rgyr dictionary in the "cmpd_name" key.       
        N_equil: int, optional 
            Number of Equilibration Steps to discard from the calculation of the energetic contributions

        Returns:
        --------------------
        dict_sasa: dict
        df: dataframe, optional
            Dataframe of SASA values for each frame in the trajectory. Returned as second value only if return_df is True

        """

        df = list(self.shrake_rupley(mdtraj_obj, mode='residue'))

        if N_equil != None:
            df = df[N_equil:]
 
        stats = list(self.get_stats(df))

        dict_sasa = {'sasa_pl_av': stats[0], 'sasa_pl_std': stats[1], 'sasa_pl_med': stats[2]}
 
        if cmpd_name != None:
            dict_sasa.update({"cmpd_name": cmpd_name})

        if return_df:
            return(dict_sasa, df)
        else:
            return(dict_sasa)
     
    
    @classmethod
    def extract_sasa_multi(self, list_mdtraj_obj, cmpd_name = None, N_equil = None, return_df = False):
        """
        Extracts solvent accessible surface area (SASA) from every frame of each trajectory.
        It returns a dictionary containing Mean, Median and Standard Deviation of SASA for both the single trajectories and the concatenated one.

        Parameters:
        --------------------
        list_mdtraj_obj: list
            List of solute trajectory objects
        cmpd_name: str
            Name of the compound to be analyzed. If provided, it is returned in the dict_rgyr dictionary in the "cmpd_name" key.       
        N_equil: int, optional 
            Number of Equilibration Steps to discard from the calculation of the energetic contributions
        return_df: bool
            True to return the dataframe with SASA values for each frame. Default is False

        Returns:
        --------------------
        dict_sasa_multi: dict
        df: dataframe, optional
            Dataframe of SASA values for each frame in the trajectory (values from multiple trajectories are concatenated). Returned as second value only if return_df is True

        """

        dict_sasa_multi = {}
        df_sasa_multi = pd.DataFrame()
        for count, trj1 in enumerate(list_mdtraj_obj): 
            dict_sasa, df_sasa = self.extract_sasa(trj1, N_equil = N_equil, return_df = True) 
            dict_sasa = {k+'_{}'.format(count): v for k, v in dict_sasa.items()}
            dict_sasa_multi.update(dict_sasa)
            df_sasa_multi = df_sasa_multi.append(df_sasa)

        # calculate statistics over all replicas
        stats = list(self.get_stats(df_sasa_multi.values.tolist()))
        dict_sasa = {'sasa_pl_av': stats[0], 'sasa_pl_std': stats[1], 'sasa_pl_med': stats[2]}
        dict_sasa = {k+'_all_replicas': v for k, v in dict_sasa.items()}
        dict_sasa_multi.update(dict_sasa)
 
        if cmpd_name != None:
            dict_sasa_multi.update({'cmpd_name': cmpd_name})

        if return_df:
            df_sasa_multi.reset_index(inplace = True, drop = True)
            df_sasa_multi.columns = ['sasa']
            return dict_sasa_multi, df_sasa_multi
        else:
            return dict_sasa_multi
        

    @classmethod
    def extract_rmsd(self, mdtraj_obj, cmpd_name = None, reference = None, frame = 0, atom_indices = None, N_equil = None, return_df = False, **kwargs):
        """
        Extracts RMSD from each frame of the simulation relative either the provided reference structure or to the initial snapshot in the trajectory (see mdtraj.rmsd for more details). 
        It returns a dictionary containing Mean, Median and Standard Deviation of RMSD.

        Parameters:
        --------------------
        mdtraj_obj: obj
            Trajectory of the solute 
        cmpd_name: str
            Name of the compound to be analyzed. If provided, it is returned in the dict_rmsd dictionary in the "cmpd_name" key       
        reference: obj, optional 
            Object containing the reference conformation to measure distances to. If none is provided, then the function uses the first snapshot of the trajectory as reference.
        frame: int, optional 
            The index of the conformation in reference to measure distances to. Default is zero
        atom_indices: array_like, optional
            The indices of the atoms to use in the RMSD calculation. If not supplied, all atoms will be used.
        N_equil: int, optional 
            Number of Equilibration Steps to discard from the calculation of the energetic contributions
        return_df: bool
            True to return the dataframe with the values of RMSD for each frame. Default is False
        **kwargs: optional
            additional arguments of the rmsd function of mdtraj. See mdtraj documentation
            
        Returns:
        --------------------
        dict_rmsd: dict
        df: dataframe, optional
            Dataframe of RMSD values for each frame in the trajectory. Returned as second value only if return_df is True

        """

        if reference == None: 
            reference = mdtraj_obj

        df = md.rmsd(mdtraj_obj, reference, frame=frame, atom_indices=atom_indices, **kwargs)

        if N_equil != None:
            df = df[N_equil:]
 
        stats = list(self.get_stats(df))

        #dict_rmsd = {'rmsd_pl_av': stats[0], 'rmsd_pl_std': stats[1], 'rmsd_pl_med': stats[2]}
        dict_rmsd = {'rmsd_pl_std': stats[1]}
 
        if cmpd_name != None:
            dict_rmsd.update({"cmpd_name": cmpd_name})

        if return_df:
            return(dict_rmsd, df)
        else:
            return(dict_rmsd)


    @classmethod
    def extract_rmsd_multi(self, list_mdtraj_obj, cmpd_name = None, reference = None, frame = 0, atom_indices = None, N_equil = None, return_df = False, **kwargs):
        """
        Extracts RMSD from every frame of each trajectory relative either the provided reference structure or to the initial snapshot in each of the trajectories (see mdtraj.rmsd for more details). 
        It returns a dictionary containing Mean, Median and Standard Deviation of RMSD both for every trajectory singularly and for the concatenated trajectory.

        Parameters:
        --------------------
        list_mdtraj_obj: list
            List of solute trajectory objects   
        cmpd_name: str
            Name of the compound to be analyzed. If provided, it is returned in the dict_rmsd dictionary in the "cmpd_name" key       
        reference: obj, optional 
            Object containing the reference conformation to measure distances to. If none is provided, then the function uses the first snapshot of the trajectory as reference.
        frame: int, optional 
            The index of the conformation in reference to measure distances to (Default is zero)
        atom_indices: array_like, optional
            The indices of the atoms to use in the RMSD calculation. If not supplied, all atoms will be used.
        N_equil: int, optional 
            Number of Equilibration Steps to discard from the calculation of the energetic contributions
        return_df: bool
            True to return the dataframe with RMSD values for each frame. Default is False
        **kwargs: optional
            additional arguments of the rmsd function of mdtraj. See mdtraj documentation
            
        Returns:
        --------------------
        dict_rmsd_multi: dict
        df: dataframe, optional
            Dataframe of RMSD values for each frame in the trajectory (values from multiple trajectories are concatenated). Returned as second value only if return_df is True

        """

        dict_rmsd_multi = {}
        for count, trj1 in enumerate(list_mdtraj_obj): 
            dict_rmsd, df_rmsd = self.extract_rmsd(trj1, reference = reference, frame = frame, atom_indices = atom_indices, N_equil = N_equil, return_df = True, **kwargs) 
            dict_rmsd = {k+'_{}'.format(count): v for k, v in dict_rmsd.items()}
            dict_rmsd_multi.update(dict_rmsd)
            if count == 0:
                df_rmsd_multi = df_rmsd
            else:
                df_rmsd_multi = np.append(df_rmsd_multi, df_rmsd)

        # calculate statistics over all replicas
        stats = list(self.get_stats(df_rmsd_multi))
        dict_rmsd = {'rmsd_pl_std': stats[1]}
        dict_rmsd = {k+'_all_replicas': v for k, v in dict_rmsd.items()}
        dict_rmsd_multi.update(dict_rmsd)
 
        if cmpd_name != None:
            dict_rmsd_multi.update({'cmpd_name': cmpd_name})

        if return_df:
            df_rmsd_multi2 = pd.DataFrame(df_rmsd_multi, columns=["RMSD"])
            df_rmsd_multi2.reset_index(inplace = True, drop = True)
            return dict_rmsd_multi, df_rmsd_multi2
        else:
            return dict_rmsd_multi 


    @classmethod
    def extract_volume_cavity(self, mdtraj_obj):
        """
        Extracts the Volumne of the cavity of the binding pocket using POVME (Pocket Volume Measurer) for each snapshot in the simulation.
        It returns a dictionary containing Mean, Median and Standard Deviation of Volume of the cavity.

        Parameters:
        --------------------
        mdtraj_obj: obj
            Trajectory of the solute 
        cmpd_name: str
            Name of the compound to be analyzed. If provided, it is returned in the dict_rmsd dictionary in the "cmpd_name" key       
        reference: obj, optional 
            Object containing the reference conformation to measure distances to. If none is provided, then the function uses the first snapshot of the trajectory as reference.
        frame: int, optional 
            The index of the conformation in reference to measure distances to (Default is zero)
        atom_indices: array_like, optional
            The indices of the atoms to use in the RMSD calculation. If not supplied, all atoms will be used.
            
        Returns:
        --------------------
        dict_pv: dict

        """

        return(print("Not Implemented yet"))


