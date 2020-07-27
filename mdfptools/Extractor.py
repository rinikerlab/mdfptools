import pickle
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import itertools
import io
import parmed
from math import sqrt
import mdtraj as md
import numpy as np
import tempfile

"""
TODOs:
    - update _solute_solvent_split for all the other instances
    - energy file parsing
    - tqdm with extraction? nah too many levels of progress tracking
    - 3d psa calculation function polishing
    - rgyr, dipole, sasa make sure only include the solute
    - custom property extractor
"""


class BaseExtractor():
    """

    .. warning :: The base class should not be used directly
    """
    string_identifier = "!Should_Call_From_Inherited_Class"

    @classmethod
    def _solute_solvent_split(cls, topology, **kwargs):
        """
        Abstract method
        """
        raise NotImplementedError

    @classmethod
    def _get_all_exception_atom_pairs(cls, system, topology):
        """
        Abstract method
        """
        raise NotImplementedError

    @classmethod
    def _extract_energies_helper(cls, mdtraj_obj, parmed_obj, platform = "CPU", **kwargs):
            """
            Helper function for extracting the various energetic components described in the original publication from each frame of simulation. OpenMM.CustomNonbondedForces are used. Specifically, the reaction field defintion is taken from GROMOS96 manual.

            Parameters
            -------------
            mdtraj_obj : mdtraj.trajectory
                The simulated trajectory
            parmed_obj : parmed.structure
                Parmed object of the fully parameterised simulated system.
            platform : str
                The computing architecture to do the calculation, default to CPU, CUDA, OpenCL is also possible.
            Returns
            --------------
            context : Openmm.Context
            integrator : Openmm.Integrator
            """
            parm, traj = parmed_obj, mdtraj_obj

            system = parm.createSystem(nonbondedMethod=CutoffPeriodic, nonbondedCutoff=1.0*nanometer, constraints=AllBonds)
            for i in system.getForces():
                i.setForceGroup(0)

            # topology = parm.topology #XXX here I made the change
            topology = mdtraj_obj.topology
            solute_atoms, solvent_atoms = cls._solute_solvent_split(topology, **kwargs)
            solute_1_4_pairs, solvent_1_4_pairs, solute_excluded_pairs, solvent_excluded_pairs, solute_self_pairs, solvent_self_pairs = cls._get_all_exception_atom_pairs(system, topology)



            forces = { force.__class__.__name__ : force for force in system.getForces() }
            nonbonded_force = forces['NonbondedForce']
            r_cutoff = nonbonded_force.getCutoffDistance()
            try:
                nonbonded_force.setReactionFieldDielectric(kwargs["solvent_dielectric"])
            except KeyError or TypeError:
                pass
            epsilon_solv = nonbonded_force.getReactionFieldDielectric()

            ONE_4PI_EPS0 = 138.935456
            c_rf = r_cutoff**(-1) * ((3*epsilon_solv) / (2*epsilon_solv + 1))
            k_rf = r_cutoff**(-3) * ((epsilon_solv - 1) / (2*epsilon_solv + 1))


            cls.group_number = 1
            cls.group_name2num = {}

            def update_grouping(name, force):
               # global group_number
               # global group_name2num
                if name in cls.group_name2num:
                    cls.group_name2num[name].append(cls.group_number)
                else:
                    cls.group_name2num[name] = [cls.group_number]
                force.setForceGroup(cls.group_number)
                cls.group_number += 1
            #     print(cls.group_name2num)
            ##################
            ################# Expressions
            ################
            V_lj = """4*epsilon*(((sigma/r)^6)^2 - (sigma/r)^6);
                epsilon = sqrt(epsilon1 * epsilon2);
                sigma = 0.5*(sigma1+sigma2)
            """

            V_crf = """(q1*q2*ONE_4PI_EPS0)*(r^(-1) + (k_rf)*(r^2) - (c_rf));
                ONE_4PI_EPS0 = %.16e;
                c_rf = %f;
                k_rf = %f
            """ % (ONE_4PI_EPS0, c_rf.value_in_unit_system(md_unit_system), k_rf.value_in_unit_system(md_unit_system))

            V_14_lj = "2^(-1)* 4*epsilon*(((sigma/r)^6)^2 - (sigma/r)^6)" #calling them xepsilon and xsigma just to see if this interferes with the `addPerParticleParameter` below

            V_14_crf = """(1.2)^(-1)*(q_prod*ONE_4PI_EPS0)*(r^(-1) + (k_rf)*(r^2) - (c_rf));
                ONE_4PI_EPS0 = %.16e;
                c_rf = %f;
                k_rf = %f
            """ % (ONE_4PI_EPS0, c_rf.value_in_unit_system(md_unit_system), k_rf.value_in_unit_system(md_unit_system))

            V_excluded_crf = """(q_prod*ONE_4PI_EPS0)*( (k_rf)*(r^2) - (c_rf));
                ONE_4PI_EPS0 = %.16e;
                c_rf = %f;
                k_rf = %f
            """ % (ONE_4PI_EPS0, c_rf.value_in_unit_system(md_unit_system), k_rf.value_in_unit_system(md_unit_system))

            V_self_crf = """0.5 * (q_prod*ONE_4PI_EPS0)*(  - (c_rf));
                ONE_4PI_EPS0 = %.16e;
                c_rf = %f;
                k_rf = %f
            """ % (ONE_4PI_EPS0, c_rf.value_in_unit_system(md_unit_system), k_rf.value_in_unit_system(md_unit_system))
            #######################
            ####################### Normal LJ, solute-solute
            #######################

            new_force = CustomNonbondedForce(V_lj)
            new_force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
            new_force.setCutoffDistance(r_cutoff)
            new_force.addPerParticleParameter('sigma')
            new_force.addPerParticleParameter('epsilon')

            for particle in range(nonbonded_force.getNumParticles()):
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle)
                new_force.addParticle([sigma, epsilon])


            new_force.createExclusionsFromBonds([(i[0].index, i[1].index) for i in parm.topology.bonds()], 3)
            new_force.addInteractionGroup(solute_atoms, solute_atoms)

            update_grouping("intra_lj", new_force)
            system.addForce(new_force)

            #######################
            ####################### Normal LJ, solute-solvent
            #######################


            new_force = CustomNonbondedForce(V_lj)
            new_force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
            new_force.setCutoffDistance(r_cutoff)
            new_force.addPerParticleParameter('sigma')
            new_force.addPerParticleParameter('epsilon')

            for particle in range(nonbonded_force.getNumParticles()):
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle)
                new_force.addParticle([sigma, epsilon])


            new_force.createExclusionsFromBonds([(i[0].index, i[1].index) for i in parm.topology.bonds()], 3)
            new_force.addInteractionGroup(solute_atoms, solvent_atoms)

            update_grouping("inter_lj", new_force)
            system.addForce(new_force)

            #######################
            ####################### Normal LJ, solvent-solvent
            #######################


            new_force = CustomNonbondedForce(V_lj)
            new_force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
            new_force.setCutoffDistance(r_cutoff)
            new_force.addPerParticleParameter('sigma')
            new_force.addPerParticleParameter('epsilon')

            for particle in range(nonbonded_force.getNumParticles()):
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle)
                new_force.addParticle([sigma, epsilon])


            new_force.createExclusionsFromBonds([(i[0].index, i[1].index) for i in parm.topology.bonds()], 3)
            new_force.addInteractionGroup(solvent_atoms, solvent_atoms)

            update_grouping("solvent_lj", new_force)
            system.addForce(new_force)

            #######################
            ####################### Normal CRF, solute-solute
            #######################

            new_force = CustomNonbondedForce(V_crf)

            new_force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)

            r_cutoff = nonbonded_force.getCutoffDistance()
            new_force.setCutoffDistance(r_cutoff)
            new_force.addPerParticleParameter('q')

            for particle in range(nonbonded_force.getNumParticles()):
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle)
                new_force.addParticle([charge])

            new_force.createExclusionsFromBonds([(i[0].index, i[1].index) for i in parm.topology.bonds()], 3)
            new_force.addInteractionGroup(solute_atoms, solute_atoms)


            update_grouping("intra_crf", new_force)
            system.addForce(new_force)


            #######################
            ####################### Normal CRF, solute-solvent
            #######################

            new_force = CustomNonbondedForce(V_crf)

            new_force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)

            r_cutoff = nonbonded_force.getCutoffDistance()
            new_force.setCutoffDistance(r_cutoff)
            new_force.addPerParticleParameter('q')

            for particle in range(nonbonded_force.getNumParticles()):
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle)
                new_force.addParticle([charge])

            new_force.createExclusionsFromBonds([(i[0].index, i[1].index) for i in parm.topology.bonds()], 3)
            new_force.addInteractionGroup(solute_atoms, solvent_atoms)


            update_grouping("inter_crf", new_force)
            system.addForce(new_force)

            #######################
            ####################### Normal CRF, solvent-solvent
            #######################

            new_force = CustomNonbondedForce(V_crf)

            new_force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)

            r_cutoff = nonbonded_force.getCutoffDistance()
            new_force.setCutoffDistance(r_cutoff)
            new_force.addPerParticleParameter('q')

            for particle in range(nonbonded_force.getNumParticles()):
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle)
                new_force.addParticle([charge])

            new_force.createExclusionsFromBonds([(i[0].index, i[1].index) for i in parm.topology.bonds()], 3)
            new_force.addInteractionGroup(solvent_atoms, solvent_atoms)


            update_grouping("solvent_crf", new_force)
            system.addForce(new_force)

            #######################
            ####################### 1-4 scaled lj, solute
            #######################

            new_force = CustomBondForce(V_14_lj)
            new_force.addPerBondParameter('sigma')
            new_force.addPerBondParameter('epsilon')

            for atom_pair in solute_1_4_pairs:
                charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(atom_pair[0])
                charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(atom_pair[1])

                sigma = 0.5*(sigma1+sigma2)
                epsilon = (epsilon1 * epsilon2).sqrt()

                new_force.addBond(atom_pair[0], atom_pair[1], [sigma.value_in_unit_system(md_unit_system), epsilon.value_in_unit_system(md_unit_system)])

            update_grouping("intra_lj", new_force)
            system.addForce(new_force)

            #######################
            ####################### 1-4 scaled lj, solvent
            #######################

            new_force = CustomBondForce(V_14_lj)
            new_force.addPerBondParameter('sigma')
            new_force.addPerBondParameter('epsilon')

            for atom_pair in solvent_1_4_pairs:
                charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(atom_pair[0])
                charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(atom_pair[1])

                sigma = 0.5*(sigma1+sigma2)
                epsilon = (epsilon1 * epsilon2).sqrt()

                new_force.addBond(atom_pair[0], atom_pair[1], [sigma.value_in_unit_system(md_unit_system), epsilon.value_in_unit_system(md_unit_system)])

            update_grouping("solvent_lj", new_force)
            system.addForce(new_force)

            ##########################
            ########################## 1-4 scaled crf, solute
            ##########################

            new_force = CustomBondForce(V_14_crf)
            new_force.addPerBondParameter('q_prod')


            for atom_pair in solute_1_4_pairs:

                charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(atom_pair[0])
                charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(atom_pair[1])

                charge = charge1 * charge2
                new_force.addBond(atom_pair[0], atom_pair[1], [charge.value_in_unit_system(md_unit_system)])

            update_grouping("intra_crf", new_force)
            system.addForce(new_force)


            ##########################
            ########################## 1-4 scaled crf, solvent
            ##########################

            new_force = CustomBondForce(V_14_crf)
            new_force.addPerBondParameter('q_prod')


            for atom_pair in solvent_1_4_pairs:
                charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(atom_pair[0])
                charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(atom_pair[1])

                charge = charge1 * charge2
                new_force.addBond(atom_pair[0], atom_pair[1], [charge.value_in_unit_system(md_unit_system)])

            update_grouping("solvent_crf", new_force)
            system.addForce(new_force)


            #######################
            ####################### Excluded pairs electrostatics ONLY, solute (1-2, 1-3)
            #######################

            new_force = CustomBondForce(V_excluded_crf)
            new_force.addPerBondParameter('q_prod')

            for atom_pair in solute_excluded_pairs:
                charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(atom_pair[0])
                charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(atom_pair[1])

                charge = charge1 * charge2

                new_force.addBond(atom_pair[0], atom_pair[1], [charge.value_in_unit_system(md_unit_system)])

            update_grouping("intra_crf", new_force)
            system.addForce(new_force)

            #######################
            ####################### Excluded pairs electrostatics ONLY, solvent (1-2, 1-3)
            #######################

            new_force = CustomBondForce(V_excluded_crf)
            new_force.addPerBondParameter('q_prod')

            for atom_pair in solvent_excluded_pairs:
                charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(atom_pair[0])
                charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(atom_pair[1])

                charge = charge1 * charge2

                new_force.addBond(atom_pair[0], atom_pair[1], [charge.value_in_unit_system(md_unit_system)])

            update_grouping("solvent_crf", new_force)
            system.addForce(new_force)

            #######################
            ####################### Self pairs electrostatics ONLY, solute
            #######################

            new_force = CustomBondForce(V_self_crf)
            new_force.addPerBondParameter('q_prod')

            for atom_pair in solute_self_pairs:

                charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(atom_pair[0])
                charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(atom_pair[1])

                charge = charge1 * charge2

                new_force.addBond(atom_pair[0], atom_pair[1], [charge.value_in_unit_system(md_unit_system)])

            update_grouping("intra_crf", new_force)
            system.addForce(new_force)

            #######################
            ####################### Self pairs electrostatics ONLY, solvent
            #######################

            new_force = CustomBondForce(V_self_crf)
            new_force.addPerBondParameter('q_prod')

            for atom_pair in solute_self_pairs:

                charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(atom_pair[0])
                charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(atom_pair[1])

                charge = charge1 * charge2

                new_force.addBond(atom_pair[0], atom_pair[1], [charge.value_in_unit_system(md_unit_system)])

            update_grouping("solvent_crf", new_force)
            system.addForce(new_force)

            #######################
            #######################
            #######################

            system.removeForce([force.getForceGroup() for force in system.getForces() if force.__class__.__name__  == "NonbondedForce" ][0])

            # for i in system.getForces():
            #     print(i,i.getForceGroup())

            integrator = LangevinIntegrator(298.15*kelvin, 1/picosecond, 0.002*picoseconds)
            platform = Platform.getPlatformByName(platform)
            context = Context(system, integrator, platform)


            if parm.box_vectors is not None:
                context.setPeriodicBoxVectors(*parm.box_vectors) #TODO might be better to take the boxes from traj

            return context, integrator

    @classmethod
    def extract_energies(cls, mdtraj_obj, parmed_obj, platform = "CPU", **kwargs):
            """
            Extracting the various energetic components described in the original publication from each frame of simulation.

            Parameters
            -------------
            mdtraj_obj : mdtraj.trajectory
                The simulated trajectory
            parmed_obj : parmed.structure
                Parmed object of the fully parameterised simulated system.
            platform : str
                The computing architecture to do the calculation, default to CPU, CUDA, OpenCL is also possible.

            Returns
            ------------
            df : dict
                Keys are each of the energetic type features. e.g. "intra_lj" are the intra-molecular LJ energies obtained from simulation.

                Values are the corresponding set of numerics, stored as lists.
            """
            df = {}


            context, integrator = cls._extract_energies_helper(mdtraj_obj, parmed_obj, platform = "CPU", **kwargs)


            df["{}_intra_crf".format(cls.string_identifier)] = []
            df["{}_intra_lj".format(cls.string_identifier)] = []
            df["{}_total_crf".format(cls.string_identifier)] = []
            df["{}_total_lj".format(cls.string_identifier)] = []
            
            #TODO can speed up the for loop?????
            for i in range(len(mdtraj_obj)):
                context.setPositions(mdtraj_obj.openmm_positions(i))

                df["{}_intra_crf".format(cls.string_identifier)].append(context.getState(getEnergy=True, groups=set(cls.group_name2num["intra_crf"])).getPotentialEnergy()._value)
                df["{}_intra_lj".format(cls.string_identifier)].append(context.getState(getEnergy=True, groups=set(cls.group_name2num["intra_lj"])).getPotentialEnergy()._value)
                df["{}_total_crf".format(cls.string_identifier)].append(context.getState(getEnergy=True, groups=set(cls.group_name2num["intra_crf"] + cls.group_name2num["inter_crf"])).getPotentialEnergy()._value)
                df["{}_total_lj".format(cls.string_identifier)].append(context.getState(getEnergy=True, groups=set(cls.group_name2num["intra_lj"] + cls.group_name2num["inter_lj"])).getPotentialEnergy()._value)


            #TODO can speed up the for loop?????
            df["{}_intra_ene".format(cls.string_identifier)] = [sum(x) for x in zip(df["{}_intra_crf".format(cls.string_identifier)], df["{}_intra_lj".format(cls.string_identifier)])]

            #TODO can speed up the for loop?????
            df["{}_total_ene".format(cls.string_identifier)] = [sum(x) for x in zip(df["{}_total_crf".format(cls.string_identifier)], df["{}_total_lj".format(cls.string_identifier)])]

            del context, integrator
            return df

    @classmethod
    def _read_xvg(cls, energy_file_xvg, **kwargs):
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
        import pandas as pd
        try:
            data = pd.read_csv(energy_file_xvg, sep="\t", header = None)
            searchfor = ['#', '@']
            N_lines_header = data[0][data[0].str.contains('|'.join(searchfor))].shape[0]
            energy_df = pd.read_csv(energy_file_xvg, header = None, skiprows = N_lines_header, delim_whitespace=True)
            N_col = energy_df.shape[1]
            colnames = ["time"]
            for i in range(N_col-1):
                colnames.append(str.split(str(data[data[0].str.contains('s{} '.format(i))]), " ")[-1].replace('"',''))
            energy_df.columns = colnames
            return energy_df

        except FileNotFoundError:
            raise FileNotFoundError("Energy file {} does not exist".format(energy_file_xvg))



    @classmethod
    def extract_energies_from_xvg(cls, energy_file_xvg, cmpd_name = None, **kwargs):
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

        energy_df = cls._read_xvg(energy_file_xvg)

        energy_array = np.array(energy_df)

        #FIXME can speed up
        idx_coul_sr_lig = [i for i, s in enumerate(energy_df) if 'Coul-SR:LIG' in s][0] #5
        idx_coul_14_lig = [i for i, s in enumerate(energy_df) if 'Coul-14:LIG' in s][0] #7
        idx_lj_sr_lig = [i for i, s in enumerate(energy_df) if 'LJ-SR:LIG' in s][0] #6
        idx_lj_14_lig = [i for i, s in enumerate(energy_df) if 'LJ-14:LIG' in s][0] #8
        idx_coul_sr_wat_lig = [i for i, s in enumerate(energy_df) if 'Coul-SR:Water' in s][0] #1
        idx_coul_14_wat_lig = [i for i, s in enumerate(energy_df) if 'Coul-14:Water' in s][0] #3
        idx_lj_sr_wat_lig = [i for i, s in enumerate(energy_df) if 'LJ-SR:Water' in s][0] #2
        idx_lj_14_wat_lig = [i for i, s in enumerate(energy_df) if 'LJ-14:Water' in s][0] #4

        df = {}

        df["{}_intra_crf".format(cls.string_identifier)] = energy_array[:,idx_coul_sr_lig] + energy_array[:,idx_coul_14_lig]
        
        df["{}_intra_lj".format(cls.string_identifier)] = energy_array[:,idx_lj_sr_lig] + energy_array[:,idx_lj_14_lig]

        df["{}_intra_ene".format(cls.string_identifier)] = np.sum(energy_array[:,[idx_coul_sr_lig, idx_coul_14_lig, idx_lj_sr_lig, idx_lj_14_lig]], axis=1) 

        df["{}_total_crf".format(cls.string_identifier)] = energy_array[:,idx_coul_sr_lig] + energy_array[:,idx_coul_14_lig] + energy_array[:,idx_coul_sr_wat_lig] + energy_array[:,idx_coul_14_wat_lig] 

        df["{}_total_lj".format(cls.string_identifier)] = energy_array[:,idx_lj_sr_lig] + energy_array[:,idx_lj_14_lig] + energy_array[:,idx_lj_sr_wat_lig] + energy_array[:,idx_lj_14_wat_lig]

        df["{}_total_ene".format(cls.string_identifier)] = np.sum(energy_array[:,[idx_coul_sr_lig, idx_coul_14_lig, idx_lj_sr_lig, idx_lj_14_lig, idx_coul_sr_wat_lig, idx_coul_14_wat_lig, idx_lj_sr_wat_lig, idx_lj_14_wat_lig]], axis=1)

        return df

    @classmethod
    def custom_extract(cls, df, **kwargs):
        raise NotImplementedError

    #TODO make sure this only includes the solute
    @classmethod
    def extract_rgyr(cls, mdtraj_obj, **kwargs):
        """
        Extracting radius of gyration from each frame of simulation.
        Assumes the first residue in the system is the solute.

        Parameters
        -------------
        mdtraj_obj : mdtraj.trajectory
            The simulated trajectory

        Returns
        ------------
        df : dict
            Key is prefix_rgyr, where prefix changes depending on the type of Extractor class used. Values are the corresponding set of numerics, stored as lists.
        """
        df = {}

        solute_atoms, _ = cls._solute_solvent_split(mdtraj_obj.topology, **kwargs)
        solute_mdtraj_obj = mdtraj_obj.atom_slice(list(solute_atoms))
        df["{}_rgyr".format(cls.string_identifier)] = list(md.compute_rg(solute_mdtraj_obj, masses = np.array([a.element.mass for a in solute_mdtraj_obj.topology.atoms])))
        return df

    @classmethod
    def extract_sasa(cls, mdtraj_obj, **kwargs):
        """
        Extracting solvent accessible surface area from each frame of simulation.
        Assumes the first residue in the system is the solute.

        Parameters
        -------------
        mdtraj_obj : mdtraj.trajectory
            The simulated trajectory

        Returns
        ------------
        df : dict
            Key is prefix_sasa, where prefix changes depending on the type of Extractor class used. Values are the corresponding set of numerics, stored as lists.
        """
        df = {}
        solute_atoms, _ = cls._solute_solvent_split(mdtraj_obj.topology, **kwargs)
        solute_mdtraj_obj = mdtraj_obj.atom_slice(list(solute_atoms))
        df["{}_sasa".format(cls.string_identifier)] = list(md.shrake_rupley(solute_mdtraj_obj, mode = "residue"))
        return df


    @classmethod
    def extract_psa3d(cls, mdtraj_obj, obj_list=None, include_SandP = None, cmpd_name = None, atom_to_remove = None, **kwargs):#TODO check that it is always consistently called 'psa3d' not 3dpsa or others
        """ ###TODO modify documentation
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
            Set to False to exclude the S and P atoms from the calculation of the 3D-PSA. (Default = True) #TODO now you have S always included right?
        atom_to_remove: str, optional
            Single atom name of the atom to remove from the selection (Default = None). 
            Useful if you want to include only S or only P in the calculation of the 3D-PSA.

        Returns
        ----------
        dict_psa3d: dict
            Keys are mean (3d_psa_av), standard deviation (3d_psa_sd), and median (3d_psa_med) of the 3D-PSA calculated over the simulation time. 
            If cmpd_name is specified, it is returned in the dictionary.
        """
        try:
            from pymol import cmd
        except ImportError:
            raise ImportError('Extract 3D PSA is not possible beacause PyMol python handle not properly installed.')

        solute_atoms, _ = cls._solute_solvent_split(mdtraj_obj.topology, **kwargs)
        solute_mdtraj_obj = mdtraj_obj.atom_slice(list(solute_atoms))

        if "parmed_obj" in kwargs:
            solute_charges = [i.charge for idx,i in enumerate(kwargs["parmed_obj"].atoms) if idx in solute_atoms ]

        tmp_dir = tempfile.mkdtemp()
        traj_filename = tempfile.mktemp(suffix=".pdb", dir = tmp_dir) 
        #TODO is this the most efficient file format? 
        solute_mdtraj_obj.save(traj_filename)
        
        if cmpd_name == None:
            cmpd_name = "cmpd1" #TODO really needed?
        

        # Load trajectory and remove solvent and salts
        obj1 = cmpd_name 
        cmd.reinitialize()
        cmd.load(traj_filename, object=obj1)
        # cmd.load_traj(traj_filename, object=obj1)

        atom_names = []
        cmd.iterate_state(-1, selection=obj1 + " and not elem H", expression="atom_names.append(name)", space=locals())

        # IO
        obj_list = cmd.get_names("objects")
        assert len(obj_list) == 1
        # Select first (and only) object, which contains the trajectory for the solute molecule
        obj = obj_list[0]
        cmd.frame(0)
        states = range(1, cmd.count_states(obj) + 1)  # get all states of the object

        #TODO simplify
        ###Generate pymol selection string. Select the atoms based on the atom types (O,N, polar H and optionally S and P) or provide a customed_selection
        # if atom_to_remove != None and isinstance(atom_to_remove, str):
        if True: #FIXME
            if include_SandP:
                select_string = "(elem N or elem O or elem S or elem P or (elem H and (neighbor elem N+O+S+P))) and {} and not name {}".format(obj, atom_to_remove)  #@carmen add: "or elem S"
            else:
                select_string = "(elem N or elem O or (elem H and (neighbor elem N+O))) and {} and not name {}".format(obj, atom_to_remove)  #@carmen add: "or elem S"
        # else:
        #     if include_SandP:
        #         select_string = "resn {} and (elem N or elem O or elem S or elem P or (elem H and (neighbor elem N+O+S+P))) and ".format(solute_resname) + obj   #@carmen add: "or elem S"
        #     else:
        #         select_string = "resn {} and (elem N or elem O or (elem H and (neighbor elem N+O))) and ".format(solute_resname) + obj  #@carmen add: "or elem S"
        ##Loop over all states
        psa = []
        for state in states:
                cmd.select("noh", select_string) #TODO is this really always called 'noh'?
                ###calc surface area
                psa.append(float(cmd.get_area("noh", state=state)) * 0.01) #unit conversion from A^2 to nm^2

        df = {"{}_psa3d".format(cls.string_identifier) :  psa} 
        return df

###############################################
###############################################

class SolutionExtractor(BaseExtractor):
    """
    Extraction from condensed phase simulation where the system is composed of one solute molecule surronded by solvents

    Parameters
    -----------
    string_identifier : str
        The string identifier tagged as prefix to all values extracted in this class.
    """

    string_identifier = "solution"

    @classmethod
    def _solute_solvent_split(cls, topology, solute_residue_name = None, **kwargs):
        """
        Distinguish solutes from solvents, used in :func:`~BaseExtractor._extract_energies_helper`

        Unless otherwise specified, the following is assumed:
            - there are only two type of residues
            - the residue that is lesser in number is the solute


        Parameters
        -----------
        topology : mdtraj.topology
        solute_residue_name: str or list, optional
            name or list of names of the residue(s) to consider as solute, default is None

        Returns
        ------------
        solute_atoms : set
            set of solute_atoms indices
        solvent_atoms : set
            set of solvent_atoms indices
        """
        if solute_residue_name is None:    
            reslist = [res.name for res in topology.residues]
            resname = set(reslist)
            resname1 = resname.pop()
            try:
                resname2 = resname.pop()
            except:
                resname2 = None #FIXME is this sensible???
            if resname2 is None:
                solvent_name = resname2
            elif reslist.count(resname1) > reslist.count(resname2):
                solvent_name = resname1
            elif reslist.count(resname1) < reslist.count(resname2):
                solvent_name = resname2
            else:
                raise ValueError("num of two species equal, cannot determine")
            solute_atoms = set()
            solvent_atoms = set()
            for atom in topology.atoms:
                if atom.residue.name == solvent_name :
                    solvent_atoms.add(atom.index)
                else:
                    solute_atoms.add(atom.index)
            return solute_atoms, solvent_atoms

        elif isinstance(solute_residue_name, list):
            solute_atoms = []
            for res_name in solute_residue_name:
                solute_atoms = solute_atoms + [atom.index for atom in topology.atoms if atom.residue.name == res_name]
        else:
            solute_atoms = [atom.index for atom in topology.atoms if atom.residue.name == solute_residue_name]
            if len(solute_atoms) == 0:
                print("The topology file does not containg any residue named '{}'. No solute atom extracted.".format(solute_residue_name))
        solvent_atoms = [atom.index for atom in topology.atoms if atom.residue.name != 'LIG'] #FIXME this would include the ions right?
        return solute_atoms, solvent_atoms



    @classmethod
    def _get_all_exception_atom_pairs(cls, system, topology):
        """
        Using the parametersied system to obtain the exception and exclusion pairs, used in :func:`~BaseExtractor._extract_energies_helper`. This is inferred purely from the parameterised system and connectivity.


        Parameters
        -----------
        system : OpenMM.System
        topology : mdtraj.topology

        Returns
        ------------
        solute_1_4_pairs : set
        solvent_1_4_pairs : set
        solute_excluded_pairs : set
        solvent_excluded_pairs : set
        solute_self_pairs : set
        solvent_self_pairs : set
        """

        forces = { force.__class__.__name__ : force for force in system.getForces() }

        angleForce = forces['HarmonicAngleForce']
    #     bondForce = forces['HarmonicBondForce']

        solute_idx, solvent_idx = cls._solute_solvent_split(topology)

        #python3, for python2 use  set((b[0].index, b[1].index) if b[0].index < b[1].index else (b[1].index, b[0].index) for b in topology.bonds())
        bonded_pairs =  {(b[0].index, b[1].index) if b[0].index < b[1].index else (b[1].index, b[0].index) for b in topology.bonds}
        angle_ends = set()
        for i in range(angleForce.getNumAngles()):
            particle1,particle2,particle3, *rest = angleForce.getAngleParameters(i)
            angle_ends.add((particle1, particle3) if particle1 < particle3 else (particle3, particle1))

        solute_self_pairs, solvent_self_pairs = set(), set()
        solute_1_4_pairs, solvent_1_4_pairs = set(), set()
        solute_excluded_pairs, solvent_excluded_pairs = set(), set()

        for i in solute_idx:
            solute_self_pairs.add((i, i))
        for i in solvent_idx:
            solvent_self_pairs.add((i, i))

        for pair in bonded_pairs:
            if pair[0] in solute_idx:
                solute_excluded_pairs.add(pair)
            else:
                solvent_excluded_pairs.add(pair)

        for pair in angle_ends:
            if pair[0] in solute_idx:
                solute_excluded_pairs.add(pair)
            else:
                solvent_excluded_pairs.add(pair)

        try:
            torsionForce = forces['PeriodicTorsionForce']

            for j in  (torsionForce.getTorsionParameters(i) for i in range(torsionForce.getNumTorsions())):
                #if improper torsion, this pair should be bonded
                pair = (j[0], j[2]) if j[0] < j[2] else (j[2], j[0])
                if pair in bonded_pairs:
                    continue

                # real atom pair separated by 3 bonds
                pair = (j[0], j[3]) if j[0] < j[3] else (j[3], j[0])
                if pair in angle_ends or pair in bonded_pairs:
                    continue

                elif pair[0] in solute_idx:
                    solute_1_4_pairs.add(pair)
                else:
                    solvent_1_4_pairs.add(pair)
        except KeyError:
            pass
        return solute_1_4_pairs,solvent_1_4_pairs, solute_excluded_pairs, solvent_excluded_pairs, solute_self_pairs, solvent_self_pairs


    @classmethod
    def extract_dipole(cls, mdtraj_obj, parmed_obj, **kwargs):
        """
        Extracting dipole moment from each frame of simulation.
        Assumes the first residue in the system is the solute.

        Parameters
        -------------
        mdtraj_obj : mdtraj.trajectory
            The simulated trajectory
        parmed_obj : parmed.structure
            Parmed object of the fully parameterised simulated system.

        Returns
        ------------
        df : dict
            Key is prefix_dipole_postfix, where prefix changes depending on the type of Extractor class used, postfix can be {x,y,z,magnitude}. Values are the corresponding set of numerics, stored as lists.
        """
        solute_atoms, _ = cls._solute_solvent_split()
        solute_atoms = list(solute_atoms)

        df = {}
        charges = [i.charge for idx,i in enumerate(parmed_obj.atoms) if idx in solute_atoms ]
        new_traj = mdtraj_obj.atom_slice(solute_atoms)

        output = md.dipole_moments(new_traj, charges)

        df["{}_dipole_x".format(cls.string_identifier)] = output[:,0]
        df["{}_dipole_y".format(cls.string_identifier)] = output[:,1]
        df["{}_dipole_z".format(cls.string_identifier)] = output[:,2]
        df["{}_dipole_magnitude".format(cls.string_identifier)] = [np.linalg.norm(i) for i in output]
        return df
###############################################
###############################################
class TrialSolutionExtractor(SolutionExtractor):
    string_identifier = "TrialSolution"

    @classmethod
    def extract_energies(cls, mdtraj_obj, parmed_obj, platform = "CPU", **kwargs):

        df = {}

        context, integrator = cls._extract_energies_helper( mdtraj_obj, parmed_obj, platform = "CPU", **kwargs)

        num_solvents = float(len(parmed_obj.residues) - 1)
        # print("number of solvents" , num_solvents)

        df["{}_intra_crf".format(cls.string_identifier)] = []
        df["{}_intra_lj".format(cls.string_identifier)] = []
        df["{}_total_crf".format(cls.string_identifier)] = []
        df["{}_total_lj".format(cls.string_identifier)] = []
        for i in range(len(mdtraj_obj)):
            context.setPositions(mdtraj_obj.openmm_positions(i))

            df["{}_intra_crf".format(cls.string_identifier)].append(context.getState(getEnergy=True, groups=set(cls.group_name2num["intra_crf"])).getPotentialEnergy()._value)
            df["{}_intra_lj".format(cls.string_identifier)].append(context.getState(getEnergy=True, groups=set(cls.group_name2num["intra_lj"])).getPotentialEnergy()._value)
            df["{}_total_crf".format(cls.string_identifier)].append(context.getState(getEnergy=True, groups=set(cls.group_name2num["inter_crf"])).getPotentialEnergy()._value / num_solvents)
            df["{}_total_lj".format(cls.string_identifier)].append(context.getState(getEnergy=True, groups=set(cls.group_name2num["inter_lj"])).getPotentialEnergy()._value / num_solvents)


        del context, integrator
        return df


class WaterExtractor(SolutionExtractor):
    """
    Synonyms class as SolutionExtractor

    Parameters
    -----------
    string_identifier : str
        The string identifier tagged as prefix to all values extracted in this class.
    """
    string_identifier = "water"

###############################################
###############################################

class LiquidExtractor(BaseExtractor):
    """Extraction from condensed phase simulation where the system is composed of one kind of molecule only.

    Parameters
    -----------
    string_identifier : str
        The string identifier tagged as prefix to all values extracted in this class.

    """
    string_identifier = "liquid"

    @classmethod
    def _extract_energies_helper(cls, mdtraj_obj, parmed_obj, platform = "CPU", **kwargs):
        try:
            kwargs = {**kwargs , **{"solvent_dielectric" :  kwargs["liquid_solvent_dielectric"]}}
        except:
            pass
        return super(LiquidExtractor, cls)._extract_energies_helper(mdtraj_obj, parmed_obj, platform, **kwargs)

    @classmethod
    def extract_h_bonds(cls, mdtraj_obj, **kwargs):
        "http://mdtraj.org/1.8.0/api/generated/mdtraj.baker_hubbard.html#mdtraj.baker_hubbar    "
        raise NotImplementedError

    @classmethod
    def extract_dipole_magnitude(cls, mdtraj_obj, parmed_obj, **kwargs):
        """
        Extracting dipole moment magnitude from each frame of simulation.
        Assumes the first residue in the system is the solute.

        Parameters
        -------------
        mdtraj_obj : mdtraj.trajectory
            The simulated trajectory
        parmed_obj : parmed.structure
            Parmed object of the fully parameterised simulated system.

        Returns
        ------------
        df : dict
            Key is prefix_dipole_magnitude, where prefix changes depending on the type of Extractor class used. Values are the corresponding set of numerics, stored as lists.
        """
        df = {}
        #TODO is it reasonable to consider all the molecules in the box?
        charges = [i.charge for i in parmed_obj.atoms]
        df["{}_dipole_magnitude".format(cls.string_identifier)] = [np.linalg.norm(i) for i in md.dipole_moments(mdtraj_obj, charges)]
        return df

    @classmethod
    def _solute_solvent_split(cls, topology, **kwargs):
        """
        Distinguish solutes from solvents, used in :func:`~BaseExtractor._extract_energies_helper`

        The following is assumed:
            - the first residue is the 'solute' , else 'solvent'


        Parameters
        -----------
        topology : mdtraj.topology

        Returns
        ------------
        solute_atoms : set
            set of solute_atoms indices
        solvent_atoms : set
            set of solvent_atoms indices
        """
        solute_atoms = set()
        solvent_atoms = set()
        for atom in topology.atoms:
            if atom.residue.index == 0 :
                solute_atoms.add(atom.index)
            else:
                solvent_atoms.add(atom.index)
        return solute_atoms, solvent_atoms

    @classmethod
    def _get_all_exception_atom_pairs(cls, system, topology):
        """
        Using the parametersied system to obtain the exception and exclusion pairs, used in :func:`~BaseExtractor._extract_energies_helper`. This is inferred purely from the parameterised system and connectivity.


        Parameters
        -----------
        system : OpenMM.System
        topology : mdtraj.topology

        Returns
        ------------
        solute_1_4_pairs : set
        solvent_1_4_pairs : set
        solute_excluded_pairs : set
        solvent_excluded_pairs : set
        solute_self_pairs : set
        solvent_self_pairs : set
        """


        forces = { force.__class__.__name__ : force for force in system.getForces() }

        angleForce = forces['HarmonicAngleForce']
    #     bondForce = forces['HarmonicBondForce']

        solute_idx, solvent_idx = cls._solute_solvent_split(topology)

        #python3, for python2 use  set((b[0].index, b[1].index) if b[0].index < b[1].index else (b[1].index, b[0].index) for b in topology.bonds())
        bonded_pairs =  {(b[0].index, b[1].index) if b[0].index < b[1].index else (b[1].index, b[0].index) for b in topology.bonds}
        angle_ends = set()
        for i in range(angleForce.getNumAngles()):
            particle1,particle2,particle3, *rest = angleForce.getAngleParameters(i)
            angle_ends.add((particle1, particle3) if particle1 < particle3 else (particle3, particle1))

        solute_self_pairs, solvent_self_pairs = set(), set()
        solute_1_4_pairs, solvent_1_4_pairs = set(), set()
        solute_excluded_pairs, solvent_excluded_pairs = set(), set()

        for i in solute_idx:
            solute_self_pairs.add((i, i))
        for i in solvent_idx:
            solvent_self_pairs.add((i, i))

        for pair in bonded_pairs:
            if pair[0] in solute_idx:
                solute_excluded_pairs.add(pair)
            else:
                solvent_excluded_pairs.add(pair)

        for pair in angle_ends:
            if pair[0] in solute_idx:
                solute_excluded_pairs.add(pair)
            else:
                solvent_excluded_pairs.add(pair)
        try:
            torsionForce = forces['PeriodicTorsionForce']

            for j in  (torsionForce.getTorsionParameters(i) for i in range(torsionForce.getNumTorsions())):
                #if improper torsion, this pair should be bonded
                pair = (j[0], j[2]) if j[0] < j[2] else (j[2], j[0])
                if pair in bonded_pairs:
                    continue

                # real atom pair separated by 3 bonds
                pair = (j[0], j[3]) if j[0] < j[3] else (j[3], j[0])
                if pair in angle_ends or pair in bonded_pairs:
                    continue

                elif pair[0] in solute_idx:
                    solute_1_4_pairs.add(pair)
                else:
                    solvent_1_4_pairs.add(pair)
        except KeyError:
            pass
        return solute_1_4_pairs,solvent_1_4_pairs, solute_excluded_pairs, solvent_excluded_pairs, solute_self_pairs, solvent_self_pairs



"""
Examples:
-------------------
parm_path = '/home/shuwang/Documents/Modelling/MDFP/Codes/vapour_pressure/crc_handbook/corrupted/RU18.1_8645.pickle'
parm = pickle.load(open(parm_path,"rb"))
traj = md.load('/home/shuwang/Documents/Modelling/MDFP/Codes/vapour_pressure/crc_handbook/corrupted/RU18.1_8645.h5')[:5]
print(LiquidExtractor.extract_dipole_magnitude(traj, parm))
"""
