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

class BaseExtractor():
    """

    .. warning :: The base class should not be used directly
    """
    string_identifier = "!Should_Call_From_Inherited_Class"

    @classmethod
    def _solute_solvent_split(cls, topology):
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

            topology = parm.topology
            solute_atoms, solvent_atoms = cls._solute_solvent_split(topology)
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


            new_force.createExclusionsFromBonds([(i[0].index, i[1].index) for i in topology.bonds()], 3)
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


            new_force.createExclusionsFromBonds([(i[0].index, i[1].index) for i in topology.bonds()], 3)
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


            new_force.createExclusionsFromBonds([(i[0].index, i[1].index) for i in topology.bonds()], 3)
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
            for i in range(len(mdtraj_obj)):
                context.setPositions(mdtraj_obj.openmm_positions(i))

                df["{}_intra_crf".format(cls.string_identifier)].append(context.getState(getEnergy=True, groups=set(cls.group_name2num["intra_crf"])).getPotentialEnergy()._value)
                df["{}_intra_lj".format(cls.string_identifier)].append(context.getState(getEnergy=True, groups=set(cls.group_name2num["intra_lj"])).getPotentialEnergy()._value)
                df["{}_total_crf".format(cls.string_identifier)].append(context.getState(getEnergy=True, groups=set(cls.group_name2num["intra_crf"] + cls.group_name2num["inter_crf"])).getPotentialEnergy()._value)
                df["{}_total_lj".format(cls.string_identifier)].append(context.getState(getEnergy=True, groups=set(cls.group_name2num["intra_lj"] + cls.group_name2num["inter_lj"])).getPotentialEnergy()._value)


            df["{}_intra_ene".format(cls.string_identifier)] = [sum(x) for x in zip(df["{}_intra_crf".format(cls.string_identifier)], df["{}_intra_lj".format(cls.string_identifier)])]

            df["{}_total_ene".format(cls.string_identifier)] = [sum(x) for x in zip(df["{}_total_crf".format(cls.string_identifier)], df["{}_total_lj".format(cls.string_identifier)])]

            del context, integrator
            return df

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
        df["{}_rgyr".format(cls.string_identifier)] = list(md.compute_rg(mdtraj_obj, masses = np.array([a.element.mass for a in mdtraj_obj.topology.atoms])))
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
        #df["{}_sasa".format(cls.string_identifier)] = list(md.compute_rg(mdtraj_obj.atom_slice(mdtraj_obj.topology.select("resid 0")))) # this was what I had before which should be incorrect
        df["{}_sasa".format(cls.string_identifier)] = list(md.shrake_rupley(mdtraj_obj.atom_slice(mdtraj_obj.topology.select("resid 0")), mode = "residue"))
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
    def _solute_solvent_split(cls, topology):
        """
        Distinguish solutes from solvents, used in :func:`~BaseExtractor._extract_energies_helper`

        The following is assumed:
            - there are only two type of residues
            - the residue that is lesser in number is the solute


        Parameters
        -----------
        topology : parmed.topology

        Returns
        ------------
        solute_atoms : set
            set of solute_atoms indices
        solvent_atoms : set
            set of solvent_atoms indices
        """
        reslist = [res.name for res in topology.residues()]
        resname = set(reslist)
        resname1 = resname.pop()
        try:
            resname2 = resname.pop()
        except:
            resname2 = None
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
        #topology = parm.topology
        for atom in topology.atoms():
            if atom.residue.name == solvent_name :
                solvent_atoms.add(atom.index)
            else:
                solute_atoms.add(atom.index)
        return solute_atoms, solvent_atoms


    @classmethod
    def _get_all_exception_atom_pairs(cls, system, topology):
        """
        Using the parametersied system to obtain the exception and exclusion pairs, used in :func:`~BaseExtractor._extract_energies_helper`. This is inferred purely from the parameterised system and connectivity.


        Parameters
        -----------
        system : OpenMM.System
        topology : parmed.topology

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
        bonded_pairs =  {(b[0].index, b[1].index) if b[0].index < b[1].index else (b[1].index, b[0].index) for b in topology.bonds()}
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
        #TODO
        charges = [i.charge for i in parmed_obj.atoms]
        df["{}_dipole_magnitude".format(cls.string_identifier)] = [np.linalg.norm(i) for i in md.dipole_moments(mdtraj_obj, charges)]
        return df

    @classmethod
    def _solute_solvent_split(cls, topology):
        """
        Distinguish solutes from solvents, used in :func:`~BaseExtractor._extract_energies_helper`

        The following is assumed:
            - the first residue is the 'solute' , else 'solvent'


        Parameters
        -----------
        topology : parmed.topology

        Returns
        ------------
        solute_atoms : set
            set of solute_atoms indices
        solvent_atoms : set
            set of solvent_atoms indices
        """
        solute_atoms = set()
        solvent_atoms = set()
        for atom in topology.atoms():
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
        topology : parmed.topology

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
        bonded_pairs =  {(b[0].index, b[1].index) if b[0].index < b[1].index else (b[1].index, b[0].index) for b in topology.bonds()}
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
