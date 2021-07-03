import tempfile
from functools import partialmethod

# import contextlib

from simtk import unit  # Unit handling for OpenMM
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.openmm.app import PDBFile

try:
    from openff.toolkit.topology import Molecule, Topology
    from openff.toolkit.typing.engines.smirnoff import ForceField
except ModuleNotFoundError:
    # from openforcefield.utils.toolkits import RDKitToolkitWrapper, ToolkitRegistry
    from openforcefield.topology import Molecule, Topology
    from openforcefield.typing.engines.smirnoff import ForceField

import parmed
from rdkit import Chem
import pickle
import shutil
import os
import numpy as np
# import mdfptools
# from mdfptools.utils import get_data_filename
# from .utils import get_data_filename
# from utils import get_data_filename
from .utils import get_data_filename, approximate_volume_by_density
"""
TODOs:
    - proper handling of tip3p water loading
    - SMILES string is stored in the `title` field of parmed object
"""
##############################################################


class BaseParameteriser():
    """

    .. warning :: The base class should not be used directly
    """
    na_ion_pmd = None
    cl_ion_pmd = None
    system_pmd = None

    @classmethod
    def via_openeye(cls):
        """
        Abstract method
        """
        raise NotImplementedError

    @classmethod
    def via_rdkit(cls):
        """
        Abstract method
        """
        raise NotImplementedError

    @classmethod
    def pmd_generator(cls):
        """
        Abstract method
        """
        raise NotImplementedError

    @classmethod
    def _rdkit_setter(cls, smiles, **kwargs):
        """
        Prepares an rdkit molecule with 3D coordinates.

        Parameters
        ------------
        smiles : str
            SMILES string of the solute moleulce

        Returns
        ---------
        mol : rdkit.Chem.Mol
        """
        from rdkit.Chem import AllChem

        # cls.smiles = smiles
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        # mol = Chem.MolFromSmiles(smiles, sanitize = True)
        mol.SetProp("_Name", smiles)
        mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol)
        Chem.GetSSSR(mol)
        # params = AllChem.ETKDG()
        # params.enforceChirality = True
        AllChem.EmbedMolecule(mol, enforceChirality=True)

        # TESTING
        # ref = Chem.MolFromSmiles('C-C-[C@H](-C)-[C@@H]1-N-C(=O)-[C@H](-C(-C)-C)-N(-C)-C(=O)-[C@H](-C-c2:c:[nH]:c3:c:c:c:c:c:2:3)-N-C(=O)-C-N(-C)-C(=O)-[C@H](-[C@@H](-C)-C-C)-N(-C)-C(=O)-[C@H](-C(-C)-C)-N-C(=O)-C-N(-C)-C(=O)-[C@H](-[C@@H](-C)-C-C)-N(-C)-C(=O)-[C@H](-C(-C)-C)-N(-C)-C(=O)-C-N(-C)-C(=O)-[C@H](-C(-C)-C)-N(-C)-C(=O)-[C@H](-C(-C)-C)-N(-C)-C-1=O')

        # mol = Chem.MolFromPDBFile('/home/shuwang/Documents/Modelling/CP/Datum/conformer_generator/OmphalotinA_MD/0-emin/emin_50.cnf.pdb', removeHs = False, sanitize = False)
        # mol.SetProp("_Name", cls.smiles)
        # mol = AllChem.AssignBondOrdersFromTemplate(ref, mol)
        return mol

    """
    @classmethod
    def _rdkit_charger(cls, mol):
        if not hasattr(cls, "charge_engine"):
            raise ValueError("No Useable charge engine Exist")
        # cls.mol = cls.charge_engine(mol)
        # return cls.mol
        return cls.charge_engine(mol)
    """

    @classmethod
    def _get_forcefield(cls, **kwargs):
        if "ff_path" in kwargs:
            try:
                return ForceField(kwargs['ff_path'], allow_cosmetic_attributes=True)
            except Exception as e:
                print("Specified forcefield cannot be found. Fallback to default forcefield")
        return ForceField('test_forcefields/smirnoff99Frosst.offxml')

    @classmethod
    def _rdkit_parameteriser(cls, mol, **kwargs):
        # from openforcefield.utils.toolkits import RDKitToolkitWrapper, ToolkitRegistry
        """
        Creates a parameterised system from rdkit molecule

        Parameters
        ----------
        mol : rdkit.Chem.Mol
        """
        try:
            molecule = Molecule.from_rdkit(mol, allow_undefined_stereo=cls.allow_undefined_stereo)
            if hasattr(cls, "_ddec_charger"):
                molecule.partial_charges = unit.Quantity(
                    np.array(cls._ddec_charger(mol, cls.rf)), unit.elementary_charge)
            else:
                try:
                    from openff.toolkit.utils import AmberToolsToolkitWrapper

                except ModuleNotFoundError:
                    from openforcefield.utils.toolkits import AmberToolsToolkitWrapper
                molecule.compute_partial_charges_am1bcc(toolkit_registry=AmberToolsToolkitWrapper())

        except Exception as e:
            raise ValueError("Charging Failed : {}".format(e))  # TODO

        return cls._off_handler(molecule, **kwargs)

    @classmethod
    def _off_handler(cls, molecule, **kwargs):
        forcefield = cls._get_forcefield(**kwargs)
        topology = Topology.from_molecules(molecule)
        openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules=[molecule])

        # ligand_pmd.title = cls.smiles

        # for i in ligand_pmd.residues:
        #     i.name = 'LIG' #XXX no longer needed when using omm_top to create parmed structure

        tmp_dir = tempfile.mkdtemp()
        # We need all molecules as both pdb files (as packmol input)
        # and mdtraj.Trajectory for restoring bonds later.
        pdb_filename = tempfile.mktemp(suffix=".pdb", dir=tmp_dir)

        # XXX legacy code for save a pdb copy for simulation box creation
        # Chem.MolToPDBFile(mol, pdb_filename)
        # from openeye import oechem # OpenEye Python toolkits
        # oechem.OEWriteMolecule( oechem.oemolostream( pdb_filename ), mol)

        # XXX swtich to off save pdb
        # ligand_pmd.save(pdb_filename, overwrite=True)
        molecule.to_file(pdb_filename, "pdb")
        omm_top = PDBFile(pdb_filename).topology
        # ligand_pmd = parmed.openmm.topsystem.load_topology(topology.to_openmm(), openmm_system, molecule._conformers[0]) #XXX off topology does not keep atom names and resnames, use omm topology instead
        ligand_pmd = parmed.openmm.topsystem.load_topology(omm_top, openmm_system, molecule._conformers[0])

        return pdb_filename, ligand_pmd

    @classmethod
    def load_ddec_models(cls, epsilon=4, **kwargs):
        """
        Charging molecule using machine learned charge instead of the default AM1-BCC method.

        Requires first installing the mlddec(https://github.com/rinikerlab/mlddec) package. Parameters are availible for elements : {H,C,N,O,Cl,Br,F}.

        Parameters
        ------------
        epsilon : int
            Dielectric constant to be used, polarity of the resulting molecule varies, possible values are {4,78}.
        """

        try:
            import mlddec
        except ImportError:
            raise ImportError('mlddec not properly installed')

        cls.rf = mlddec.load_models(epsilon)
        # cls.charge_engine = cls._ddec_charger
        cls._ddec_charger = mlddec.get_charges

    # @classmethod
    # def _ddec_charger(cls, mol):
    #     try:
    #         import mlddec
    #     except ImportError:
    #         raise ImportError('mlddec not properly installed')
    #
    #     return mlddec.get_charges(mol, cls.rf)

    @classmethod
    def unload_ddec_models(cls, **kwargs):
        """
        Unload the machine-learned charge model, which takes over 1 GB of memory.
        """
        del cls.rf

    @classmethod
    def _openeye_setter(cls, smiles, **kwargs):
        """
        Prepares an openeye molecule with 3D coordinates.

        Parameters
        ------------
        smiles : str
            SMILES string of the solute moleulce

        Returns
        ---------
        mol : oechem.OEMol
        """
        from openeye import oechem  # OpenEye Python toolkits
        from openeye import oeomega  # Omega toolkit
        # from openeye import oequacpac #Charge toolkit

        # cls.smiles = smiles
        cls.omega = oeomega.OEOmega()

        # modified in accordance to recommendation on omega example script: https://docs.eyesopen.com/toolkits/cookbook/python/modeling/am1-bcc.html
        # reduced the default total number of confs from 800 to 100 to save execution time
        eWindow = 15.0
        cls.omega.SetEnergyWindow(eWindow)
        # if "openeye_maxconf" in kwargs and type(kwargs["openeye_maxconf"]) is int and kwargs["openeye_maxconf"] > 0 :
        #     cls.omega.SetMaxConfs(kwargs["openeye_maxconf"])
        # else:
        #     cls.omega.SetMaxConfs(100)
        cls.omega.SetMaxConfs(1)
        cls.omega.SetRMSThreshold(1.0)

        cls.omega.SetIncludeInput(False)
        cls.omega.SetStrictStereo(False)  # Refuse to generate conformers if stereochemistry not provided
        # cls.charge_engine = oequacpac.OEAM1BCCCharges()
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smiles)
        oechem.OEAddExplicitHydrogens(mol)
        oechem.OETriposAtomNames(mol)

        # Generate conformers with Omega; keep only best conformer
        status = cls.omega(mol)
        if not status:
            raise ValueError("Error generating conformers for %s." % (smiles))  # TODO
        return mol

    # @classmethod
    # def _openeye_charger(cls, mol):
    #     # Set to use a simple neutral pH model
    #     #oequacpac.OESetNeutralpHModel(mol) #FIXME input smiles should ensure neutral molecule
    #
    #     # Generate conformers with Omega; keep only best conformer
    #     status = cls.omega(mol)
    #     if not status:
    #         raise ValueError("Error generating conformers for %s." % (cls.smiles)) #TODO
    #
    #     # Assign AM1-BCC charges
    #     oequacpac.OEAssignCharges(mol, cls.charge_engine)
    #     return mol

    @classmethod
    def _openeye_parameteriser(cls, mol, **kwargs):
        """
        Creates a parameterised system from openeye molecule

        Parameters
        ----------
        mol : oechem.OEMol
        """
        try:
            molecule = Molecule.from_openeye(mol, allow_undefined_stereo=cls.allow_undefined_stereo)
            try:
                from openff.toolkit.utils import OpenEyeToolkitWrapper
            except ModuleNotFoundError:
                from openforcefield.utils.toolkits import OpenEyeToolkitWrapper
            molecule.compute_partial_charges_am1bcc(toolkit_registry=OpenEyeToolkitWrapper())

        except Exception as e:
            raise ValueError("Charging Failed : {}".format(e))  # TODO

        return cls._off_handler(molecule, **kwargs)

    @classmethod
    def save(cls, file_name, file_path="./", **kwargs):
        """
        Save to file the parameterised system.

        Parameters
        ------------
        file_name : str
            No file type postfix is necessary
        file_path : str
            Default to current directory

        Returns
        --------
        path : str
            The absolute path where the trajectory is written to.
        """
        path = '{}/{}.pickle'.format(file_path, file_name)
        pickle_out = open(path, "wb")
        pickle.dump(cls.system_pmd, pickle_out)
        pickle_out.close()

        return os.path.abspath(path)

    run = via_openeye


class LiquidParameteriser(BaseParameteriser):
    """
    Parameterisation of liquid box, i.e. multiple replicates of the same molecule
    """

    @classmethod  # TODO boxsize packing scale factor should be customary
    def run(cls, smiles, density, *, allow_undefined_stereo=False, num_lig=100, box_scaleup_factor=1.5, backend="openeye", **kwargs):
        """
        Parameterisation perfromed either with openeye or rdkit toolkit.

        Parameters
        ----------------
        smiles : str
            SMILES string of the molecule to be parametersied
        density : simtk.unit
            Density of liquid box
        allow_undefined_stereo : bool
            Flag passed to OpenForceField `Molecule` object during parameterisation. When set to False an error is returned if SMILES have no/ambiguous steroechemistry. Default to False here as a sanity check for user.
        num_lig : int
            Number of replicates of the molecule
        box_scaleup_factor : float
            Dicatates the packed volume with respect to the volume estimated from density. Default is 1.5.
        backend : str:
            Either `rdkit` or `openeye`

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        if backend not in ["openeye", "rdkit"]: #TODO allow openff_molecule option
            raise ValueError("backend should be either 'openeye' or 'rdkit'")  # XXX allow more options

        cls.box_scaleup_factor = box_scaleup_factor
        cls.allow_undefined_stereo = allow_undefined_stereo
        cls.smiles = smiles
        if backend == "openeye":
            mol = cls._openeye_setter(smiles, **kwargs)
            # mol = cls._openeye_charger(mol)
            cls.pdb_filename, cls.ligand_pmd = cls._openeye_parameteriser(mol, **kwargs)
        else:
            mol = cls._rdkit_setter(smiles, **kwargs)
            # mol = cls._openeye_charger(mol)
            cls.pdb_filename, cls.ligand_pmd = cls._rdkit_parameteriser(mol, **kwargs)
        return cls._via_helper(density, num_lig, **kwargs)


    @classmethod
    def _via_helper(cls, density, num_lig, **kwargs):
        # TODO !!!!!!!!!!!! approximating volume by density if not possible via rdkit at the moment.
        """
        Helper function for via_rdkit or via_openeye

        Parameters
        ----------------
        density : simtk.unit
            Density of liquid box
        num_lig : int
            Number of replicates of the molecule

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        import mdtraj as md  # TODO packmol can accept file name as input too, no need for this really
        from openmoltools import packmol
        density = density.value_in_unit(unit.gram / unit.milliliter)

        ligand_mdtraj = md.load(cls.pdb_filename)[0]
        try:  # TODO better error handling if openeye is not detected
            #box_size = packmol.approximate_volume_by_density([smiles], [num_lig], density=density, 		box_scaleup_factor=1.1, box_buffer=2.0)
            box_size = packmol.approximate_volume_by_density(
                [cls.smiles], [num_lig], density=density, 		box_scaleup_factor=cls.box_scaleup_factor, box_buffer=2.0)
        except:
            box_size = approximate_volume_by_density(
                [cls.smiles], [num_lig], density=density, 		box_scaleup_factor=box_scaleup_factor, box_buffer=2.0)

        packmol_out = packmol.pack_box([ligand_mdtraj], [num_lig], box_size=box_size)

        cls.system_pmd = cls.ligand_pmd * num_lig
        cls.system_pmd.positions = packmol_out.openmm_positions(0)
        cls.system_pmd.box_vectors = packmol_out.openmm_boxes(0)
        try:
            shutil.rmtree("/".join(cls.pdb_filename.split("/")[:-1]))
            del cls.ligand_pmd
        except Exception as e:
            print("Error due to : {}".format(e))

        cls.system_pmd.title = cls.smiles
        return cls.system_pmd

    via_openeye = partialmethod(run, backend="openeye")
    via_rdkit = partialmethod(run, backend="rdkit")


class SolutionParameteriser(BaseParameteriser):
    """
    Parameterisation of solution box, i.e. one copy of solute molecule surronded by water.

    Parameters
    --------------
    solvent_pmd : parmed.structure
        Parameterised tip3p water as parmed object
    """
    # forcefield = ForceField('tip3p.offxml')
    # molecule = Molecule.from_file(get_data_filename("water.sdf"))
    # print(molecule.partial_charges)
    # topology = Topology.from_molecules(molecule)
    # openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules= [molecule])
    # solvent_pmd = parmed.openmm.topsystem.load_topology(topology.to_openmm(), openmm_system, [[1,0,0],[0,1,0],[0,0,1]])

    solvent_pmd = None

    # default_padding = 1.25 #nm

    @classmethod
    def run(cls, smiles, *, solvent_smiles=None, allow_undefined_stereo=False, num_solvent=100, density=None, default_padding=1.25*unit.nanometer, box_scaleup_factor=1.5, backend="openeye", **kwargs):
        """
        Parameterisation perfromed via openeye.

        Parameters
        --------------------
        smiles : str
            SMILES string of the solute molecule
        solvent_smiles : str
            SMILES string of the solvent molecule, default is None, only relevant if the solute is not water.
        allow_undefined_stereo : bool
            Flag passed to OpenForceField `Molecule` object during parameterisation. When set to False an error is returned if SMILES have no/ambiguous steroechemistry. Default to False here as a sanity check for user.
        num_solvent : int
            The number of solvent molecules added into the system, only relevant if the solvent is not water. The default value is 100, but it is left for the user to determine the number of solvent molecule really needed to surrond the solute and create a big enough system.
        density : simtk.unit.quantity.Quantity
            Density of the solvent, default is None, only relevant if the solvent is not water
        default_padding : simtk.unit
            Dictates amount of water surronding the solute. Default is 1.25 nanometers, only relevant if water is the solvent.
        box_scaleup_factor : float
            Dicatates the packed volume with respect to the volume estimated from density. Default is 1.5, only relevant if the solvent is not water
        backend : str:
            Either `rdkit` or `openeye`

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        # TODO currently only supports one solute molecule
        # sanity checks
        cls.smiles = smiles
        cls.allow_undefined_stereo = allow_undefined_stereo
        cls.default_padding = default_padding.value_in_unit(unit.nanometer)
        cls.solvent_smiles = solvent_smiles
        cls.box_scaleup_factor = box_scaleup_factor
        cls.backend = backend
        if solvent_smiles is not None and density is None:
            raise ValueError("Density missing for the solvent {}".format(solvent_smiles))
        if density is not None:
            if type(density) is not unit.quantity.Quantity:
                raise ValueError("density needs to have unit")
            if solvent_smiles is None:
                raise ValueError("Solvent SMILES missing.")
        if backend not in ["openeye", "rdkit", "off_molecule"]:
            raise ValueError("backend should be either 'openeye' or 'rdkit'")  # XXX allow more options

        if backend == "openeye":
            mol = cls._openeye_setter(smiles, **kwargs)
            # mol = cls._openeye_charger(mol)
            cls.pdb_filename, cls.ligand_pmd = cls._openeye_parameteriser(mol, **kwargs)
            if solvent_smiles:
                mol = cls._openeye_setter(solvent_smiles, **kwargs)
                cls.solvent_pdb_filename, cls.solvent_pmd = cls._openeye_parameteriser(mol, **kwargs)

        elif backend == "rdkit":
            mol = cls._rdkit_setter(smiles, **kwargs)
            # mol = cls._rdkit_charger(mol)
            cls.pdb_filename, cls.ligand_pmd = cls._rdkit_parameteriser(mol, **kwargs)
            if solvent_smiles:
                mol = cls._rdkit_setter(solvent_smiles, **kwargs)
                cls.solvent_pdb_filename, cls.solvent_pmd = cls._rdkit_parameteriser(mol, **kwargs)
        elif backend in ["off_molecule"]:
            if "solute_molecule" in kwargs and type(kwargs["solute_molecule"]) is Molecule:
                cls.pdb_filename, cls.ligand_pmd = cls._off_handler(kwargs["solute_molecule"], **kwargs)
            if "solvent_molecule" in kwargs and type(kwargs["solvent_molecule"]) is Molecule:
                cls.solvent_pdb_filename, cls.solvent_pmd = cls._off_handler(kwargs["solvent_molecule"], **kwargs)

        if cls.solvent_pmd is None:
            try:
                cls.solvent_pmd = parmed.load_file(get_data_filename("tip3p.prmtop"))
            except ValueError:
                raise ValueError("Water file cannot be located")
        if solvent_smiles is None:
            cls._via_helper_water(**kwargs)
        else:
            cls._via_helper_other_solvent(density, num_solvent, **kwargs)

        return cls._add_counter_charges(**kwargs)

    @classmethod
    def _via_helper_other_solvent(cls, density, num_solvent, **kwargs):
        from openmoltools import packmol
        density = density.value_in_unit(unit.gram / unit.milliliter)

        if cls.backend == "openeye":
            box_size = packmol.approximate_volume_by_density([cls.smiles, cls.solvent_smiles], [
                                                             1, num_solvent], density=density, 		box_scaleup_factor=cls.box_scaleup_factor, box_buffer=cls.default_padding)
        else:
            box_size = approximate_volume_by_density([cls.smiles, cls.solvent_smiles], [
                                                     1, num_solvent], density=density, 		box_scaleup_factor=cls.box_scaleup_factor, box_buffer=cls.default_padding)

        packmol_out = packmol.pack_box([cls.pdb_filename, cls.solvent_pdb_filename],
                                       [1, num_solvent], box_size=box_size)
        import mdtraj as md

        cls.system_pmd = cls.ligand_pmd + (cls.solvent_pmd * num_solvent)
        cls.system_pmd.positions = packmol_out.openmm_positions(0)
        cls.system_pmd.box_vectors = packmol_out.openmm_boxes(0)
        try:
            # TODO should maybe delete the higher parent level? i.e. -2?
            shutil.rmtree("/".join(cls.pdb_filename.split("/")[:-1]))
            shutil.rmtree("/".join(cls.solvent_pdb_filename.split("/")[:-1]))
            del cls.ligand_pmd, cls.solvent_pmd
        except Exception as e:
            print("Error due to : {}".format(e))

        cls.system_pmd.title = cls.smiles
        return cls.system_pmd

    @classmethod
    def _via_helper_water(cls, **kwargs):
        """
        Helper function for via_rdkit or via_openeye

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        from pdbfixer import PDBFixer  # for solvating

        fixer = PDBFixer(cls.pdb_filename)
        if "padding" not in kwargs:
            fixer.addSolvent(padding=cls.default_padding)
        else:
            fixer.addSolvent(padding=float(kwargs["padding"]))

        tmp_dir = tempfile.mkdtemp()
        cls.pdb_filename = tempfile.mktemp(suffix=".pdb", dir=tmp_dir)
        with open(cls.pdb_filename, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        complex = parmed.load_file(cls.pdb_filename)

        solvent = complex["(:HOH)"]
        num_solvent = len(solvent.residues)

        solvent_pmd = cls.solvent_pmd * num_solvent
        solvent_pmd.positions = solvent.positions

        cls.system_pmd = cls.ligand_pmd + solvent_pmd
        cls.system_pmd.box_vectors = complex.box_vectors

        try:
            shutil.rmtree("/".join(cls.pdb_filename.split("/")[:-1]))
            del cls.ligand_pmd
        except:
            pass

        cls.system_pmd.title = cls.smiles
        return cls.system_pmd

    @classmethod
    def _add_counter_charges(cls, **kwargs):
        """in case the solute molecule has a net charge, 
        counter charge are added to the system in the form of ions,
        Na+ or Cl-, in order to keep charge neutrality.
        """

        solute_charge = int(Chem.GetFormalCharge(Chem.MolFromSmiles(cls.smiles)))
        if solute_charge == 0:
            return cls.system_pmd

        if solute_charge > 0:  # add -ve charge
            if cls.cl_ion_pmd is None:
                cls.cl_ion_pmd = parmed.load_file(get_data_filename("cl.prmtop"))
            ion_pmd = cls.cl_ion_pmd * solute_charge

        elif solute_charge < 0:  # add +ve charge
            if cls.na_ion_pmd is None:
                cls.na_ion_pmd = parmed.load_file(get_data_filename("na.prmtop"))
            ion_pmd = cls.na_ion_pmd * abs(solute_charge)

        # replace the last few solvent molecules and replace them by the ions
        ion_pmd.coordinates = np.array([np.mean(cls.system_pmd[":{}".format(
            len(cls.system_pmd.residues) - i)].coordinates, axis=0) for i in range(abs(solute_charge))])
        cls.system_pmd = cls.system_pmd[":1-{}".format(len(cls.system_pmd.residues) - abs(solute_charge))]
        cls.system_pmd += ion_pmd

        return cls.system_pmd

    via_openeye = partialmethod(run, backend="openeye")
    via_rdkit = partialmethod(run, backend="rdkit")


class VaccumParameteriser(BaseParameteriser):
    @classmethod
    def via_openeye(cls, smiles, allow_undefined_stereo=False, **kwargs):
        """
        Parameterisation perfromed via openeye toolkit.

        Parameters
        ----------------
        smiles : str
            SMILES string of the molecule to be parametersied
        allow_undefined_stereo : bool
            Flag passed to OpenForceField `Molecule` object during parameterisation. When set to False an error is returned if SMILES have no/ambiguous steroechemistry. Default to False here as a sanity check for user.


        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        cls.smiles = smiles
        cls.allow_undefined_stereo = allow_undefined_stereo
        mol = cls._openeye_setter(smiles, **kwargs)
        # mol = cls._openeye_charger(mol)
        cls.pdb_filename, cls.ligand_pmd = cls._openeye_parameteriser(mol, **kwargs)

        cls.system_pmd = cls.ligand_pmd

        cls.system_pmd.title = cls.smiles
        return cls.system_pmd

    @classmethod
    def via_rdkit(cls, smiles, allow_undefined_stereo=False, **kwargs):
        """
        Parameterisation perfromed via rdkit toolkit.

        Parameters
        ----------------
        smiles : str
            SMILES string of the molecule to be parametersied
        allow_undefined_stereo : bool
            Flag passed to OpenForceField `Molecule` object during parameterisation. When set to False an error is returned if SMILES have no/ambiguous steroechemistry. Default to False here as a sanity check for user.

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        cls.smiles = smiles
        cls.allow_undefined_stereo = allow_undefined_stereo
        mol = cls._rdkit_setter(smiles, **kwargs)
        # mol = cls._rdkit_charger(mol)
        cls.pdb_filename, cls.ligand_pmd = cls._rdkit_parameteriser(mol, **kwargs)

        cls.system_pmd = cls.ligand_pmd

        cls.system_pmd.title = cls.smiles
        return cls.system_pmd

    run = via_openeye


"""
from mdfptools.Parameteriser import *
print(SolutionParameteriser().run("CC"))
SolutionParameteriser.load_ddec_models()
print(SolutionParameteriser().via_rdkit("CC"))
SolutionParameteriser.unload_ddec_models()
print(LiquidParameteriser().run("CC", density = 12 * unit.gram / unit.liter))
"""
