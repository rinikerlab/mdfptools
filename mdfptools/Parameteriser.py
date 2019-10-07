import tempfile

# import contextlib

from simtk import unit #Unit handling for OpenMM
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.openmm.app import PDBFile
# from openforcefield.utils.toolkits import RDKitToolkitWrapper, ToolkitRegistry
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField

import parmed
# from rdkit import Chem
import pickle
import shutil
import os
import numpy as np
# import mdfptools
# from mdfptools.utils import get_data_filename
# from .utils import get_data_filename
# from utils import get_data_filename
from .utils import get_data_filename
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
    def _rdkit_setter(cls, smiles):
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
        from rdkit import Chem
        from rdkit.Chem import AllChem

        cls.smiles = smiles
        mol = Chem.MolFromSmiles(smiles, sanitize = False)
        # mol = Chem.MolFromSmiles(smiles, sanitize = True)
        mol.SetProp("_Name", cls.smiles)
        mol.UpdatePropertyCache(strict = False)
        mol = Chem.AddHs(mol)
        Chem.GetSSSR(mol)
        # params = AllChem.ETKDG()
        # params.enforceChirality = True
        AllChem.EmbedMolecule(mol, enforceChirality = True)

        ### TESTING
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
    def _rdkit_parameteriser(cls, mol):
        from rdkit import Chem
        from openforcefield.utils.toolkits import RDKitToolkitWrapper, ToolkitRegistry

        """
        Creates a parameterised system from rdkit molecule

        Parameters
        ----------
        mol : rdkit.Chem.Mol
        """

        try:
            forcefield = ForceField('test_forcefields/smirnoff99Frosst.offxml')
            molecule = Molecule.from_rdkit(mol)
            if hasattr(cls, "_ddec_charger"):
                molecule.partial_charges = unit.Quantity(np.array(cls._ddec_charger(mol, cls.rf)), unit.elementary_charge)
            else:
                from openforcefield.utils.toolkits import AmberToolsToolkitWrapper
                molecule.compute_partial_charges_am1bcc(toolkit_registry = AmberToolsToolkitWrapper())

            topology = Topology.from_molecules(molecule)
            openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules= [molecule])

            ligand_pmd = parmed.openmm.topsystem.load_topology(topology.to_openmm(), openmm_system, molecule._conformers[0])
        except Exception as e:
            raise ValueError("Parameterisation Failed : {}".format(e)) #TODO

        ligand_pmd.title = cls.smiles

        for i in ligand_pmd.residues:
            i.name = 'LIG'

        tmp_dir = tempfile.mkdtemp()
        # We need all molecules as both pdb files (as packmol input)
        # and mdtraj.Trajectory for restoring bonds later.
        pdb_filename = tempfile.mktemp(suffix=".pdb", dir=tmp_dir)
        Chem.MolToPDBFile(mol, pdb_filename)
        cls.pdb_filename = pdb_filename
        cls.ligand_pmd = ligand_pmd

    @classmethod
    def load_ddec_models(cls, epsilon = 4):
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
    def unload_ddec_models(cls):
        """
        Unload the machine-learned charge model, which takes over 1 GB of memory.
        """
        del cls.rf

    @classmethod
    def _openeye_setter(cls, smiles):
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
        from openeye import oechem # OpenEye Python toolkits
        from openeye import oeomega # Omega toolkit
        # from openeye import oequacpac #Charge toolkit

        cls.smiles = smiles
        cls.omega = oeomega.OEOmega()
        cls.omega.SetMaxConfs(1)
        cls.omega.SetIncludeInput(False)
        cls.omega.SetStrictStereo(False) #Refuse to generate conformers if stereochemistry not provided
        # cls.charge_engine = oequacpac.OEAM1BCCCharges()
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smiles)
        oechem.OEAddExplicitHydrogens(mol)
        oechem.OETriposAtomNames(mol)

        # Generate conformers with Omega; keep only best conformer
        status = cls.omega(mol)
        if not status:
            raise ValueError("Error generating conformers for %s." % (cls.smiles)) #TODO
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
    def _openeye_parameteriser(cls, mol):
        """
        Creates a parameterised system from openeye molecule

        Parameters
        ----------
        mol : oechem.OEMol
        """
        try:
            forcefield = ForceField('test_forcefields/smirnoff99Frosst.offxml')
            molecule = Molecule.from_openeye(mol)
            from openforcefield.utils.toolkits import OpenEyeToolkitWrapper
            molecule.compute_partial_charges_am1bcc(toolkit_registry = OpenEyeToolkitWrapper())

            topology = Topology.from_molecules(molecule)
            openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules= [molecule])

            ligand_pmd = parmed.openmm.topsystem.load_topology(topology.to_openmm(), openmm_system, molecule._conformers[0])
        except:
            raise ValueError("Parameterisation Failed") #TODO

        ligand_pmd.title = cls.smiles

        for i in ligand_pmd.residues:
            i.name = 'LIG'

        tmp_dir = tempfile.mkdtemp()
        # We need all molecules as both pdb files (as packmol input)
        # and mdtraj.Trajectory for restoring bonds later.
        pdb_filename = tempfile.mktemp(suffix=".pdb", dir=tmp_dir)
        from openeye import oechem # OpenEye Python toolkits
        oechem.OEWriteMolecule( oechem.oemolostream( pdb_filename ), mol)
        cls.pdb_filename = pdb_filename
        cls.ligand_pmd = ligand_pmd

    @classmethod
    def save(cls, file_name, file_path = "./"):
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
        pickle.dump(cls.system_pmd , pickle_out)
        pickle_out.close()

        return os.path.abspath(path)

    run = via_openeye

class LiquidParameteriser(BaseParameteriser):
    """
    Parameterisation of liquid box, i.e. multiple replicates of the same molecule
    """

    @classmethod
    def via_openeye(cls, smiles, density, num_lig = 100):
        """
        Parameterisation perfromed via openeye toolkit.

        Parameters
        ----------------
        smiles : str
            SMILES string of the molecule to be parametersied
        density : simtk.unit
            Density of liquid box
        num_lig : int
            Number of replicates of the molecule

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        mol = cls._openeye_setter(smiles)
        # mol = cls._openeye_charger(mol)
        cls._openeye_parameteriser(mol)
        return cls._via_helper(density, num_lig)

    @classmethod
    def via_rdkit(cls, smiles, density, num_lig = 100):
        #TODO !!!!!!!!!!!! approximating volue by density if not possible via rdkit at the moment.
        """
        Parameterisation perfromed via rdkit.

        Parameters
        ----------------
        smiles : str
            SMILES string of the molecule to be parametersied
        density : simtk.unit
            Density of liquid box
        num_lig : int
            Number of replicates of the molecule

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        mol = cls._rdkit_setter(smiles)
        # mol = cls._openeye_charger(mol)
        cls._rdkit_parameteriser(mol)
        return cls._via_helper(density, num_lig)

    @classmethod
    def _via_helper(cls, density, num_lig):
        #TODO !!!!!!!!!!!! approximating volue by density if not possible via rdkit at the moment.
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
        import mdtraj as md
        from openmoltools import packmol
        density = density.value_in_unit(unit.gram / unit.milliliter)

        ligand_mdtraj = md.load(cls.pdb_filename)[0]
        #box_size = packmol.approximate_volume_by_density([smiles], [num_lig], density=density, 		box_scaleup_factor=1.1, box_buffer=2.0)
        box_size = packmol.approximate_volume_by_density([smiles], [num_lig], density=density, 		box_scaleup_factor=1.5, box_buffer=2.0)
        packmol_out = packmol.pack_box([ligand_mdtraj], [num_lig], box_size = box_size)


        cls.system_pmd = cls.ligand_pmd * num_lig
        cls.system_pmd.positions = packmol_out.openmm_positions(0)
        cls.system_pmd.box_vectors = packmol_out.openmm_boxes(0)
        try:
            shutil.rmtree(cls.pdb_filename)
            del cls.ligand_pmd, cls.pdb_filename
        except Exception as e:
            print("Error due to : {}".format(e))

        cls.system_pmd.title = cls.smiles
        return cls.system_pmd

    run = via_openeye

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

    try:
        solvent_pmd = parmed.load_file(get_data_filename("tip3p.prmtop")) #FIXME #TODO
    except ValueError:
        print("Water file cannot be located")

    # default_padding = 1.25 #nm

    @classmethod
    def via_openeye(cls, smiles, default_padding = 1.25*unit.nanometer, **kwargs):
        """
        Parameterisation perfromed via openeye.

        Parameters
        --------------------
        smiles : str
            SMILES string of the solute molecule
        default_padding : simtk.unit
            Dictates amount of water surronding the solute. Default is 1.25 nanometers

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        #TODO currently only supports one solute molecule
        mol = cls._openeye_setter(smiles)
        # mol = cls._openeye_charger(mol)
        cls._openeye_parameteriser(mol)
        cls.default_padding = default_padding.value_in_unit(unit.nanometer)

        return cls._via_helper(**kwargs)

    @classmethod
    def via_rdkit(cls, smiles, default_padding = 1.25*unit.nanometer, **kwargs):
        """
        Parameterisation perfromed via openeye.

        Parameters
        --------------------
        smiles : str
            SMILES string of the solute molecule
        default_padding : simtk.unit
            Dictates amount of water surronding the solute. Default is 1.25 nanometers

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        mol = cls._rdkit_setter(smiles)
        # mol = cls._rdkit_charger(mol)
        cls._rdkit_parameteriser(mol)
        cls.default_padding = default_padding.value_in_unit(unit.nanometer)
        return cls._via_helper(**kwargs)

    @classmethod
    def _via_helper(cls, **kwargs):
        """
        Helper function for via_rdkit or via_openeye

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        from pdbfixer import PDBFixer # for solvating

        fixer = PDBFixer(cls.pdb_filename)
        if "padding" not in kwargs:
            fixer.addSolvent(padding = cls.default_padding)
        else:
            fixer.addSolvent(padding = float(kwargs["padding"]))

        tmp_dir = tempfile.mkdtemp()
        cls.pdb_filename = tempfile.mktemp(suffix=".pdb", dir=tmp_dir)
        with open(cls.pdb_filename, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        complex = parmed.load_file(cls.pdb_filename)

        solvent = complex["(:HOH)"]
        num_solvent = len(solvent.residues)

        solvent_pmd = cls.solvent_pmd *  num_solvent
        solvent_pmd.positions = solvent.positions

        cls.system_pmd = cls.ligand_pmd + solvent_pmd
        cls.system_pmd.box_vectors = complex.box_vectors

        try:
            shutil.rmtree(cls.pdb_filename)
            del cls.ligand_pmd, cls.pdb_filename
        except:
            pass

        cls.system_pmd.title = cls.smiles
        return cls.system_pmd


    run = via_openeye

class VaccumParameteriser(BaseParameteriser):
    @classmethod
    def via_openeye(cls, smiles):
        """
        Parameterisation perfromed via openeye toolkit.

        Parameters
        ----------------
        smiles : str
            SMILES string of the molecule to be parametersied

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        mol = cls._openeye_setter(smiles)
        # mol = cls._openeye_charger(mol)
        cls._openeye_parameteriser(mol)

        cls.system_pmd = cls.ligand_pmd

        cls.system_pmd.title = cls.smiles
        return cls.system_pmd


    @classmethod
    def via_rdkit(cls, smiles):
        """
        Parameterisation perfromed via rdkit toolkit.

        Parameters
        ----------------
        smiles : str
            SMILES string of the molecule to be parametersied

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        mol = cls._rdkit_setter(smiles)
        # mol = cls._rdkit_charger(mol)
        cls._rdkit_parameteriser(mol)

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
