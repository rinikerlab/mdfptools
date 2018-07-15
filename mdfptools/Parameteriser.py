import tempfile
import contextlib
from openforcefield import utils
from openmoltools import packmol

from simtk import unit #Unit handling for OpenMM
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.openmm.app import PDBFile
from openforcefield.typing.engines.smirnoff import *


import mdtraj as md
import parmed
from pdbfixer import PDBFixer # for solvating
# from rdkit import Chem
import pickle
import shutil
import os
##############################################################
def get_data_filename(relative_path): #TODO put in utils
    """Get the full path to one of the reference files in testsystems.
    In the source distribution, these files are in ``openforcefield/data/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.
    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the repex folder).
    """

    from pkg_resources import resource_filename
    fn = resource_filename('mdfptools', os.path.join('data', relative_path))

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn

class BaseParameteriser():
	system_pmd = None

	@classmethod
	def via_openeye(cls):
		raise NotImplementedError

	@classmethod
	def via_rdkit(cls):
		raise NotImplementedError

	@classmethod
	def pmd_generator(cls):
		raise NotImplementedError

	@classmethod
	def _rdkit_setter(cls, smiles):
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

	@classmethod
	def _rdkit_charger(cls, mol):
		if not hasattr(cls, "charge_engine"):
			raise ValueError("No Useable charge engine Exist")
		# cls.mol = cls.charge_engine(mol)
		# return cls.mol
		return cls.charge_engine(mol)

	@classmethod
	def _rdkit_parameteriser(cls, mol):
		from rdkit import Chem
		def create_system_from_molecule_rdk(forcefield, mol, verbose=False):
			"""
			Generate a System from the given OEMol and SMIRNOFF forcefield, return the resulting System.
			Parameters
			----------
			forcefield : ForceField
			    SMIRNOFF forcefield
			mol : RDKit molecule
			    Molecule to test (must have coordinates)
			Returns
			----------
			topology : OpenMM Topology
			system : OpenMM System
			positions : initial atomic positions (OpenMM)
			"""
			# Create system
			topology = utils.generateTopologyFromRDKMol(mol)
			system = forcefield.createSystem(topology, [mol], verbose=verbose)
			# Get positions
			coordinates = mol.GetConformer().GetPositions()
			natoms = len(coordinates)
			positions = np.zeros([natoms,3], np.float32)
			for index in range(natoms):
			    (x,y,z) = coordinates[index]
			    positions[index,0] = x
			    positions[index,1] = y
			    positions[index,2] = z
			positions = unit.Quantity(positions, unit.angstroms)
			return topology, system, positions
		def generateSMIRNOFFStructureRDK(molecule):
			"""
			Given an RDKit molecule, create an OpenMM System and use to
			generate a ParmEd structure using the SMIRNOFF forcefield parameters.
			"""
			from openforcefield.typing.engines.smirnoff import forcefield_rdk
			from openforcefield.typing.engines.smirnoff.forcefield_utils import create_system_from_molecule
			ff = utils.get_data_filename('forcefield/smirnoff99Frosst.ffxml')
			with open(ff) as ffxml:
			    mol_ff = forcefield_rdk.ForceField(ffxml)
			#TODO : integrate charges
			charged_molecule = molecule
			mol_top, mol_sys, mol_pos = create_system_from_molecule_rdk(mol_ff, charged_molecule)
			cls.top = mol_top
			cls.sys = mol_sys
			molecule_structure = parmed.openmm.load_topology(mol_top, mol_sys, xyz=mol_pos)
			return molecule_structure
		try:
			ligand_pmd = generateSMIRNOFFStructureRDK(mol)
		except:
			raise ValueError("Parameterisation Failed") #TODO

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
	def load_ddec_models(cls):
		from sklearn.externals import joblib
		if not hasattr(cls, "rf"):

			# supported elements (atomic numbers)
			# H, C, N, O, F, P, S, Cl, Br, I
			cls.element_list = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
			elementdict = {8:"O", 7:"N", 6:"C", 1:"H", \
			               9:"F", 15:"P", 16:"S", 17:"Cl", \
			               35:"Br", 53:"I"}

			#directory, containing the models
			cls.rf = {element : joblib.load(get_data_filename(elementdict[element] + ".model")) for element in cls.element_list}

		cls.charge_engine = cls._ddec_charger

	@classmethod
	def _ddec_charger(cls, mol):
		from rdkit import DataStructs
		from rdkit.Chem import AllChem
		num_atoms = mol.GetNumAtoms()

		# maximum path length in atompairs-fingerprint
		APLength = 4

		# check for unknown elements
		curr_element_list = []
		for at in mol.GetAtoms():
		  element = at.GetAtomicNum()
		  if element not in cls.element_list:
			  raise ValueError("Error: element {} has not been parameterised".format(element))
		  curr_element_list.append(element)
		curr_element_list = set(curr_element_list)

		pred_q = [0]*num_atoms
		sd_rf = [0]*num_atoms
		# loop over the atoms
		for i in range(num_atoms):
		  # generate atom-centered AP fingerprint
		  fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, maxLength=APLength, fromAtoms=[i])
		  arr = np.zeros(1,)
		  DataStructs.ConvertToNumpyArray(fp, arr)
		  # get the prediction by each tree in the forest
		  element = mol.GetAtomWithIdx(i).GetAtomicNum()
		  per_tree_pred = [tree.predict(arr.reshape(1,-1)) for tree in (cls.rf[element]).estimators_]
		  # then average to get final predicted charge
		  pred_q[i] = np.average(per_tree_pred)
		  # and get the standard deviation, which will be used for correction
		  sd_rf[i] = np.std(per_tree_pred)

		#########################
		# CORRECT EXCESS CHARGE #
		#########################

		# calculate excess charge
		deltaQ = sum(pred_q)- float(AllChem.GetFormalCharge(mol))
		charge_abs = 0.0
		for i in range(num_atoms):
		  charge_abs += sd_rf[i] * abs(pred_q[i])
		deltaQ /= charge_abs
		# correct the partial charges
		for i,atm in enumerate(mol.GetAtoms()):
		  tmp = pred_q[i] - abs(pred_q[i]) * sd_rf[i] * deltaQ
		  atm.SetDoubleProp("PartialCharge", tmp)
		return mol

	@classmethod
	def unload_ddec_models(cls):
		del cls.rf, cls.element_list, cls.charge_engine

	@classmethod
	def _openeye_setter(cls, smiles):
		from openeye import oechem # OpenEye Python toolkits
		from openeye import oeomega # Omega toolkit
		from openeye import oequacpac #Charge toolkit
		from openeye import oedocking # Docking toolkit

		cls.smiles = smiles
		cls.omega = oeomega.OEOmega()
		cls.omega.SetMaxConfs(100) #Generate up to 100 conformers since we'll use for docking
		cls.omega.SetIncludeInput(False)
		cls.omega.SetStrictStereo(False) #Refuse to generate conformers if stereochemistry not provided
		cls.charge_engine = oequacpac.OEAM1BCCCharges()
		mol = oechem.OEMol()
		oechem.OESmilesToMol(mol, smiles)
		oechem.OEAddExplicitHydrogens(mol)
		oechem.OETriposAtomNames(mol)
		return mol

	@classmethod
	def _openeye_charger(cls, mol):
		# Set to use a simple neutral pH model
		#oequacpac.OESetNeutralpHModel(mol) #FIXME input smiles should ensure neutral molecule

		# Generate conformers with Omega; keep only best conformer
		status = cls.omega(mol)
		if not status:
		    raise ValueError("Error generating conformers for %s." % (cls.smiles)) #TODO

		# Assign AM1-BCC charges
		oequacpac.OEAssignCharges(mol, cls.charge_engine)
		return mol

	@classmethod
	def _openeye_parameteriser(cls, mol):
		try:
			ligand_pmd = utils.generateSMIRNOFFStructure(mol)
		except:
			raise ValueError("Parameterisation Failed") #TODO

		ligand_pmd.title = cls.smiles

		for i in ligand_pmd.residues:
			i.name = 'LIG'

		tmp_dir = tempfile.mkdtemp()
        # We need all molecules as both pdb files (as packmol input)
        # and mdtraj.Trajectory for restoring bonds later.
		pdb_filename = tempfile.mktemp(suffix=".pdb", dir=tmp_dir)
		oechem.OEWriteMolecule( oechem.oemolostream( pdb_filename ), mol)
		cls.pdb_filename = pdb_filename
		cls.ligand_pmd = ligand_pmd

	@classmethod
	def save(cls, filename, filepath = "./"):
		pickle_out = open(filepath + "{}.pickle".format(filename), "wb")
		pickle.dump(cls.system_pmd , pickle_out)
		pickle_out.close()

	run = via_openeye

class LiquidParameteriser(BaseParameteriser):
	@classmethod
	def via_openeye(cls, smiles, density, num_lig = 100):
		"""
		density : simtk.unit
		"""
		mol = cls._openeye_setter(smiles)
		mol = cls._openeye_charger(mol)
		cls._openeye_parameteriser(mol)

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
		except:
			pass

		return cls.system_pmd


	@classmethod
	def via_rdkit(cls, smiles):
		raise NotImplementedError

	run = via_openeye

class SolutionParameteriser(BaseParameteriser):
	solvent_pmd = parmed.load_file(get_data_filename("tip3p.prmtop")) #FIXME #TODO
	default_padding = 1.25


	@classmethod
	def via_openeye(cls, smiles):
		#TODO currently only supports one solute molecule

		mol = cls._openeye_setter(smiles)
		mol = cls._openeye_charger(mol)
		cls._openeye_parameteriser(mol)

		return cls.pmd_generator()

	@classmethod
	def via_rdkit(cls, smiles):
		mol = cls._rdkit_setter(smiles)
		mol = cls._rdkit_charger(mol)
		cls._rdkit_parameteriser(mol)
		return cls.pmd_generator()

	@classmethod
	def pmd_generator(cls):
		fixer = PDBFixer(cls.pdb_filename)
		fixer.addSolvent(padding = cls.default_padding)

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

		return cls.system_pmd


	run = via_openeye

class VaccumParameteriser(BaseParameteriser):
	@classmethod
	def via_openeye(cls, smiles, density, num_lig = 100):
		mol = cls._openeye_setter(smiles)
		mol = cls._openeye_charger(mol)
		cls._openeye_parameteriser(mol)

		cls.system_pmd = cls.ligand_pmd

		return cls.system_pmd


	@classmethod
	def via_rdkit(cls, smiles):
		raise NotImplementedError

	run = via_openeye

"""
from mdfptools.Parameteriser import *
print(SolutionParameteriser().run("CC"))
SolutionParameteriser.load_ddec_models()
print(SolutionParameteriser().via_rdkit("CC"))
SolutionParameteriser.unload_ddec_models()
print(LiquidParameteriser().run("CC", density = 12 * unit.gram / unit.liter))
"""
