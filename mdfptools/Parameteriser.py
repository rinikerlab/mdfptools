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
from rdkit import Chem
import pickle
import shutil
##############################################################
class BaseParameteriser():
	system_pmd = None

	@classmethod
	def via_openeye(cls):
		raise NotImplementedError

	@classmethod
	def via_rdkit(cls):
		raise NotImplementedError

	@classmethod
	def _openeye_setter(cls):
		from openeye import oechem # OpenEye Python toolkits
		from openeye import oeomega # Omega toolkit
		from openeye import oequacpac #Charge toolkit
		from openeye import oedocking # Docking toolkit

		cls.omega = oeomega.OEOmega()
		cls.omega.SetMaxConfs(100) #Generate up to 100 conformers since we'll use for docking
		cls.omega.SetIncludeInput(False)
		cls.omega.SetStrictStereo(False) #Refuse to generate conformers if stereochemistry not provided
		cls.chargeEngine = oequacpac.OEAM1BCCCharges()

	@classmethod
	def _openeye_charger(cls, smiles):
		mol = oechem.OEMol()
		oechem.OEParseSmiles(mol, smiles)
		# Set to use a simple neutral pH model
		#oequacpac.OESetNeutralpHModel(mol) #FIXME input smiles should ensure neutral molecule

		# Generate conformers with Omega; keep only best conformer
		status = cls.omega(mol)
		if not status:
		    raise ValueError("Error generating conformers for %s." % (smiles)) #TODO

		# Assign AM1-BCC charges
		oequacpac.OEAssignCharges(mol, cls.chargeEngine)


		try:
			ligand_pmd = utils.generateSMIRNOFFStructure(mol)
		except:
			raise ValueError("Parameterisation Failed") #TODO

		ligand_pmd.title = smiles

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
		cls._openeye_setter()
		cls._openeye_charger(smiles)

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
	solvent_pmd = parmed.load_file("./mdfptools/data/tip3p.prmtop") #FIXME #TODO

	@classmethod
	def via_openeye(cls, smiles):
		#TODO currently only supports one solute molecule

		cls._openeye_setter()
		cls._openeye_charger(smiles)

		fixer = PDBFixer(cls.pdb_filename)
		fixer.addSolvent(padding = 1.25)

		tmp_dir = tempfile.mkdtemp()
		cls.pdb_filename = tempfile.mktemp(suffix=".pdb", dir=tmp_dir)
		PDBFile.writeFile(fixer.topology, fixer.positions, open(cls.pdb_filename, 'w'))

		complex = parmed.load_file(cls.pdb_filename)

		solvent = complex["(:HOH)"]
		num_solvent = len(solvent.residues)

		cls.solvent_pmd *= num_solvent
		cls.solvent_pmd.positions = solvent.positions

		cls.system_pmd = cls.ligand_pmd + cls.solvent_pmd
		cls.system_pmd.box_vectors = complex.box_vectors

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

class VaccumParameteriser(BaseParameteriser):
	@classmethod
	def via_openeye(cls, smiles, density, num_lig = 100):
		cls._openeye_setter()
		cls._openeye_charger(smiles)

		cls.system_pmd = cls.ligand_pmd

		return cls.system_pmd


	@classmethod
	def via_rdkit(cls, smiles):
		raise NotImplementedError

	run = via_openeye

"""
print(SolutionParameteriser().run("CC"))
print(LiquidParameteriser().run("CC", density = 12 * unit.gram / unit.liter))
"""
