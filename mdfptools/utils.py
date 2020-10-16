import sys
from rdkit import Chem
from rdkit.Chem import SaltRemover, AllChem, Draw
import simtk.unit as units


def get_data_filename(relative_path): #TODO put in utils
    """Get the full path to one of the reference files in testsystems.
    In the source distribution, these files are in ``mdfptools/data/``,
    but on installation, they're moved to somewhere in the user's python site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the repex folder).

    Returns
    ---------
    fn : str
        filename
    """

    import os
    from pkg_resources import resource_filename
    fn = resource_filename('mdfptools', os.path.join('data', relative_path))

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn

def canonical_smiles_from_smiles(smiles, sanitize = True):
    """
    Apply canonicalisation with rdkit

    Parameters
    ------------
    smiles : str
    sanitize : bool
        Wether to apply rdkit sanitisation, default yes.

    Returns
    ---------
    canonical_smiles : str
        Returns None if canonicalisation fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize = sanitize)
        mol.UpdatePropertyCache()
        #mol = Chem.AddHs(mol)
        Chem.GetSSSR(mol)
        return Chem.MolToSmiles(mol,canonical=True, allHsExplicit=True,  kekuleSmiles = False, allBondsExplicit = True, isomericSmiles = True)
    except:
        return None

def hashing(smiles):
    """
    Converts a string to hexdecimal representation (length 32). Specifically, it is used in mdfptools to convert canonical smiles to hex so it can be used as filename when store to disk.

    Parameters
    -----------
    smiles : str

    Returns
    ------------
    hex_str : str
        Hexdecimal representation
    """
    import hashlib
    hash_object = hashlib.md5(canonical_smiles_from_smiles(smiles).encode("utf-8"))
    return hash_object.hexdigest()


def screen_organic(smiles):
    """
    Heuristic to determine if a input SMILES string is considered as only organic matter.


    Parameters
    -----------
    smiles : str

    Returns
    ------------
    is_organic : bool
    """
    if smiles is None: return False
    remover = SaltRemover.SaltRemover()

# SMARTS pattern for organic elements
# H, B, C, N, O, F, P, S, Cl, Br, I
    patt = '[!$([#1,#5,#6,#7,#8,#9,#15,#16,#17,#35,#53])]'
    mpatt = Chem.MolFromSmarts(patt)
    m = Chem.MolFromSmiles(smiles, sanitize = True)
    if m is None: return False

    # remove salts
    res = remover.StripMol(m)
    if res is not None and res.GetNumAtoms() < m.GetNumAtoms():
        return False

    # take only the largest fragment
    frags = AllChem.GetMolFrags(m, asMols=True)
    if len(frags) > 1:
        return False
#     nums = [(f.GetNumAtoms(), f) for f in frags]
#     nums.sort(reverse=True)
#     m = nums[0][1]

    # take only organic molecules
    if not m.HasSubstructMatch(mpatt):
        return True
    else:
        return False


def approximate_volume_by_density(smiles_strings, n_molecules_list, density=1.0,
                                  box_scaleup_factor=1.1, box_buffer=2.0):
    """Generate an approximate box size based on the number and molecular weight of molecules present, and a target density for the final solvated mixture. If no density is specified, the target density is assumed to be 1 g/ml.

    This is the adapted from the `openmoltools` package where the mol handling is switched to RDKit here. The calculated molecular weight can differ slightly to that of openeye, but better than `Chem.ExactMolWt`

    Parameters
    ----------
    smiles_strings : list(str)
        List of smiles strings for each component of mixture.
    n_molecules_list : list(int)
        The number of molecules of each mixture component.
    box_scaleup_factor : float, optional, default = 1.1
        Factor by which the estimated box size is increased
    density : float, optional, default 1.0
        Target density for final system in g/ml
    box_buffer : float [ANGSTROMS], optional, default 2.0.
        This quantity is added to the final estimated box size
        (after scale-up). With periodic boundary conditions,
        packmol docs suggests to leave an extra 2 Angstroms
        buffer during packing.
    Returns
    -------
    box_size : float
        The size (edge length) of the box to generate.  In ANGSTROMS.
    Notes
    -----
    By default, boxes are only modestly large. This approach has not been extensively tested for stability but has been used in th Mobley lab for perhaps ~100 different systems without substantial problems.
    """

    density = density * units.grams/units.milliliter

    #Load molecules to get molecular weights
    wts = []
    mass = 0.0*units.grams/units.mole * 1./units.AVOGADRO_CONSTANT_NA #For calculating total mass
    for (idx,smi) in enumerate(smiles_strings):
        mol = Chem.MolFromSmiles(smi)
        wts.append( Chem.Descriptors.MolWt(mol)*units.grams/units.mole )
        mass += n_molecules_list[idx] * wts[idx] * 1./units.AVOGADRO_CONSTANT_NA

    #Estimate volume based on mass and density
    #Density = mass/volume so volume = mass/density (volume units are ml)
    vol = mass/density
    #Convert to box length in angstroms
    edge = vol**(1./3.)

    #Compute final box size
    box_size = edge*box_scaleup_factor/units.angstroms + box_buffer

    return box_size