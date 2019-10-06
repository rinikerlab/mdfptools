import sys
from rdkit import Chem
from rdkit.Chem import SaltRemover, AllChem, Draw

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
