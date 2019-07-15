import hashlib, sys
from rdkit import Chem
from rdkit.Chem import SaltRemover, AllChem, Draw



def canonical_smiles_from_smiles(smiles, sanitize = True):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize = sanitize)
        mol.UpdatePropertyCache()
        #mol = Chem.AddHs(mol)
        Chem.GetSSSR(mol)
        return Chem.MolToSmiles(mol,canonical=True, allHsExplicit=True,  kekuleSmiles = False, allBondsExplicit = True, isomericSmiles = True)
    except:
        return None

def hashing(smiles):
    hash_object = hashlib.md5(canonical_smiles_from_smiles(smiles).encode("utf-8"))
    return hash_object.hexdigest()

    
def screen_organic(smiles):
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
