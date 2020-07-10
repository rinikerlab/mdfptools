#############################
#  DESCRIPTION
#  The clean_2d_structures.py script uses the SaltRemover function of RDKit to separate compounds from salts.
#  The largest fragment is assumed to be the compound and written in the ouput file.
#   
############################

import argparse
from rdkit import Chem
from rdkit.Chem import SaltRemover, AllChem
import sys

parser = argparse.ArgumentParser(description='Remove Salts from SDF structure files. Output = filename_clean.sdf')
parser.add_argument('-isdf', metavar='input_file.sdf', help='input SDF file', required = True)
args = parser.parse_args()

if args.isdf:
    input_file = args.isdf
    suppl = Chem.SDMolSupplier(input_file)
    output_name = input_file.split(".")[0]
    w = Chem.SDWriter('{}_clean.sdf'.format(output_name))
    # salt remover
    remover = SaltRemover.SaltRemover()
    count = 0
    for m in suppl:
        # remove salts
        res = remover.StripMol(m)
        if res is not None and res.GetNumAtoms() > 0:
          m = res
        else:
          continue
        # take only the largest fragment
        frags = AllChem.GetMolFrags(m, asMols=True)
        if len(frags) < 1: continue
        nums = [(f.GetNumAtoms(), f) for f in frags]
        nums = sorted(nums, key=lambda x: x[0])
        m = nums[0][1]
        try:
            w.write(m)
            count += 1
        except:
            continue
    w.close()
    print(count)



