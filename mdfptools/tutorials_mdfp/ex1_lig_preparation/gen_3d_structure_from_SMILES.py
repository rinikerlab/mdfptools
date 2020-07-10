#############################
#  DESCRIPTION
#  The gen_3d_structure scripts use the EmbedMolecule function of RDKit to generate the 3D structure of compounds, which is then minimized using MM force field within RDKit. 
#  The ChemAxon majorms function is used to determined the protonation state at a pH of 7. 
#  The lines to generate the structures of the neutralized molecules are commneted since the simulations in the lipid bilayer did not lead to significant results. 
#   
############################

import argparse
from rdkit import Chem
from rdkit.Chem import SaltRemover, AllChem
import sys
import subprocess
import pandas as pd

parser = argparse.ArgumentParser(description='Generate 3D-structures at pH = 7 from a SMILES file. \n One output _pH7.pdb file for each SMILES in the input .smi file')
parser.add_argument('-ismi', metavar='input_file.smi', help='Input file containing a list of SMILES. It must contain a "SMILES" column. Columns "CMPD_ID" and "is_sub" are optional. The "CMPD_ID" column contains compound names while the "is_sub" column contains the classification labels. If the "CMPD_ID" column is missing, then compounds will be named as mol_N. If the "is_sub" column is present then the pdb output files will be saved into two directories, "substrates" and "nonsubstrates", instead of the current directory', required = True)
parser.add_argument('-sep', type=str, default=" ", help='column separator. Default = " "')
parser.add_argument('-calc_pH', metavar='False', help='True to calculate the main protomer form using the cxcalc function of ChemAxon', required = False, default=False)
parser.add_argument('-pH', metavar='7.0', help='pH at which to evaluate the protomer forms', required = False, default=7.0)
parser.add_argument('-output_folder', help='folder in which to save the output PDB files', required = False, default=".")
args = parser.parse_args()

out_dir = args.output_folder 

if args.ismi:
    input_file = args.ismi
    data = pd.read_table(input_file, sep=args.sep)
    try:
        smiles = list(data['SMILES'])
    except:
        print("IOError: The input file does not contain a SMILES column")
    try: 
        cmpd_name = list(data['CMPD_ID'])
    except:
        cmpd_name = ['mol_{}'.format(i) for i in range(1,len(smiles)+1)] 
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        molname = cmpd_name[i]
        try:
            m2=Chem.AddHs(m)
            AllChem.EmbedMolecule(m2)
            AllChem.MMFFOptimizeMolecule(m2)
            m2.SetProp('_Name', molname)
            m2.SetProp('CMPD_NAME', molname)
            m2.GetProp('CMPD_NAME')
            w = Chem.SDWriter('{}_3d.sdf'.format(molname))
            w.write(m2)
        except:
            print("Skip {}".format(molname))
            continue
        if args.calc_pH == True:
            args_subproc = "sed -i '/CHG/d' " + molname + "_3d.sdf"
            subprocess.call(args_subproc, shell=True)
            args_subproc = "cxcalc -o " + molname + "_pH7.sdf majorms -H {} -f sdf " + molname + "_3d.sdf".format(args.pH)
            subprocess.call(args_subproc, shell=True)
            args_subproc = "obabel -isdf " + molname + "_pH7.sdf  -opdb -O " + out_dir + "/" + molname + "_pH7.pdb -h"
            subprocess.call(args_subproc, shell=True)
            args_subproc = "rm *_pH7.sdf *_3d.sdf"
            subprocess.call(args_subproc, shell=True)
            # To eventually neutralize the compounds using ChemAxon
            #args_subproc = "standardize -c neutralize " + molname + "_3d.sdf -f sdf -o " + molname + "_neutr.sdf"
            #subprocess.call(args_subproc, shell=True)
            #args_subproc = "obabel -isdf " + molname + "_neutr.sdf" + " -opdb -O " + molname + "_neutr.pdb"
            #subprocess.call(args_subproc, shell=True)
            #args_subproc = "rm *_neutr.sdf"
            #subprocess.call(args_subproc, shell=True)
        else:
            args_subproc = "obabel -isdf " + molname + "_3d.sdf  -opdb -O " + out_dir + "/" + molname + "_3d.pdb -h"
            subprocess.call(args_subproc, shell=True)            
        print("Done")
    
    
