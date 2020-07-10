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

parser = argparse.ArgumentParser(description='Generate 3D-structures at pH = 7 from a SDF file. \n Output = One _pH7.pdb file for each molecule in the input SDF file')
parser.add_argument('-isdf', metavar='input_file.sdf', help='input SDF file', required = True)
parser.add_argument('-calc_pH', metavar='False', help='True to calculate the main protomer form using the cxcalc function of ChemAxon', required = False, default=False)
parser.add_argument('-pH', metavar='7.0', help='pH at which to evaluate the protomer forms', required = False, default=7.0)
parser.add_argument('-output_folder', help='folder in which to save the output PDB files', required = False, default=".")

args = parser.parse_args()

out_dir = args.output_folder 

if args.isdf:
    input_file = args.isdf
    suppl = Chem.SDMolSupplier(input_file)
    output_name = input_file.split(".")[0]
    w = Chem.SDWriter('{}_3d.sdf'.format(output_name))
    count = 1
    for m in suppl:
        try:
            molname = m.GetProp("RootNumber")
        except:
            try:
                molname = m.GetProp("_Name")
            except:
                molname = "mol_{}".format(count)
                count+=1
        if len(molname) == 0:
            molname = "mol_{}".format(count)
            count+=1
        print(molname)
        m2=Chem.AddHs(m)
        AllChem.EmbedMolecule(m2)
        try:
            AllChem.MMFFOptimizeMolecule(m2)
            w = Chem.SDWriter('{}_3d.sdf'.format(molname))
            w.write(m2)
        except:
            print("Skip {}".format(m2.GetProp("_Name")))
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


