from ComposerGmx import *
import argparse
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description='DESCRIPTION: \n ---------------- \n Compose MDFP from a MD trajectory. \n It returns a dataframe (pkl file) containing the MDFP terms and 2D-properties of the analyzed compound. \n To compute MD terms, the arguments needed are: -traj_file, -coord_file, -top_file, and -energy_file \n To compute 2D count and properties, specify -sdf_file or -ismi \n \n ARGUMENTS \n ---------------- \n', formatter_class=RawTextHelpFormatter)
parser.add_argument('-traj_file',  metavar='traj.xtc', help='Trajectory file (.xtc, .trr, .h5)', required = False)
parser.add_argument('-coord_file', metavar='coord.gro', help='Coordinates file (.gro, .pdb)', required = False)
parser.add_argument('-energy_file', metavar='energy.xvg', help='File containing the energy terms extracted from the simulations (.xvg)', required = False)
parser.add_argument('-sdf_file', metavar='compound.sdf', help='SDF file of the compound. Used to compute topological counts and properties. Alternative to -ismi', required = False)
parser.add_argument('-ismi', metavar='SMILES', help='SMILES of the compound. Used to compute topological counts and properties. Alternative to -sdf_file.', required = False)
parser.add_argument('-cmpd_name', metavar='compound_ID', help='Name of the compound being analyzed. Returned in the output dataframe if specified', required = False)
parser.add_argument('-output_filename', metavar='output', help='If specified, the output file name is "MDFP_output.pkl". Otherwise the output file is "MDFP.pkl" or "MDFP_cmpd_name.pkl" if -cmpd_name is specified)', required = False) 
parser.add_argument('-output_folder', help='Folder in which the output file is written (Default = ".")', required = False) 
args = parser.parse_args()


Fingerprints = []
Dict_Fingerprint = {}

# append compound name to the dictionary
if args.cmpd_name:
    print("Analyzing {}".format(args.cmpd_name))
    Dict_Fingerprint.update({"cmpd_name": args.cmpd_name})

#load 2D-structure of the molecules and compute 2D-counts and topological properties 
if args.sdf_file:
    try:
        mol = Chem.MolFromMolFile(args.sdf_file)
        smiles = Chem.MolToSmiles(mol)
        Dict_Fingerprint.update({"smiles": smiles})
        try:
            # Obtain 2D-count and properties
            Dict_Fingerprint.update(ComposerGMX.Properties2D_From_Mol(mol))
        except:
            print("Error: 2D topological counts could not be obtained")
    except:
        print("{} could not be read".format(args.sdf_file))

if args.ismi:
    try:
        mol = AllChem.MolFromSmiles(args.ismi)
        Dict_Fingerprint.update({"smiles": args.ismi})
        try:
            # Obtain 2D-count and properties
            Dict_Fingerprint.update(ComposerGMX.Properties2D_From_SMILES(mol))
        except:
            print("Error: 2D topological counts could not be obtained")
    except:
        print("A molecule could not generated from the SMILES {}".format(args.ismi))


# extract energy terms from energy files
if args.energy_file:
    Dict_Fingerprint.update(ComposerGMX.EnergyTermsGmx(args.energy_file))
else:
    print("Energy terms could not be obtained. Energy file not provided as input.")

if args.traj_file and args.coord_file:
    if args.coord_file.endswith(".gro"):
        pdb_file = os.path.splitext(args.coord_file)[0] + '.pdb'
        if not os.path.isfile(pdb_file):
            ComposerGMX.gro2pdb(args.coord_file)
    elif args.coord_file.endswith(".pdb"):
       pdb_file = args.coord_file
    try:
        # load trajectory
        pdb = md.load(pdb_file)
        topology = pdb.topology
        solute_atoms = ComposerGMX.solute_solvent_split(topology)[0]
        if args.traj_file.endswith(".trr"):
            solute_traj = md.load_trr(args.traj_file, top=pdb_file, atom_indices = solute_atoms)
        if args.traj_file.endswith(".xtc") or args.traj_file.endswith(".h5"):
            solute_traj = md.load(args.traj_file, top=pdb_file, atom_indices = solute_atoms)
        # Compute MDFP
        Dict_Fingerprint.update(ComposerGMX.extract_rgyr(solute_traj)) 
        Dict_Fingerprint.update(ComposerGMX.extract_sasa(solute_traj))
    except:
        print("Error: Rgyr and SASA could not be obtained")
    try:
        Dict_Fingerprint.update(ComposerGMX.extract_psa3d(args.traj_file, args.coord_file, include_SandP = True))
    except:
        print("Error: 3D-PSA could not be obtained")

print("Done")
Fingerprints.append(Dict_Fingerprint)
MDFP = pd.DataFrame(Fingerprints)

if args.output_folder:
    output_folder = args.output_folder
else:
    output_folder = "."

if args.output_filename:
    MDFP.to_pickle("{}/MDFP_{}.pkl".format(output_folder, args.output_filename))
elif args.cmpd_name:
    MDFP.to_pickle("{}/MDFP_{}.pkl".format(output_folder, args.cmpd_name))
else:
    MDFP.to_pickle("{}/MDFP.pkl".format(output_folder))




