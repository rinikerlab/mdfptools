#############################################################
#  DESCRIPTION
#  Use Ambertools to generate ligand parameters (Step 2 of Tutorial 1).
#  Derive BCC partial charges using antechamber and 
#  extract bonded and non-bonded GAFF parameters using parmchk
#############################################################


for mol in *.pdb
do
name=`echo $mol | cut -d . -f 1`
cmpd_name=`echo $mol | cut -d _ -f 1`
charge=`cxcalc formalcharge -H 7.0 $mol | tail -1 | awk '{print $2}'`

# calculate AM1-BCC charges 
antechamber -i $mol -fi pdb -o "$name".mol2 -fo mol2 -c bcc -s 2 -nc "$charge"

# check for missing bonded and non-bonded parameters in GAFF force field. 
# It generates frcmod file that can be loaded into LEaP in order to add missing parameters. 
# Command "ATTN" means revision. It requires manual parametrization.
parmchk2 -i "$name".mol2 -f mol2 -o "$name".frcmod
done


rm ATOMTYPE.INF sqm.in sqm.out sqm.pdb tleap.log *tauto_proto* ANTECHAMBER_* leap.log

for mol in *.mol2
do
name=`echo $mol | cut -d . -f 1`
sed 's/test/'$name'/g' tleap_lig.in > tleap_lig_tmp.in
tleap -f tleap_lig_tmp.in >> tleap.log
done



