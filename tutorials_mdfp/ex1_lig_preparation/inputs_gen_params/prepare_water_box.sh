#############################################
#  DESCRIPTION 
#  Solvate the compounds in a water box using tleap.
#  A cubic box with a minimum distance of 10 A between the compound and the wall is built.
#  The python script amber2gromacs.py converts the outputs from tleap (.inpcrd and .prmtop)
#  into a format readable by GROMACS (.gro and .top)
#
#  Inputs: compound_netcharge.mol2   compound_netcharge.frcmod 
#  Outputs ./WAT_box/*.gro and ./WAT_box/*top files
##############################################

[[ -d WAT_box ]] || mkdir WAT_box 


source deactivate
for mol in *netcharge.mol2
do
name=`echo $mol | cut -d . -f 1`
sed 's/test/'$name'/g' tleap.in > tleap_tmp.in
tleap -f tleap_tmp.in >> tleap.log 
done



