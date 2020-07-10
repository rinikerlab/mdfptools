#! /bin/bash

#########################################################################################################
#  DESCRIPTION
#  Run the steps descripbed in the section 4 "Run MD Simulations" of Tutorial 1
#  
#  These are:
#  1. Energy Minimization
#  2. 50 ps NPT equilibration with velocity-rescaling thermostat and Berendsen barostat
#  3. 50 ps NPT equilibration with Nose-Hover thermostat and Parrinello-Rahman barostat
#  4. 5 ns NVT production. The temperature is kept at 298.15K using the Nose-Hoover chain thermostat
#  5. Re-write the trajectory centering the compound in the box
#  6. Re-compute energy terms using the Reaction Field method for long-range electrostatic interactions
#
#########################################################################################################

usage() 
{
echo "
run_md_water_4tutorial1.sh 
        Usage: $0 [-w working_folder] [-i inputs_folder]
        Options:
                -w   working_folder           directory containing topology .top and structure .gro files 
                -i   inputs_folder            directory containing the input .mdp files for gromacs
                -h                            Show this message
"
1>&2; exit 1;}

while getopts ":w:i:" options; do
    case "${options}" in
        w)
            working_folder=${OPTARG}
            if [[ $working_folder == */ ]]; then tmp=`echo $working_folder | rev | cut -d / -f2- | rev`; working_folder="$tmp"; fi
            if [ ! -d "$working_folder" ]; then
                echo ""
                echo "ERROR: Folder ${working_folder} does not exist"
                usage
                exit 1
            fi
            if [ ! "$(ls -A ${working_folder}/*.top)" ]; then
                echo ""
                echo "ERROR: Folder ${working_folder} does not contain any topology .top file."
                echo "Are you sure this is the right working directory?"
                usage
                exit 1 
            fi
            if [ ! "$(ls -A ${working_folder}/*.gro)" ]; then
                echo ""
                echo "ERROR: Folder ${working_folder} does not contain any structure .gro file."
                echo "Are you sure this is the right working directory?"
                usage
                exit 1 
            fi
            ;;
        i)
            inputs_folder=${OPTARG}
            if [[ $inputs_folder == */ ]]; then tmp=`echo $inputs_folder | rev | cut -d / -f2- | rev`; inputs_folder="$tmp"; fi
            if [ ! -d "$inputs_folder" ]; then
                echo ""
                echo "ERROR: Folder ${inputs_folder} does not exist"
                usage
                exit 1
            fi
            if [ ! "$(ls -A ${inputs_folder}/*.mdp)" ]; then
                echo ""
                echo "ERROR: Folder ${inputs_folder} does not contain any input .mdp file."
                echo "Are you sure this is the right working directory?"
                usage
                exit 1 
            fi
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "${working_folder}" ] || [ -z "${inputs_folder}" ]; then
    usage
fi

echo "working_folder = ${working_folder}"
echo "inputs_folder = ${inputs_folder}"


# Copy input files to the working directory
cp ${inputs_folder}/*mdp ${working_folder}/ 

# Enter the working directory
cd $working_folder

for file in *.top
do
name=`echo $file | cut -d . -f 1`

gmx make_ndx -f ${name}.gro -o index_${name}.ndx<<EOF
q
EOF
check_group=`grep Water_and_ions index_"$name".ndx`

if [ -z "$check_group" ]
then
group="a"
else
group="b"
fi

# Run Minimization 
gmx grompp -f minim_wat.mdp -c "$name".gro -p "$name".top -n index_"$name".ndx -o em_"$name".tpr
gmx mdrun -npme 0 -s em_${name}.tpr -c em_${name}.gro; wait; sleep 3; 

# Run 50 ps NPT equilibration with V-rescale Thermostat and Berendsen Barostat
gmx grompp -f npt_wat_1${group}.mdp -c em_${name}.gro -p "$name".top -n index_"$name".ndx -o npt1_"$name".tpr
gmx mdrun -npme 0 -s npt1_"$name".tpr -c npt1_${name}.gro -e npt1_${name}.edr; wait; sleep 3; 

# Run 50 ps NPT equilibration with Nose-Hoover Thermostat and Parrinello-Rahman Barostat
gmx grompp -f npt_wat_2${group}.mdp -c npt1_${name}.gro -p "$name".top -n index_"$name".ndx -o npt2_"$name".tpr; 
gmx mdrun -npme 0 -s npt2_"$name".tpr -c npt2_${name}.gro -e npt2_${name}.edr; wait; sleep 3; 

rm \#* traj.trr

# Run 5 ns NVT production with Nose-Hoover Chain Thermostat 
gmx grompp -f nvt_wat_3${group}.mdp -c npt2_${name}.gro -p "$name".top -n index_"$name".ndx -o nvt3_"$name".tpr;
gmx mdrun -npme 0 -s nvt3_${name}.tpr -o nvt3_${name}.trr -cpo nvt3_${name}.cpt -c nvt3_${name}.gro -e nvt3_${name}.edr; wait; sleep 3;

# Post-Processing. Rewrite the trajectory centering the ligand in the box
gmx trjconv -f nvt3_${name}.trr -s nvt3_${name}.tpr -n index_${name}.ndx -pbc mol -ur compact -center -o center_nvt3_${name}.trr<<EOF
2
System
EOF
wait 
sleep 3;

# Post-Processing. Evaluate energy terms with RF scheme for electrostic interactions
gmx grompp -f nvt_rf_wat_3${group}.mdp -c nvt3_${name}.gro -p ${name}.top -n index_${name}.ndx -o nvt3_rf_${name}.tpr;
gmx mdrun -s nvt3_rf_${name}.tpr -rerun center_nvt3_${name}.trr -e nvt3_rf_${name}.edr;
if [ -z "$check_group" ]
then
gmx energy -f nvt3_rf_${name}.edr -o nvt3_rf_${name}.xvg<<EOF
Coul-SR:Water-LIG
LJ-SR:Water-LIG
Coul-14:Water-LIG
LJ-14:Water-LIG
Coul-SR:LIG-LIG
LJ-SR:LIG-LIG
Coul-14:LIG-LIG
LJ-14:LIG-LIG

EOF
wait
sleep 3;
else
group="b"
gmx energy -f nvt3_rf_${name}.edr -o nvt3_rf_${name}.xvg<<EOF
Coul-SR:Water_and_ions-LIG
LJ-SR:Water_and_ions-LIG
Coul-14:Water_and_ions-LIG
LJ-14:Water_and_ions-LIG
Coul-SR:LIG-LIG
LJ-SR:LIG-LIG
Coul-14:LIG-LIG
LJ-14:LIG-LIG

EOF
wait
sleep 3;
fi

# Rewrite the trajectories in the xtc format that occupies less memory compared to the trr format.
gmx trjconv -f center_nvt3_${name}.trr -o center_nvt3_${name}.xtc;
# Convert gro to pdb structure. It will be needed for reading the trajectory in python
gmx trjconv -f nvt3_${name}.gro -s nvt3_${name}.tpr -o nvt3_${name}.pdb<<EOF
System
EOF
wait
sleep 3;

# Remove trr trajectories that occupy a lot of memory
rm nvt3_${name}.trr center_nvt3_${name}.trr;
rm \#*;
wait
sleep 3

done

