import tempfile
import parmed
from simtk import unit
from simtk.openmm import app

from simtk.openmm import *
from simtk.openmm.app import *
from mdtraj.reporters import HDF5Reporter
from .utils import *

import os

class BaseSimulator():
    """
    .. warning :: The base class should not be used directly

    Parameters
    ------------
    temperature : simtk.unit
        default 298.15 K
    pressure : simtk.unit
        default 1.013 bar
    time_step : simtk.unit
        default 2 fs


    .. todo::
        - setter and getter for phy constants
    """

    temperature = 298.15 * unit.kelvin
    pressure = 1.013 * unit.bar
    time_step = 0.002 * unit.picoseconds
    equil_steps = 50000  #100 ps

    @classmethod
    def via_openmm(cls, parmed_obj, file_name, file_path = "./", platform = "CUDA", num_steps = 5000 * 500, write_out_freq = 5000, report_equilibration = False, report_production = False, **kwargs):
        """
        Runs simulation using OpenMM.

        Parameters
        ------------
        parmed_obj : parmed.structure
            Parmed object of the fully parameterised simulated system.
        file_name : str
            No file type postfix is necessary
        file_path : str
            Default to current directory
        platform : str
            The computing architecture to do the calculation, default to CUDA, CPU, OpenCL is also possible.
        num_steps : int
            Number of production simulation to run, default 2,500,000 steps, i.e. 5 ns.
        write_out_freq : int
            Write out every nth frame of simulated trajectory, default to every 5000 frame write out one, i.e. 10 ps per frame.

        Returns
        --------
        path : str
            The absolute path where the trajectory is written to.
        """
        platform = Platform.getPlatformByName(platform)
        pmd = parmed_obj
        path = '{}/{}.h5'.format(file_path, file_name)

        system = pmd.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer, constraints=app.AllBonds)

        thermostat = AndersenThermostat(cls.temperature, 1/unit.picosecond)
        system.addForce(thermostat)
        barostat = MonteCarloBarostat(cls.pressure , cls.temperature)
        system.addForce(barostat)
        integrator = VerletIntegrator(cls.time_step)
        simulation = Simulation(pmd.topology, system, integrator, platform)

        simulation.context.setPeriodicBoxVectors(*pmd.box_vectors)
        simulation.context.setPositions(pmd.positions)
        simulation.minimizeEnergy()

        #Eq
        try:
            cls.equil_steps = kwargs["equil_steps"]
        except KeyError:
            pass
        if report_equilibration:
            #print(cls.equil_steps, " steps")
            simulation.reporters.append(StateDataReporter("{}/equilibration_{}.dat".format(file_path, file_name), cls.equil_steps//5000, step=True, volume = True, temperature = True))
        simulation.step(cls.equil_steps)

        state = simulation.context.getState(getPositions = True, getVelocities = True)
        pmd.positions, pmd.velocities, pmd.box_vectors = state.getPositions(),state.getVelocities(), state.getPeriodicBoxVectors()

        #Production
        del system
        del simulation

        system = pmd.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer, constraints=app.AllBonds)

        thermostat = AndersenThermostat(cls.temperature, 1/unit.picosecond)
        system.addForce(thermostat)
        #barostat = MonteCarloBarostat(1.013 * unit.bar, 298.15 * unit.kelvin)
        #system.addForce(barostat)
        integrator = VerletIntegrator(cls.time_step)
        simulation = Simulation(pmd.topology, system, integrator, platform)
        simulation.context.setPeriodicBoxVectors(*pmd.box_vectors)
        simulation.context.setPositions(pmd.positions)
        if report_production:
            simulation.reporters.append(StateDataReporter("{}/production_{}.dat".format(file_path, file_name), num_steps//50000, step=True, potentialEnergy = True, temperature = True))
        simulation.reporters.append(HDF5Reporter(path, write_out_freq))
        simulation.step(num_steps)

        return os.path.abspath(path)

    @classmethod
    def via_gromacs(cls, parmed_obj, file_name, file_path = "./", num_steps = 5000 * 500, write_out_freq = 5000, num_threads = 1, minimisation_mdp = None, equilibration_mdp = None, production_mdp = None, tmp_dir = None, report_equilibration = False, report_production = False, debug = False, **kwargs): #TODO infer threads based on system size
        """
        Simulation via GROMACS will be added in the future.

        based on cell size, make recomendation on number of threads to use 
    
        Parameters
        ------------
        parmed_obj : parmed.structure
            Parmed object of the fully parameterised simulated system.
        file_name : str
            No file type postfix is necessary
        file_path : str
            Default to current directory
        platform : str
            The computing architecture to do the calculation, default to CUDA, CPU, OpenCL is also possible.
        num_steps : int
            Number of production simulation to run, default 2,500,000 steps, i.e. 5 ns.
        write_out_freq : int
            Write out every nth frame of simulated trajectory, default to every 5000 frame write out one, i.e. 10 ps per frame.

        Returns
        --------
        path : str
            The absolute path where the trajectory is written to.
        """
        from biobb_md.gromacs.make_ndx import make_ndx, MakeNdx
        # from biobb_md.gromacs.grompp_mdrun import grompp_mdrun
        from biobb_md.gromacs.mdrun import mdrun
        from biobb_md.gromacs.grompp import grompp
        from biobb_common.tools.file_utils import zip_top
        

        assert tmp_dir is None or os.path.realpath(file_path) != os.path.realpath(tmp_dir), "Simulation results will not be stored in a temporary directory"

        if debug:
            import shutil
            tmp_dir = "{}/.debug/".format(file_path)
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)
        elif tmp_dir is None:
            tmp_dir = tempfile.mkdtemp(dir = file_path) #FIXME just change this to debug dir in dubug mode
        else:
            try:
                os.mkdir(tmp_dir)
            except:
                raise ValueError("Cannot create an empty temporary directory for storing intermediate files")
        
        parmed_obj.residues[0].name = "LIG"
        parmed_obj = parmed.gromacs.gromacstop.GromacsTopologyFile().from_structure(parmed_obj)


        parmed_obj.defaults.fudgeLJ = 0.5
        parmed_obj.defaults.fudgeQQ = 0.83333
        parmed_obj.defaults.gen_pairs = "yes"


        # Create prop dict and inputs/outputs

        xtc_file = '{}/{}.xtc'.format(file_path, file_name)
        gro_file = "{}/{}.gro".format(file_path, file_name)
        top_file = "{}/{}.top".format(tmp_dir, "topol")
        topzip_file = "{}/{}.zip".format(tmp_dir, "topol")
        
        # parmed_obj.save("{}/{}.pdb".format(tmp_dir, stage), overwrite=True)
        parmed_obj.save(gro_file, overwrite=True)
        parmed_obj.save(top_file, overwrite=True)

        index_file = "{}/{}.ndx".format(tmp_dir, "index")
        prop = {
            # "can_write_console_log" : False,
            'selection': "! r LIG"
        }

        # Create and launch bb
        # tmp = MakeNdx(input_structure_path = gro_file,
        #         output_ndx_path = index_file,
        #         properties = prop)
        # import logging
        # tmp.out_log = logging.Logger("x")
        # tmp.err_log = logging.Logger("y")
        # tmp.launch()
        # print(getattr(tmp, "out_log"))
        # print(tmp.err_log)
        make_ndx(input_structure_path = gro_file,
                output_ndx_path = index_file,
                properties = prop)

        ####################################################

        zip_top(topzip_file, top_file) #FIXME

        stage = "minimisation"
        mdp_dict = mdp2dict(get_data_filename("{}.mdp".format(stage)))
        if eval("{}_mdp".format(stage)) is not None:
            mdp_dict.update(eval("{}_mdp".format(stage)))
        next_gro_file = "{}/{}.gro".format(tmp_dir, stage)

        # grompp_mdrun(input_gro_path = gro_file,
        #     input_ndx_path = index_file,
        #     input_top_zip_path= topzip_file,
        #     # input_mdp_path = "{}.mdp".format(stage),
        #     output_trr_path = "{}/{}.trr".format(tmp_dir, stage),
        #     output_gro_path = next_gro_file,
        #     output_edr_path = "{}/{}.edr".format(tmp_dir, stage),
        #     output_log_path = "{}/{}.log".format(tmp_dir, stage),
        #     output_xtc_path = "{}/{}.xtc".format(tmp_dir, stage),
        #     num_threads_omp = num_threads,
        #     properties = {
        #         "mdp" : mdp_dict
        #         }
        #     )
        grompp(input_gro_path = gro_file,
            input_ndx_path = index_file,
            input_top_zip_path= topzip_file,
            output_tpr_path = "{}/{}.tpr".format(tmp_dir, stage),
            # input_mdp_path = "{}.mdp".format(stage),
            properties = {
                "mdp" : mdp_dict
                }
            )
        mdrun(
            input_tpr_path = "{}/{}.tpr".format(tmp_dir, stage),
            output_trr_path = "{}/{}.trr".format(tmp_dir, stage),
            output_gro_path = next_gro_file,
            output_edr_path = "{}/{}.edr".format(tmp_dir, stage),
            output_log_path = "{}/{}.log".format(tmp_dir, stage),
            output_xtc_path = "{}/{}.xtc".format(tmp_dir, stage),
            num_threads_omp = 1 #XXX seems for minimisation speed is very slow when multiple threads are used, especially on cluster. Maybe need better handle
        )
        gro_file = next_gro_file

        ###################################################333


        stage = "equilibration"
        mdp_dict = mdp2dict(get_data_filename("{}.mdp".format(stage)))
        if eval("{}_mdp".format(stage)) is not None:
            mdp_dict.update(eval("{}_mdp".format(stage)))
        next_gro_file = "{}/{}.gro".format(tmp_dir, stage)
        grompp(input_gro_path = gro_file,
            input_ndx_path = index_file,
            input_top_zip_path= topzip_file,
            output_tpr_path = "{}/{}.tpr".format(tmp_dir, stage),
            # input_mdp_path = "{}.mdp".format(stage),
            properties = {
                "mdp" : mdp_dict
                }
            )
        mdrun(
            input_tpr_path = "{}/{}.tpr".format(tmp_dir, stage),
            output_trr_path = "{}/{}.trr".format(tmp_dir, stage),
            output_gro_path = next_gro_file,
            output_edr_path = "{}/{}.edr".format(tmp_dir, stage),
            output_log_path = "{}/{}.log".format(tmp_dir, stage),
            output_xtc_path = "{}/{}.xtc".format(tmp_dir, stage),
            num_threads_omp = num_threads
        )
        gro_file = next_gro_file

        ######################################################3

        stage = "production"
        mdp_dict = mdp2dict(get_data_filename("{}.mdp".format(stage)))
        if eval("{}_mdp".format(stage)) is not None:
            mdp_dict.update(eval("{}_mdp".format(stage)))
        next_gro_file = "{}/{}.gro".format(tmp_dir, stage)
        grompp(input_gro_path = gro_file,
            input_ndx_path = index_file,
            input_top_zip_path= topzip_file,
            output_tpr_path = "{}/{}.tpr".format(tmp_dir, stage),
            # input_mdp_path = "{}.mdp".format(stage),
            properties = {
                "mdp" : mdp_dict
                }
            )
        mdrun(
            input_tpr_path = "{}/{}.tpr".format(tmp_dir, stage),
            output_trr_path = "{}/{}.trr".format(tmp_dir, stage),
            output_gro_path = next_gro_file,
            output_edr_path = "{}/{}.edr".format(tmp_dir, stage),
            output_log_path = "{}/{}.log".format(tmp_dir, stage),
            output_xtc_path = xtc_file,
            num_threads_omp = num_threads
        )
        # gro_file = next_gro_file

        if debug is not True and tmp_dir is not None:
            import shutil
            shutil.rmtree(tmp_dir)
        return os.path.abspath(xtc_file)

    run = via_openmm

class SolutionSimulator(BaseSimulator):
    """
    Perform solution simulation, namely one copy of solute in water box. Currently identical to BaseSimulator

    Parameters
    -----------
    equil_steps : int
        number of steps during equilibraion, default 50,000 steps, i.e. 100 ps
    """
    equil_steps = 50000  #100 ps

class LiquidSimulator(BaseSimulator):
    """
    Perform liquid simulation, namely multiple copy of the same molecule.

    Parameters
    -----------
    equil_steps : int
        number of steps during equilibraion, default 500,000 steps, i.e. 1 ns
    """
    equil_steps = 500000  #1 ns
