from simtk import unit
from simtk.openmm import app

from simtk.openmm import *
from simtk.openmm.app import *
from mdtraj.reporters import HDF5Reporter


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
            print(cls.equil_steps, " steps")
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
    def via_gromacs(cls):
        """
        Simulation via GROMACS will be added in the future.
        """
        raise NotImplementedError

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
