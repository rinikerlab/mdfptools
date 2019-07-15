from simtk import unit
from simtk.openmm import app


from simtk.openmm import *
from simtk.openmm.app import *
from mdtraj.reporters import HDF5Reporter




class BaseSimulator():
    def via_openmm(parmed_obj, name, platform = "CUDA", num_steps = 500000, write_out_freq = 5000): #2ps timestep thus 1 ns simulation
        platform = Platform.getPlatformByName(platform)
        pmd = parmed_obj
        path = './{}.h5'.format(name)

        system = pmd.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer, constraints=app.AllBonds)

        thermostat = AndersenThermostat(298.15 * unit.kelvin, 1/unit.picosecond)
        system.addForce(thermostat)
        barostat = MonteCarloBarostat(1.013 * unit.bar, 298.15 * unit.kelvin)
        system.addForce(barostat)
        integrator = VerletIntegrator(0.002 * unit.picoseconds)
        simulation = Simulation(pmd.topology, system, integrator, platform)

        simulation.context.setPeriodicBoxVectors(*pmd.box_vectors)
        simulation.context.setPositions(pmd.positions)
        simulation.minimizeEnergy()

        #Eq
        # simulation.reporters.append(StateDataReporter("./" + hash_code + ".dat", 10000, step=True, volume = True, temperature = True))
        simulation.step(50000)

        state = simulation.context.getState(getPositions = True, getVelocities = True)
        pmd.positions, pmd.velocities, pmd.box_vectors = state.getPositions(),state.getVelocities(), state.getPeriodicBoxVectors()

        #Production
        del system
        del simulation

        system = pmd.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer, constraints=app.AllBonds)

        thermostat = AndersenThermostat(298.15 * unit.kelvin, 1/unit.picosecond)
        system.addForce(thermostat)
        #barostat = MonteCarloBarostat(1.013 * unit.bar, 298.15 * unit.kelvin)
        #system.addForce(barostat)
        integrator = VerletIntegrator(0.002 * unit.picoseconds)
        simulation = Simulation(pmd.topology, system, integrator, platform)
        simulation.context.setPeriodicBoxVectors(*pmd.box_vectors)
        simulation.context.setPositions(pmd.positions)
        # simulation.reporters.append(StateDataReporter("./" + hash_code + ".dat", 5000, step=True,potentialEnergy=True, temperature=True))
        simulation.reporters.append(HDF5Reporter(path, write_out_freq))
        simulation.step(num_steps)

        return path

    run = via_openmm

class SolutionSimulator(BaseSimulator):
    pass

class LiquidSimulator(BaseSimulator):
    pass
