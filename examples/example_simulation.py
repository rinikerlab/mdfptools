from simtk import unit #Unit handling for OpenMM
from simtk.openmm import app
from simtk.openmm import *
from simtk.openmm.app import *
# from simtk.openmm.app import PDBReporter
import mdtraj as md
from mdtraj.reporters import HDF5Reporter
from mdfptools.Parameteriser import *
from mdfptools.Composer import MDFPComposer

def simulate(pmd, hash_code):
    print("Simulating...")
    platform = Platform.getPlatformByName('CUDA')
    path = './{}.h5'.format(hash_code)

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
    # simulation.reporters.append(StateDataReporter("./" + i + ".dat", 10000, step=True, volume = True))
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
    simulation.reporters.append(HDF5Reporter(path, 500))
    simulation.step(5000 * 500)

    return path
smiles = "[O]=[C]1-[NH]-[C](=[O])-[c]2:[cH]:[cH]:[cH]:[cH]:[c]:2-1"
# pmd = SolutionParameteriser.run(smiles)
pmd = VaccumParameteriser.run(smiles)
simulate(pmd, "tmp")
