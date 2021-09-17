import numpy as np
from mdfptools.Composer import Solution42BitsComposer, WaterComposer, LiquidComposer
import mdtraj as md
import parmed

traj = md.load("./data/water_gmx_example_1.xtc", top = "./data/water_gmx_example_1.gro")
print(traj)
pd = parmed.load_file("./data/water_gmx_example_1.top")
# pd.
out = Solution42BitsComposer.run(traj, pd, smiles = "CCCCC1C2(CCN(CC2)C3(CCN(CC3)C(=O)C4=C(C=CC=C4C)C)C)OC(=O)N1CC5CCCCC5", xvg_file_path = "./data/water_gmx_example_1.xvg" )

# print(out)

x = {
 'MW': 538.40031894409,
 'HA_count': 39,
 'RB_count': 8,
 'N_count': 3,
 'O_count': 3,
 'F_count': 0,
 'P_count': 0,
 'S_count': 0,
 'Cl_count': 0,
 'Br_count': 0,
 'I_count': 0,
 'HBD_count': 1,
 'HBA_count': 5,
 '2d_shape': 6.4040808072224715,
 '2d_psa': 0.5429,
 'is_zwit': 0,
 'intra_crf_av_wat': -362.0758371761648,
 'intra_crf_std_wat': 4.52381883701882,
 'intra_crf_med_wat': -362.216736,
 'intra_lj_av_wat': 67.78596650349931,
 'intra_lj_std_wat': 10.158554294182972,
 'intra_lj_med_wat': 67.22666999999998,
 'total_crf_av_wat': -655.1365829774045,
 'total_crf_std_wat': 28.993538457079023,
 'total_crf_med_wat': -655.15396,
 'total_lj_av_wat': -122.93220517016599,
 'total_lj_std_wat': 17.238890131060966,
 'total_lj_med_wat': -123.475686,
 'intra_ene_av_wat': -294.2898706726655,
 'intra_ene_std_wat': 10.655169132970629,
 'intra_ene_med_wat': -294.86599299999995,
 'total_ene_av_wat': -778.0687881475706,
 'total_ene_std_wat': 30.144125750237034,
 'total_ene_med_wat': -778.144234,
 'wat_rgyr_av': 0.5777995780914261,
 'wat_rgyr_std': 0.011999500888837476,
 'wat_rgyr_med': 0.5772215954495133,
 'wat_sasa_av': 8.873008,
 'wat_sasa_std': 0.09212954,
 'wat_sasa_med': 8.88922,
 '3d_psa_av': 0.5771285802529628,
 '3d_psa_sd': 0.011117498534355709,
 '3d_psa_med': 0.5779158782958984}

x = [x[i] for i in x]
print(out)
import pickle
# pickle.dump(out, (open("./data/fp42_water_gmx_example_1.pickle", "wb")))
# print(np.isclose(out.get_mdfp(), x, atol=1e-01))
###################### 
# import pickle
# traj = md.load("./data/water_example_1.h5")
# print(traj)
# pd = pickle.load(open("./data/water_example_1.pickle", "rb"))
# # pd.
# out = WaterComposer.run(traj, pd)


# # print((out.get_mdfp(), pickle.load(open("./data/fp_water_example_1.pickle", "rb")).get_mdfp()))
# # print(np.isclose(out.get_mdfp(), pickle.load(open("./data/fp_water_example_1.pickle", "rb")).get_mdfp()))

# ###################### 
# import pickle
# traj = md.load("./data/liquid_example_1.h5")
# print(traj)
# pd = pickle.load(open("./data/liquid_example_1.pickle", "rb"))
# # pd.
# out = LiquidComposer.run(traj, pd)


# # print((out.get_mdfp(), pickle.load(open("./data/fp_liquid_example_1.pickle", "rb")).get_mdfp()))
# # print(np.isclose(out.get_mdfp(), pickle.load(open("./data/fp_liquid_example_1.pickle", "rb")).get_mdfp()))