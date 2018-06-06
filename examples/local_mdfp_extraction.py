import pandas as pd
import pickle
import multiprocessing
from MDFP_Composer import *

#multiplrocessing, still too slow for ~hundreds of mols
df = pd.DataFrame.from_csv("/home/shuwang/Documents/Modelling/MDFP/Codes/vapour_pressure/crc_handbook_subset.tsv", sep = "\t", )

def func(row_number, path = "/home/shuwang/Documents/Modelling/MDFP/Codes/vapour_pressure/crc_handbook/corrupted/"):
    id = df["ref_id"][row_number]
    print(id, " started.")
    try:
        traj = md.load(path + id + ".h5")
        parm = pickle.load(open(path + id + ".pickle", "rb"))
        result = mdfp_extraction(df["canonical_smiles"][row_number], traj, parm)
        pickle.dump(result, open(path + "mdfp_" + id + ".pickle", "wb"))
    except:
        print(id, " failed")
        return
    print(id, " completed.")

def mdfp_extraction(smiles, traj, parm):
    result = MDFP_Composer(smiles, traj, parm) #TODO
    return result

# func(1)
cores = multiprocessing.cpu_count()
with multiprocessing.Pool(processes=cores) as pool:
    # pool.starmap(func, list(range(len(df))))
    pool.map(func, range(len(df)))
