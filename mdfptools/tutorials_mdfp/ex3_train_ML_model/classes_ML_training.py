import matplotlib as mpl
mpl.use('Agg')

import os
import pandas as pd
import pickle
import numpy as np
import glob
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import text
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import SimDivFilters
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from PIL import Image
from pandas.plotting import table
from sklearn.ensemble import VotingClassifier
from sklearn import neighbors
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import  KNeighborsClassifier
from pandas.plotting import table
from rdkit.Chem import rdFingerprintGenerator
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator


class DataPrep:

    def __init__(self):
        pass

    @classmethod
    def add_ECFP4(self, df_dataset, smiles_column = "smiles", nBits = 2048, radius = 2, morgan_column = "ECFP4"):
        """
        Compute Morgan fingerprints (default radius = 2 and nBits = 2048) from SMILES using RDKit.   
        A dataframe has to be provided as input and the SMILES have to be listed in the smiles_column (default smiles_column = "smiles").
        The Morgan fingerprints are returned in the dataframe in the additional column morgan_column (default morgan_column = "ECFP4").

        Parameters
        ----------
        df_dataset: df
            dataframe of the dataset. It must contain a smiles_column to be able to generate Morgan fingerprints
        smiles_column: str, optional
            name of the column containing the SMILES (Default = "smiles")
        nBits: int, optional
            number of bits of the fingerprint (Default = 2048)
        radius: int, optional
            radius of the fingerprint (Default = 2)
        morgan_column: str, optional
            name of the column in which the Morgan fingerprints are returned (Default = "ECFP4")

        Returns
        ----------
        df_dataset: df
            input dataframe with an additional column containing the Morgand fingerprints 
        """

        if smiles_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing SMILES is required to calculate Morgan fingerprints".format(smiles_column))
            return(df_dataset) 

        df_dataset[smiles_column] = df_dataset[smiles_column].astype(str)
        ms = [Chem.MolFromSmiles(x) for x in df_dataset[smiles_column]]
        ECFP4_obj = [AllChem.GetMorganFingerprintAsBitVect(x,radius, nBits=nBits) for x in ms]

        ECFP4 = []
        for fp in ECFP4_obj:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            ECFP4.append(arr)
        
        df_dataset['ECFP4'] = ECFP4
        return(df_dataset)
        

    @classmethod
    def add_RDKitFP(self, df_dataset, smiles_column = "smiles", fpSize = 2048, maxPath = 5, **kwargs):
        """
        Compute RDKit topological fingerprints (default maxPath = 5 and nBits = 2048) from SMILES using RDKit.   
        A dataframe has to be provided as input and the SMILES have to be listed in the smiles_column (default smiles_column = "smiles").
        The RDKit topological fingerprints are returned in the original dataframe in the additional column "RDKitFP".

        Parameters
        ----------
        df_dataset: df
            dataframe of the dataset. It must contain a smiles_column to be able to generate topological RDKit fingerprints
        smiles_column: str, optional
            name of the column containing the SMILES (Default is "smiles")
        fpSize: int, optional
            number of bits of the fingerprint (Default is 2048)
        maxPath: int, optional
            maximum number of bonds to include in the subgraphs (Default is 5, different from RDKit default of 7)
        kwargs:
            additional arguments of the rdFingerprintGenerator.GetRDKitFPGenerator function. See RDKit documentation. The Default arguments of RDKit are set by Default.

        Returns
        ----------
        df_dataset: df
            input dataframe with the additional column "RDKitFP" containing the topological RDKit fingerprints 
        """

        if smiles_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing SMILES is required to calculate topological RDKit fingerprints".format(smiles_column))
            return(df_dataset) 

        df_dataset[smiles_column] = df_dataset[smiles_column].astype(str)
        ms = [Chem.MolFromSmiles(x) for x in df_dataset[smiles_column]]
        generator = Chem.rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=maxPath, fpSize=fpSize, **kwargs)
        fps = [generator.GetFingerprint(x) for x in ms]
        df_dataset['RDKitFP'] = fps
        return(df_dataset)


    @classmethod
    def add_RDKit2D(self, df_dataset, smiles_column = "smiles"):
        """
        Compute the property-based fingerprint "RDKit2D" from SMILES using the Descriptastorus package (https://github.com/bp-kelley/descriptastorus).   
        "RDKit2D" contains a collection of 200 molecular topological properties and 2D atom and fragment counts.
        To check the list of properties use the function DataPrep.get_RDKit2D_colnames() 
        A dataframe has to be provided as input and the SMILES have to be listed in the smiles_column (default smiles_column = "smiles").
        The property-based fingerprints are returned in the original dataframe in the additional column "RDKit2D".

        Parameters
        ----------
        df_dataset: df
            dataframe of the dataset. It must contain a smiles_column to be able to generate RDKit2D fingerprints
        smiles_column: str, optional
            name of the column containing the SMILES (Default is "smiles")

        Returns
        ----------
        df_dataset: df
            input dataframe with the additional column "RDKit2D" containing the property-based fingerprints 
        """
        if smiles_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing SMILES is required to calculate property-based fingerprints".format(smiles_column))
            return(df_dataset) 
        df_dataset[smiles_column] = df_dataset[smiles_column].astype(str)
        rdkit2d = []
        generator = MakeGenerator(("RDKit2D",))
        for n, smi in enumerate(df_dataset[smiles_column]):
            try:
                data = generator.process(smi)
                if data[0] == True:
                    data.pop(0)
                if data[0] == False:
                    data.pop(0)
                rdkit2d.append(data)
            except:
                rdkit2d.append([0]*200)
                print("Error: RDKit2D not generated for {}".format(df_dataset['cmpd_name'][n]))

        df_dataset['RDKit2D'] = rdkit2d
        return(df_dataset)


    @classmethod
    def add_MDFP_RDKit2D(self, df_dataset, smiles_column = "smiles"):
        """
        Add to the dataframe, a column named "MDFP_RDKit2D" containing the hybrid MDFP_RDKit2D descriptor. 
        The property-based fingerprint "RDKit2D" is computed from SMILES using the Descriptastorus package (https://github.com/bp-kelley/descriptastorus).   
        See the description of function DataPrep.add_RDKit2D(). The MDFP terms can be printed typing in the python shell: MDFP_terms.mdfp
        A dataframe has to be provided as input and the SMILES have to be listed in the smiles_column (default smiles_column = "smiles").

        Parameters
        ----------
        df_dataset: df
            dataframe of the dataset. It must contain a smiles_column to be able to generate RDKit2D fingerprints
        smiles_column: str, optional
            name of the column containing the SMILES (Default is "smiles")

        Returns
        ----------
        df_dataset: df
            input dataframe with the additional column "MDFP_RDKit2D" containing the hybrid MDFP_RDKit2D fingerprints 
        """
        try:
            if "RDKit2D" not in list(df_dataset): 
                df_dataset = self.add_RDKit2D(df_dataset, smiles_column = smiles_column)

            tmp = pd.concat([df_dataset[MDFP_terms.mdfp], pd.DataFrame(df_dataset['RDKit2D'].tolist(), index=df_dataset.index)], axis =1)
            df_dataset['MDFP_RDKit2D'] = tmp.values.tolist()

        except:
            print('MDFP_RDKit2D could not be generated')

        return(df_dataset)


    @classmethod
    def add_ECFP4_RDKit2D(self, df_dataset, smiles_column = "smiles"):
        """
        Add to the dataframe, a column named "ECFP4_RDKit2D" containing the hybrid ECFP4_RDKit2D descriptor. 
        The property-based fingerprint "RDKit2D" is computed from SMILES using the Descriptastorus package (https://github.com/bp-kelley/descriptastorus).   
        See the description of functions DataPrep.add_RDKit2D() and DataPrep.add_ECFP4()
        A dataframe has to be provided as input and the SMILES have to be listed in the smiles_column (default smiles_column = "smiles").

        Parameters
        ----------
        df_dataset: df
            dataframe of the dataset. It must contain a smiles_column to be able to generate RDKit2D fingerprints
        smiles_column: str, optional
            name of the column containing the SMILES (Default is "smiles")

        Returns
        ----------
        df_dataset: df
            input dataframe with the additional column "ECFP4_RDKit2D" containing the hybrid ECFP4_RDKit2D fingerprints 
        """
        try:
            if "RDKit2D" not in list(df_dataset): 
                df_dataset = self.add_RDKit2D(df_dataset, smiles_column = smiles_column)

            if "ECFP4" not in list(df_dataset):
                df_dataset = self.add_ECFP4(df_dataset, smiles_column = smiles_column)
             
            tmp = pd.concat([pd.DataFrame(df_dataset['ECFP4'].tolist(), index=df_dataset.index), pd.DataFrame(df_dataset['RDKit2D'].tolist(), index=df_dataset.index)], axis =1)
            df_dataset['ECFP4_RDKit2D'] = tmp.values.tolist()

        except:
            print('ECFP4_RDKit2D could not be generated')

        return(df_dataset)


    @classmethod
    def add_ECFP4_MDFP_RDKit2D(self, df_dataset, smiles_column = "smiles"):
        """
        Add to the dataframe, a column named "ECFP4_MDFP_RDKit2D" containing the hybrid ECFP4_MDFP_RDKit2D descriptor. 
        The property-based fingerprint "RDKit2D" is computed from SMILES using the Descriptastorus package (https://github.com/bp-kelley/descriptastorus).   
        See the description of functions DataPrep.add_RDKit2D() and DataPrep.add_ECFP4(). The MDFP terms can be printed typing in the python shell: MDFP_terms.mdfp
        A dataframe has to be provided as input and the SMILES have to be listed in the smiles_column (default smiles_column = "smiles").

        Parameters
        ----------
        df_dataset: df
            dataframe of the dataset. It must contain a smiles_column to be able to generate RDKit2D fingerprints
        smiles_column: str, optional
            name of the column containing the SMILES (Default is "smiles")

        Returns
        ----------
        df_dataset: df
            input dataframe with the additional column "ECFP4_MDFP_RDKit2D" containing the hybrid ECFP4_MDFP_RDKit2D fingerprints 
        """
        try:
            if "RDKit2D" not in list(df_dataset): 
                df_dataset = self.add_RDKit2D(df_dataset, smiles_column = smiles_column)

            if "ECFP4" not in list(df_dataset):
                df_dataset = self.add_ECFP4(df_dataset, smiles_column = smiles_column)
             
            tmp = pd.concat([df_dataset[MDFP_terms.mdfp], pd.DataFrame(df_dataset['ECFP4'].tolist(), index=df_dataset.index), pd.DataFrame(df_dataset['RDKit2D'].tolist(), index=df_dataset.index)], axis =1)
            df_dataset['ECFP4_MDFP_RDKit2D'] = tmp.values.tolist()

        except:
            print('ECFP4_MDFP_RDKit2D could not be generated')

        return(df_dataset)



    @classmethod
    def get_RDKit2D_colnames(self):
        """ Print terms composing the "RDKit2D" fingerprint """
        generator = MakeGenerator(("RDKit2D",))
        colnames = [col[0] for col in generator.GetColumns()]
        colnames.pop(0)
        return(colnames)


    @classmethod
    def standard_scaler_train_test(self, df_training_set, df_test_set, features_list = None):
        """
        Standardize features by removing the mean and scaling to unit variance. 
        It uses the StandardScaler function of scikit-learn. See scikit-learn documentation.
        It takes as input the dataframes of the training and test sets and it returns the standardized datasets.
        Only the features specified in the features_list are standardized.

        Parameters
        ----------
        df_training_set: df
            dataframe of the training set
        df_test_set: df
            dataframe of the test set
        features_list: list
            list of features to standardize, i.e. column names. By default, all features and, eventually, descriptors (e.g. RDKit2D) listed in the dataset are standardized. 

        Returns
        ----------
        df_training_set: df
            dataframe of the standardized training set
        df_test_set: df
            dataframe of the standardized test set
        """

        if features_list == None:
            features_list = MDFP_terms.scalable_features

        df_training_set.reset_index(inplace = True, drop = True)
        df_test_set.reset_index(inplace = True, drop = True)
        all_colnames_train = list(df_training_set)
        all_colnames_test = list(df_test_set)
        scalable_columns_tmp = features_list
        scalable_columns = list(set(all_colnames_train) & set(scalable_columns_tmp))
        not_scalable_colums_train = list(set(all_colnames_train) ^ set(scalable_columns))
        not_scalable_colums_test = list(set(all_colnames_test) ^ set(scalable_columns))
        scaler = StandardScaler()
        if 'RDKit2D' in scalable_columns: 
            scalable_columns.remove('RDKit2D')
            training_scaled = pd.DataFrame(scaler.fit_transform(df_training_set[scalable_columns]), columns = scalable_columns)
            test_scaled = pd.DataFrame(scaler.fit_transform(df_test_set[scalable_columns]), columns = scalable_columns)
            rdkit2d_training_scaled = np.array((scaler.fit_transform(df_training_set['RDKit2D'].tolist())))
            rdkit2d_test_scaled = np.array((scaler.fit_transform(df_test_set['RDKit2D'].tolist())))
            training_scaled['RDKit2D'] = rdkit2d_training_scaled.tolist()
            test_scaled['RDKit2D'] = rdkit2d_test_scaled.tolist()
            df_training_new = pd.concat([training_scaled, df_training_set[not_scalable_colums_train]], axis = 1 )
            df_test_new = pd.concat([test_scaled, df_test_set[not_scalable_colums_test]], axis = 1 )
            df_training_new.reset_index(inplace = True, drop = True)
            df_test_new.reset_index(inplace = True, drop = True)
        else:
            training_scaled = pd.DataFrame(scaler.fit_transform(df_training_set[scalable_columns]), columns = scalable_columns)
            test_scaled = pd.DataFrame(scaler.fit_transform(df_test_set[scalable_columns]), columns = scalable_columns)
            df_training_new = pd.concat([training_scaled, df_training_set[not_scalable_colums_train]], axis = 1 )
            df_test_new = pd.concat([test_scaled, df_test_set[not_scalable_colums_test]], axis = 1 )
            df_training_new.reset_index(inplace = True, drop = True)
            df_test_new.reset_index(inplace = True, drop = True)
        return(df_training_new, df_test_new)

    @classmethod
    def standard_scaler(self, df_dataset, features_list = None):
        """
        Standardize features by removing the mean and scaling to unit variance. 
        It uses the StandardScaler function of scikit-learn. See scikit-learn documentation.
        It takes as input the dataframes of the dataset and it returns the standardized dataset.
        Only the features specified in the features_list are standardized.

        Parameters
        ----------
        df_dataset: df
            dataframe of the dataset
        features_list: list
            list of features to standardize, i.e. column names. By default, all features and, eventually, descriptors (e.g. RDKit2D) listed in the dataset are standardized. 

        Returns
        ----------
        df_dataset: df
            dataframe of the standardized dataset
        """

        if features_list == None:
            features_list = MDFP_terms.scalable_features

        all_colnames_train = list(df_dataset)
        scalable_columns = features_list
        scalable_columns_in_set = list(set(all_colnames_train) & set(scalable_columns))
        not_scalable_colums_train = list(set(all_colnames_train) ^ set(scalable_columns_in_set))
        scaler = StandardScaler()
        if 'RDKit2D' in scalable_columns_in_set: 
            scalable_columns_in_set.remove('RDKit2D')
            training_scaled = pd.DataFrame(scaler.fit_transform(df_dataset[scalable_columns_in_set]), columns = scalable_columns_in_set)
            rdkit2d_scaled = np.array((scaler.fit_transform(df_dataset['RDKit2D'].tolist())))
            training_scaled['RDKit2D'] = rdkit2d_scaled.tolist()
            df_training_new = pd.concat([training_scaled, df_dataset[not_scalable_colums_train]], axis = 1 )
            df_training_new.reset_index(inplace = True, drop = True)
        else:
            training_scaled = pd.DataFrame(scaler.fit_transform(df_dataset[scalable_columns_in_set]), columns = scalable_columns_in_set)
            df_training_new = pd.concat([training_scaled, df_dataset[not_scalable_colums_train]], axis = 1 )
            df_training_new.reset_index(inplace = True, drop = True)

        return(df_training_new)


    @classmethod
    def combine_descriptors_train_test(self, df_training_set, df_test_set, descriptor1, descriptor2): 
        """It combines two descriptors into a unique descriptor in the order descriptor1 + descriptor2
        It takes as an input the dataframes of the training and test sets and it returns the arrays of the hybrid descriptor for the training and test set, respectively.

        Possible Descriptors: 
        MDFP_terms.mdfp, MDFP_terms.mdfp_dipole, MDFP_terms.mdfp_plus, MDFP_terms.mdfp_p2, MDFP_terms.mdfp_p3, MDFP_terms.mdfp_pp, MDFP_terms.mdfp_plus_plus, MDFP_terms.mdfp_p3_plus_plus, MDFP_terms.counts_2d, MDFP_terms.counts_extra, MDFP_terms.counts_all, MDFP_terms.counts_prop_2d, MDFP_terms.dipole_mom, 'ECFP4', 'RDKit2D', 'RDKitFP'

        Parameters:
        -----------
        df_training_set: df
            dataframe of the training set
        df_test_set: df
            dataframe of the test set
        descriptor1: list
            list of terms of descriptor1. See description above.  
        descriptor2: list
            list of terms of descriptor2. See description above.  
        
        Returns
        ----------
        combi_train_array: array
            array of the combined descriptor for the training set
        combi_test_array: array
            array of the combined descriptor for the test set
        """

        #Combine ECFP4 and mdfp
        if isinstance(descriptor1, str) and not isinstance(df_training_set[descriptor1].iloc[0], (int, float)) and len(df_training_set[descriptor1].iloc[0]) > 1:
            d1_train_array = np.vstack(df_training_set[descriptor1])
            d1_test_array = np.vstack(df_test_set[descriptor1])
        else: 
            training_set = df_training_set[descriptor1]
            test_set = df_test_set[descriptor1]
            d1_train_array = np.array(training_set)
            d1_test_array = np.array(test_set)
        
        if isinstance(descriptor2, str) and not isinstance(df_training_set[descriptor2].iloc[0], (int, float)) and len(df_training_set[descriptor2].iloc[0]) > 1:
            d2_train_array = np.vstack(df_training_set[descriptor2])
            d2_test_array = np.vstack(df_test_set[descriptor2])
        else: 
            training_set = df_training_set[descriptor2]
            test_set = df_test_set[descriptor2]
            d2_train_array = np.array(training_set)
            d2_test_array = np.array(test_set)

        combi_train_array = np.column_stack((d1_train_array, d2_train_array))
        combi_test_array = np.column_stack((d1_test_array, d2_test_array))

        return(combi_train_array, combi_test_array)

        
    @classmethod
    def combine_descriptors(self, df_dataset, descriptor1, descriptor2):
        """It combines two descriptors into a unique descriptor in the order descriptor1 + descriptor2
        It takes as an input the dataframes of the dataset and it returns the arrays of the hybrid descriptor for each of the elements of the dataset..

        Possible Descriptors: 
        MDFP_terms.mdfp, MDFP_terms.mdfp_dipole, MDFP_terms.mdfp_plus, MDFP_terms.mdfp_p2, MDFP_terms.mdfp_p3, MDFP_terms.mdfp_pp, MDFP_terms.mdfp_plus_plus, MDFP_terms.mdfp_p3_plus_plus, MDFP_terms.counts_2d, MDFP_terms.counts_extra, MDFP_terms.counts_all, MDFP_terms.counts_prop_2d, MDFP_terms.dipole_mom, 'ECFP4', 'RDKit2D', 'RDKitFP'

        Parameters:
        -----------
        df_dataset: df
            dataframe of the dataset
        descriptor1: list
            list of terms of descriptor1. See description above.  
        descriptor2: list
            list of terms of descriptor2. See description above.  
        
        Returns
        ----------
        combi_dataset_array: array
            array of the combined descriptors for each of the elements of the dataset
        """

        #Combine ECFP4 and mdfp
        if isinstance(descriptor1, str) and not isinstance(df_dataset[descriptor1].iloc[0], (int, float)) and len(df_dataset[descriptor1].iloc[0]) > 1:
            d1_train_array = np.vstack(df_dataset[descriptor1])
        else: 
            training_set = df_dataset[descriptor1]
            d1_train_array = np.array(training_set)
        
        if isinstance(descriptor2, str) and not isinstance(df_dataset[descriptor2].iloc[0], (int, float)) and len(df_dataset[descriptor2].iloc[0]) > 1:
            d2_train_array = np.vstack(df_dataset[descriptor2])
        else: 
            training_set = df_dataset[descriptor2]
            d2_train_array = np.array(training_set)

        combi_dataset_array = np.column_stack((d1_train_array, d2_train_array))

        return(combi_dataset_array)
        

    @classmethod
    def balance_sulfur_help(self, properties_sulfur, Ntest, smiles_column = "smiles", labels_column = "is_sub", random_seed = None):
        """ Help function for balance_sulfur.
        It takes as input the dataframe of the sulfur-containing compounds (properties_sulfur). 
        It selects Ntest substrates and nonsubstrates based on chemical diversity using the MaxMin function of RDKit. 
        """
        if random_seed == None:
            random_seed1 = int(np.random.randint(1000, size=1))
        else:
            random_seed1 = random_seed

        properties_sulfur['smiles'] = properties_sulfur['smiles'].astype(str)

        #pick diversity subset from total dataset
        #properties_sulfur['index_col'] = properties_sulfur.index
        df_subs = properties_sulfur.loc[properties_sulfur['is_sub'] == 1] 
        df_nsubs = properties_sulfur.loc[properties_sulfur['is_sub'] == 0]
        indexes_subs = list(df_subs.index)
        indexes_nsubs = list(df_nsubs.index)
        #indexes_subs = list(df_subs['index_col'])
        #indexes_nsubs = list(df_nsubs['index_col'] )
        mols_subs =  [Chem.MolFromSmiles(m) for m in df_subs['smiles']]
        mols_nsubs = [Chem.MolFromSmiles(m) for m in df_nsubs['smiles']]  
        morgan_fps_subs  = [AllChem.GetMorganFingerprint(x,2) for x in mols_subs]
        morgan_fps_nsubs = [AllChem.GetMorganFingerprint(x,2) for x in mols_nsubs]
        nfps_subs = len(morgan_fps_subs)
        nfps_nsubs = len(morgan_fps_nsubs)
        def distij_subs(i,j,morgan_fps=morgan_fps_subs):
            return 1-DataStructs.DiceSimilarity(morgan_fps[i],morgan_fps[j])
        def distij_nsubs(i,j,morgan_fps=morgan_fps_nsubs):
            return 1-DataStructs.DiceSimilarity(morgan_fps[i],morgan_fps[j])
        picker = SimDivFilters.MaxMinPicker()
        pickIndices_subs = picker.LazyPick(distij_subs, nfps_subs, Ntest, seed=random_seed1)
        pickIndices_nsubs = picker.LazyPick(distij_nsubs, nfps_nsubs, Ntest, seed=random_seed1)
        indexes_sulf_subs = list(pickIndices_subs)
        indexes_sulf_nsubs = list(pickIndices_nsubs)
        indexes_sulf_subs2 = np.take(indexes_subs, indexes_sulf_subs)
        indexes_sulf_nsubs2 = np.take(indexes_nsubs, indexes_sulf_nsubs) 
        df_sulf_subs = properties_sulfur.iloc[indexes_sulf_subs2]
        df_sulf_nsubs = properties_sulfur.iloc[indexes_sulf_nsubs2]
        df_sulfur_balanced = pd.concat([df_sulf_subs, df_sulf_nsubs], axis=0)
        #df_sulfur_balanced = df_sulfur_balanced.drop(columns=['index_col'])
        df_sulfur_balanced.reset_index(inplace = True, drop = True)
        return(df_sulfur_balanced)


    @classmethod
    def balance_sulfur(self, df_dataset, smiles_column = "smiles", labels_column = "is_sub", random_seed = None):
        """
        Balance the number of sulfur containing substrates and nonsubstrates in the dataset. 
        The sulfur containing substrates and nonsubstrates are selected so as to maximize chemical diversity.
        Substrates are compounds with labels_column = 1 while nonsubstrates have labels_column = 0.

        Parameters:
        -----------
        df_dataset: df
            dataframe of features. It must also contain a "smiles" column and a "is_sub" classification column
        smiles_column: str, optional
            name of the column containing the SMILES (Default = "smiles")
        labels_column: str, optional
            name of the column containing the classification labels (Default = "is_sub")
        random_seed: int, optional
            integer Number to use as seed for the LazyPick function for the selection of chemically diverse compounds

        Returns:
        ----------
        df_sulfur_balanced: df
            dataframe containing a balanced number of sulfur containing substrates and nonsubstrates
        is_substrate_sulfur_balanced: list
            list of classification labels
        """

        if 'smiles' not in list(df_dataset):
            print("Error: No column containing smiles or names smiles")

        df_dataset['smiles'] = df_dataset['smiles'].astype(str)

        # select compounds containing sulfur
        df_dataset_sulfur = df_dataset[df_dataset.smiles.str.contains("S")]
        df_dataset_sulfur.reset_index(inplace = True, drop = True)
        # Count substrates and nonsubstrates containing sulfur atoms and take the minimum        
        Nsubs_with_S = len(df_dataset_sulfur.loc[df_dataset_sulfur[labels_column] == 1])
        Nnsubs_with_S = len(df_dataset_sulfur.loc[df_dataset_sulfur[labels_column] == 0])
        Ncmpds_with_S = min(Nsubs_with_S, Nnsubs_with_S)
         
        # select diverse N = Ncmpds_with_S substrates and nonsubstrates containing sulfur and split them in a training and test set based on diversity
        df_sulf = self.balance_sulfur_help(df_dataset_sulfur, Ncmpds_with_S, smiles_column = smiles_column, labels_column = labels_column, random_seed = random_seed)
        
        #remove compounds containing S atoms
        df_no_sulf = df_dataset[~df_dataset.smiles.str.contains("S")]
        df_no_sulf.reset_index(inplace = True, drop = True)
        
        
        # combining sulfur and non-sulfur containing datasets
        df_sulf_balanced = pd.concat([df_sulf, df_no_sulf], axis=0)
        df_sulf_balanced.reset_index(inplace = True, drop = True)
        is_substrate_sulf_balanced = list(df_sulf_balanced['is_sub'])

        return(df_sulf_balanced, is_substrate_sulf_balanced)


class Evaluate:

    def __init__(self):
        pass

    @classmethod
    def vote_y_pred(self, class_pred):
        """Hard Voting - returns the winning label for one compound
        
        Parameters:
        -----------
        class_pred: list
            list of classification predictions for a single compound

        Returns:
        ----------
        class: integer
            class determined with a majority vote. 
        """
        counts = np.bincount(class_pred)
        return(np.argmax(counts))


    @classmethod
    def calc_metrics (self, clf, fp_train, y_true_train, fp_test, y_true_test, decision_threshold = 0.5):  # y_pred = model.predict(fp_test)   y_true = classes_test
        """Returns metrics for classification task. 
        It takes as input the ML model, the training and test sets, the true and the predicted classes.
        It returns a list containing: train score, test score, TP, TN, FP, FN, AUC, precision, selectivity (SE), specificity (SP), Cohen's Kappa (Kappa), f1-score (F1), and Matthew's correlation coefficient (MCC)

        Parameters:
        -----------
        clf: scikit-learn model
            trained classification model 
        fp_train: df or array
            dataframe or array of descriptors of the training set
        y_true_train: list
            list of the true classes of the instances of the training set
        fp_test: df or array
            dataframe or array of the fingerprints of the test set
        y_true_test: list
            list of the true classes of the instances of the test set
        decision_threshold: float, optional
            decision threshold to determine the predicted classes. (Default = 0.5)
            Shifting the decision threshold is a strategy to rebalance the predictions.

        Returns:
        ----------
        output_metrics: list
            list of the output classification metrics as described above
        """

        # extract predictions based on a specified decision threshold
        thresh = decision_threshold
        try:
            probs_test = clf.predict_proba(fp_test)[:,1]
            y_pred = [1 if x>=thresh else 0 for x in probs_test]
        except:
            print("predict_proba is not available. predict() will be used. (if SVC make sure that probability=True)")
            y_pred = clf.predict(fp_test)
        # metrices
        tn, fp, fn, tp = confusion_matrix(y_true_test, y_pred).ravel()
        se = tp/(tp + fn)
        sp = tn/(tn + fp)
        precision = tp/(tp + fp)
        ga = (tp + tn)/(tp + fp + fn + tn)
        kappa = cohen_kappa_score(y_true_test, y_pred)
        mcc = matthews_corrcoef(y_true_test, y_pred)
        f1 = f1_score(y_true_test, y_pred)
        auroc = roc_auc_score(y_true_test, probs_test)
        Train_Score = clf.score(fp_train, y_true_train)
        output_metrics = [Train_Score, ga, tp, tn, fp, fn, auroc, precision, se, sp, kappa, f1, mcc]
        return output_metrics


    @classmethod
    def help_extract_best_kappa(self, df_metrics_abbvie_concat, output_filename):  
        "help function for save_metrics_10fold"
        N_fingerprints = df_metrics_abbvie_concat['fingerprint'].nunique()

        df_metrics_abbvie_concat['diff_decision_threshold'] = abs(df_metrics_abbvie_concat['decision_threshold'] - 0.5)
        by_row_index_abbvie = df_metrics_abbvie_concat.groupby(['fingerprint', 'decision_threshold'])
        by_row_index_abbvie2 = df_metrics_abbvie_concat.drop(columns=['diff_decision_threshold']).groupby(['fingerprint', 'decision_threshold'])
        df_abbvie_gbt_means = by_row_index_abbvie.mean()
        df_abbvie_gbt_std = by_row_index_abbvie2.std()
        df_abbvie_gbt_std.columns = ["Std Train Score", "Std Test Score", "Std TP", "Std TN", "Std FP", "Std FN", "Std AUC", "Std Prec", "Std SE", "Std SP", "Std Kappa", "Std F1", "Std MCC"]
        df_abbvie_gbt_metrics_new = pd.concat([df_abbvie_gbt_means, df_abbvie_gbt_std], axis = 1)
        df_metrics2_threshold = df_abbvie_gbt_metrics_new
        #df_metrics2_threshold['diff_decision_threshold'] = abs(df_metrics2_threshold['decision_threshold'] - 0.5)
        idx = df_metrics2_threshold.groupby(by="fingerprint", sort = False)['Kappa'].transform(max) == df_metrics2_threshold['Kappa']    
        tmp_best = df_metrics2_threshold[idx]
        if tmp_best.shape[0] != N_fingerprints:
            idx = tmp_best.groupby(by="fingerprint", sort = False)['Test Score'].transform(max) == tmp_best['Test Score']
            tmp_best2  = tmp_best[idx]
            if tmp_best2.shape[0] != N_fingerprints:
                idx = tmp_best2.groupby(by="fingerprint", sort = False)['diff_decision_threshold'].transform(min) == tmp_best2['diff_decision_threshold']
                df_metrics2_best_threshold  = tmp_best2[idx]
            else:
                df_metrics2_best_threshold = tmp_best2
        else:
            df_metrics2_best_threshold = tmp_best

        # save metrics for the best kappa
        df_metrics2_best_threshold.to_csv("{}_best_kappa.csv".format(output_filename), index=False)
        df_metrics2_best_threshold.to_pickle("{}_best_kappa.pkl".format(output_filename))
 
        df_abbvie_gbt_metrics_new2 = df_metrics2_best_threshold[['Train Score','Test Score','Std Test Score', 'TP', 'TN', 'FP', 'FN', 'AUC', 'Prec', 'SE', 'SP', 'Kappa', 'F1', 'MCC']]

        return df_abbvie_gbt_metrics_new2


    @classmethod
    def extract_best_kappa(self, df_metrics2_threshold):  
        idx = df_metrics2_threshold.groupby(by="fingerprint", sort = False)['Kappa'].transform(max) == df_metrics2_threshold['Kappa']    
        df_metrics2_best_threshold = df_metrics2_threshold[idx]
        return df_metrics2_best_threshold

    
    @classmethod
    def calc_metrics_ensemble(self, clf, mdfp_test, y_true, y_pred):  # y_pred = model.predict(mdfp_test)   y_true = classes_test
        """Returns metrics for classification task for the bagging IRUS method. 
        It takes as input the ML model, the test set, the true and the predicted classes.
        It returns a list containing: test score, TP, TN, FP, FN, AUC, precision, selectivity (SE), specificity (SP), Cohen's Kappa (Kappa), f1-score (F1), and Matthew's correlation coefficient (MCC)

        Parameters:
        -----------
        clf: scikit-learn model
            trained classification model 
        mdfp_test: df or array
            dataframe or array of the fingerprints of the test set
        y_true: list
            list of the true classes of the instances of the test set
        y_pred: list
            list of the predicted classes of the instances of the test set

        Returns:
        ----------
        metrics: list
            list of the output classification metrics as described above
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        se = tp/(tp + fn)
        sp = tn/(tn + fp)
        precision = tp/(tp + fp)
        ga = (tp + tn)/(tp + fp + fn + tn)
        kappa = cohen_kappa_score(y_true, y_pred) 
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        probs_test = clf.predict_proba(mdfp_test)[:,1]
        auroc = roc_auc_score(y_true, probs_test)
        metrics = [ga, tp, tn, fp, fn, auroc, precision, se, sp, kappa, f1, mcc]
        return metrics



    @classmethod
    def plot_prediction_confidence(self, predictions, df_test_set, decision_threshold = 0.5, output_folder = ".", output_filename = "test", return_mispredicted = False):
        """ 
        Plot prediction probabilities for the test set and the confidence for the mispredicted compounds (false positives and false negatives)

        Parameters:
        -----------
        predictions: list
            list of dictionaries. Returned setting return_pred = True when training the ML models (not available for MLENS)
        df_test_set: df
            dataframe containing the test set. It must contain the columns 'cmpd_name' and 'is_sub'
        decision_threshold: float, optional
            classification threshold on the prediction probability. It should be the same as the one employed to generate the predictions (Default = 0.5)
        output_folder: str, optional
            path to the output folder
        output_filename: str, optional (Default = ".")
            basename for the output plots (Default = "test")
        return_mispredicted: bool, optional
            True to return a dictionary of dataframes containing the name of the compounds, true and predicted labels, and the prediction probability. The dictionary arguments correspond to the fingerprint names

        Returns:
        ----------
        hist_predict_proba_BASENAME_FP.png
            histogram of the prediction probabilities for the test set. BASENAME can be provided as input. FP is the fingerprint, read from the input predictions
        hist_confidence_mispredictions_BASENAME_FP.png
            histogram of the confidence of the wrongly predicted compounds. BASENAME can be provided as input. FP is the fingerprint, read from the input predictions
        dict_mispredicted: dict, optional
            dictionary of dataframes containing the name of the compounds, true and predicted labels, and the prediction probability. The dictionary arguments correspond to the fingerprint names. Returned only if return_mispredicted set to True. 
        """

        dict_mispredicted = {}

        for pred1 in predictions:
            decision_threshold = decision_threshold
            df_pred = pd.DataFrame(pred1)
            fp = df_pred["fingerprint"][0]
            #df_pred2 = df_pred[["y_pred", "pred_proba"]].rename(columns={"y_pred": "y_pred_{}".format(fp), "pred_proba": "pred_proba_{}".format(fp)})
            df_pred2 = df_pred[["y_pred", "pred_proba"]]
            df_pred2['fingerprint'] = fp
            b = pd.concat([df_test_set[['cmpd_name', 'is_sub']], df_pred2], axis =1)
            # plot 1: hist of predicted probabilities
            plt.gcf().clear()
            plt.clf()
            plt.cla()
            plt.close()
            h1 = plt.hist(b['pred_proba'], 50, color = 'steelblue', histtype='stepfilled', edgecolor='black')
            plt.axvline(x=decision_threshold, color = 'black', linestyle='--')
            plt.text(0.2, max(h1[0]) - max(h1[0])/5, s = "nsub", fontsize=12)
            plt.text(0.7, max(h1[0]) - max(h1[0])/5, s = "sub", fontsize=12)
            plt.ylabel('frequency', fontsize=14)
            plt.xlabel('prediction probability', fontsize=14)
            plt.savefig('{}/hist_predict_proba_{}_{}.png'.format(output_folder, output_filename, fp))
            # calculate confidence for false_positives and false_negatives
            b['mispredicted'] = abs(b['is_sub'] - b['pred_proba'])
            df_mispredicted = b.loc[b['mispredicted'] > 0.5]
            false_positives = df_mispredicted.loc[df_mispredicted['is_sub'] == 0].sort_values(by=['mispredicted'])
            false_negatives = df_mispredicted.loc[df_mispredicted['is_sub'] == 1].sort_values(by=['mispredicted'])
            #plot2: confidence of the predictions mispredicted compounds (one could also plot 'pred_proba')
            plt.gcf().clear()
            plt.clf()
            plt.cla()
            plt.close()
            fig, axs = plt.subplots(3, sharex=True, sharey=True)
            fig.suptitle('Confidence (true_label - prediction_probability)')
            h2 = axs[0].hist(false_positives['mispredicted'], 20, color = 'steelblue', histtype='stepfilled', edgecolor='black')
            h3 = axs[1].hist(false_negatives['mispredicted'], 20, color = 'steelblue', histtype='stepfilled', edgecolor='black')
            h4 = axs[2].hist(df_mispredicted['mispredicted'], 20, color = 'steelblue', histtype='stepfilled', edgecolor='black')
            y_text = max([max(h2[0]) - max(h2[0])/8, max(h3[0]) - max(h3[0])/8, max(h4[0]) - max(h4[0])/8])
            axs[0].text(0.82, y_text, s = "FP: {}".format(false_positives.shape[0]), fontsize=12)
            axs[1].text(0.82, y_text, s = "FN: {}".format(false_negatives.shape[0]), fontsize=12)
            axs[2].text(0.82, y_text, s = "Mispredicted: {}".format(df_mispredicted.shape[0]), fontsize=12)
            axs.flat[1].set_ylabel("frequency", fontsize = 14)   #axs.ylabel('frequency', fontsize=14)
            plt.xlabel('confidence', fontsize=14)
            plt.savefig('{}/hist_confidence_mispredictions_{}_{}.png'.format(output_folder, output_filename, fp))
            df_mispredicted = df_mispredicted.rename(columns={"mispredicted": "confidence"})
            dict_mispredicted[fp] = df_mispredicted

        if return_mispredicted:
            return dict_mispredicted


    @classmethod
    def plot_feature_importance(self, clf, feature_names = None, colors = None, fig_outname = None, out_fig_folder = None):
        """
        Plot the feature importances. If the descriptor contains more than 35 features, only the first 35 features are plotted.
        It takes as input the trained classification model and  the list of features. 

        Parameters:
        -----------
        clf: scikit-learn model
            trained classification model 
        feature_names: list, optional
            list of feature names composing the descriptor (Default = None, features are automatically named as f1, f2, ... fN)
        colors: list, optional
            list of colors, one for each of the features (Default = "gray")
        fig_outname: str, optional
            BASENAME of the output plot. if specified the output plot is named feature_importance_BASENAME.png, otherwise it is named feature_importance.png
        out_fig_folder: str, optional
            Path to the output folder where the plots of feature importances are stored (Default = ".")

        Returns:
        ----------
        feature_importance_BASENAME.png
            plot of feature importances 
        """

        if out_fig_folder == None:
            out_fig_folder = "."
        else:
            if not os.path.exists(out_fig_folder):
                os.makedirs(out_fig_folder)

        if fig_outname == None:
            fig_outname = '{}/feature_importance.png'.format(out_fig_folder)
        else:
            fig_outname = '{}/feature_importance_{}.png'.format(out_fig_folder, fig_outname)

        # Extract feature importances from models
        try:
            feature_importance = clf.feature_importances_
        except:
            pass
        # for SVM
        try:
            feature_importance = clf.coef_[0]
        except:
            pass

        if 'feature_importance' not in locals():
            print("Error: the plot of the feature importances could not be generated")
            return

        if feature_names == None:
            terms_to_test = ['f{}'.format(s) for s in range(1, len(feature_importance)+1)]
        else:
            terms_to_test = feature_names

        if colors == None:
            colors = ['gray'] * len(feature_importance)

        colors2 = np.array(colors)
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        sorted_idx = sorted_idx[::-1]
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.gcf().clear()
        plt.clf()
        plt.cla()
        plt.close()
        plt.bar(pos, feature_importance[sorted_idx], align='center', color = colors2[sorted_idx])
        plt.xticks(pos, np.array(terms_to_test)[sorted_idx], rotation='vertical')
        plt.subplots_adjust(bottom=0.4)
        plt.ylabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(fig_outname)
        return
   
 
    @classmethod
    def plot_feature_importance_top35(self, clf, feature_names = None, colors = None, fig_outname = None, out_fig_folder = None):
        """
        Plot the feature importances. If the descriptor contains more than 35 features, only the first 35 features are plotted.
        It takes as input the trained classification model and  the list of features. 

        Parameters:
        -----------
        clf: scikit-learn model
            trained classification model 
        feature_names: list, optional
            list of feature names composing the descriptor (Default = None, features are automatically named as f1, f2, ... fN)
        colors: list, optional
            list of colors, one for each of the features (Default = "gray")
        fig_outname: str, optional
            BASENAME of the output plot. if specified the output plot is named feature_importance_top35_BASENAME.png, otherwise it is named feature_importance_top35.png
        out_fig_folder: str, optional
            Path to the output folder where the plots of feature importances are stored (Default = ".")

        Returns:
        ----------
        feature_importance_top35_BASENAME.png
            plot of feature importances 
        """

        if out_fig_folder == None:
            out_fig_folder = "."
        else:
            if not os.path.exists(out_fig_folder):
                os.makedirs(out_fig_folder)

        if fig_outname == None:
            fig_outname = '{}/feature_importance_top35.png'.format(out_fig_folder)
        else:
            fig_outname = '{}/feature_importance_top35_{}.png'.format(out_fig_folder, fig_outname)

        # Extract feature importances from models
        try:
            feature_importance = clf.feature_importances_
        except:
            pass
        # for SVM
        try:
            feature_importance = clf.coef_[0]
        except:
            pass

        if 'feature_importance' not in locals():
            print("Error: the plot of the feature importances could not be generated")
            return

        if feature_names == None:
            terms_to_test = ['f{}'.format(s) for s in range(1, len(feature_importance)+1)]
        else:
            terms_to_test = feature_names

        if colors == None:
            colors = ['gray'] * len(feature_importance)

        colors2 = np.array(colors)
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        sorted_idx2 = sorted_idx[::-1]
        top35_features = sorted_idx2[0:35]
        pos = np.arange(top35_features.shape[0]) + .5
        plt.gcf().clear()
        plt.clf()
        plt.cla()
        plt.close()
        plt.bar(pos, feature_importance[top35_features], align='center', color = colors2[top35_features])
        plt.xticks(pos, np.array(terms_to_test)[top35_features], rotation='vertical')
        plt.subplots_adjust(bottom=0.4)
        plt.ylabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(fig_outname)
        return
   
 
    @classmethod
    def extract_important_substructures_top20(self, clf, df_training, smiles_column = "smiles", fig_outname = None, out_fig_folder = None):
        """
        Plot the feature importances for structural Morgan descriptors. Only the top 20 substructures are plotted.
        It takes as input the trained classification model, the training dataset containing a column of SMILES.

        Parameters:
        -----------
        clf: scikit-learn model
            trained classification model 
        df_training: df
            dataframe of the training set containing both a column of SMILES 

        N_top: int, optional
            number of top substructures to plot
        smiles_column: str, optional
            name of the column containing the SMILES (Default = "smiles")
        fig_outname: str, optional
            BASENAME of the output plot. if specified the output plot is named substructure_importance_top20_BASENAME.png, otherwise it is named substructure_importance_top20.png
        out_fig_folder: str, optional
            Path to the output folder where the plots of feature importances are stored (Default = ".")

        Returns:
        ----------
        substructure_importance_top20_BASENAME.png
            plot of feature importances 
            The grid for the plot is currently set as x*y = 5*4
        """

        if smiles_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing SMILES is required to plot substructure importances".format(smiles_column))
            return

        if morgan_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing Morgan fingerprints is required to plot substructure importances".format(smiles_column))
            return

        if out_fig_folder == None:
            out_fig_folder = "."
        else:
            if not os.path.exists(out_fig_folder):
                os.makedirs(out_fig_folder)

        if fig_outname == None:
            fig_outname = '{}/substructure_importance_top20.png'.format(out_fig_folder)
        else:
            fig_outname = '{}/substructure_importance_top20_{}.png'.format(out_fig_folder, fig_outname)

        # Extract feature importances from models
        try:
            feature_importance = clf.feature_importances_
        except:
            pass
        # for SVM
        try:
            feature_importance = clf.coef_[0]
        except:
            pass

        if 'feature_importance' not in locals():
            print("Error: the plot of the feature importances could not be generated")
            return

        N_top = 20
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        sorted_idx2 = sorted_idx[::-1]
        fragments_top20 = sorted_idx2[0:N_top]

        # extract 20 molecules with the top 20 fragments
        ECFP4_train = Read_data.get_ECFP4(df_training_set)
        mol_with_top20_bits = []
        for bit1 in fragments_top20:
            for i, fp1 in enumerate(ECFP4_train):
                    if fp1[bit1]!=0:
                        mol_with_top20_bits.append(i)
                        break

        # plot each of the fragments and obtain a list of png images
        df_training[smiles_column] = df_training[smiles_column].astype(str)
        mols = [Chem.MolFromSmiles(m) for m in df_training[smiles_column]]
        
        plot_mols = []
        for i in range(N_top):
            mol_index = mol_with_top20_bits[i]
            m = mols[mol_index]
            frag_bit = fragments_top20[i]
            bi = {}
            bla = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo=bi)
            plot_mols.append(Draw.DrawMorganBit(m, frag_bit, bi))

        # Plot all images in a unique file
        # set space between images
        x_offset = 5 
        y_offset = 5 
        widths, heights = zip(*(i.size for i in plot_mols))
        widths = max(widths) + x_offset*4
        heights = max(heights) + y_offset*5
        total_width = 4 * widths
        total_height = 5 * heights

        #chose image format between normal and landscape. Default 'normal'
        img_format = "landscape"
        
        if img_format == "normal":
            new_im = Image.new('RGBA', (total_width, total_height), (0,0,0,0))
        elif img_format == "landscape":
            new_im = Image.new('RGBA', (total_height, total_width), (0,0,0,0))
        
        for i,im in enumerate(plot_mols):
          x_pos = (i-int(i/4)*4)
          y_pos = int(i/4)
          ### normal ###
          if img_format == "normal":
            new_im.paste(im, (widths*x_pos,heights*y_pos))
          ### landscape ###
          elif img_format == "landscape":
            new_im.paste(im, (heights*y_pos, widths*x_pos))

        new_im.save(fig_outname)
        return

    @classmethod
    def plot_metrics_classification_table_mmp_10fold(self, df_metrics2, plot_name_metrics):
            if plt:
                plt.gcf().clear()
                plt.clf()
                plt.cla()
                plt.close()
            fig, ax = plt.subplots(figsize=(18, 7.5)) # set size frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            col_wid_scores = [0.06] * (df_metrics2.shape[1]-2)
            col_widths = [0.12,0.12, 0.07, 0.07, 0.07] + [0.04] * 6 + col_wid_scores
            tabla = table(ax, df_metrics2, loc='upper right', colWidths=col_widths)  # where df_metrics2 is your data frame
            tabla.auto_set_font_size(False) # Activate set fontsize manually
            tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
            tabla.scale(1.2, 1.2) # change size table
            plt.savefig('{}.png'.format(plot_name_metrics), transparent=True, bbox_inches='tight')
            plt.gcf().clear()
            plt.clf()
            plt.cla()
            plt.close()

    @classmethod
    def plot_metrics_classification_table(self, df_metrics2, plot_name_metrics):
            if plt:
                plt.gcf().clear()
                plt.clf()
                plt.cla()
                plt.close()
            fig, ax = plt.subplots(figsize=(17, 7.5)) # set size frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            col_wid_scores = [0.05] * (df_metrics2.shape[1]-2)
            col_widths = [0.17,0.1, 0.1] + col_wid_scores
            tabla = table(ax, df_metrics2, loc='upper right', colWidths=col_widths)  # where df_metrics2 is your data frame
            tabla.auto_set_font_size(False) # Activate set fontsize manually
            tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
            tabla.scale(1.2, 1.2) # change size table
            plt.savefig('metrics_{}.png'.format(plot_name_metrics), transparent=True, bbox_inches='tight')
            plt.gcf().clear()
            plt.clf()
            plt.cla()
            plt.close()

    @classmethod
    def plot_metrics_classification_table_rus(self, df_metrics2, plot_name_metrics):
            if plt:
                plt.gcf().clear()
                plt.clf()
                plt.cla()
                plt.close()
            fig, ax = plt.subplots(figsize=(22, 7.5)) # set size frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            col_wid_scores = [0.05] * (df_metrics2.shape[1]-2)
            col_widths = [0.17, 0.08, 0.07, 0.08, 0.07, 0.07] + col_wid_scores
            tabla = table(ax, df_metrics2, loc='upper right', colWidths=col_widths)  # where df_metrics2 is your data frame
            tabla.auto_set_font_size(False) # Activate set fontsize manually
            tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
            tabla.scale(1.2, 1.2) # change size table
            plt.savefig('{}.png'.format(plot_name_metrics), transparent=True)
            plt.gcf().clear()
            plt.clf()
            plt.cla()
            plt.close()


    @classmethod
    def plot_metrics_classification_table_10fold(self, df_metrics2, plot_name_metrics):
            if plt:
                plt.gcf().clear()
                plt.clf()
                plt.cla()
                plt.close()
            fig, ax = plt.subplots(figsize=(20, 7.5)) # set size frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            col_wid_scores = [0.05] * (df_metrics2.shape[1]-2)
            col_widths = [0.08, 0.08, 0.07] + col_wid_scores
            tabla = table(ax, df_metrics2, loc='upper right', colWidths=col_widths)  # where df_metrics2 is your data frame
            tabla.auto_set_font_size(False) # Activate set fontsize manually
            tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
            tabla.scale(1.2, 1.2) # change size table
            plt.savefig('{}.png'.format(plot_name_metrics), transparent=True)
            plt.gcf().clear()
            plt.clf()
            plt.cla()
            plt.close()


    @classmethod
    def plot_metrics_classification_table_10fold_rus(self, df_metrics2, plot_name_metrics):
            if plt:
                plt.gcf().clear()
                plt.clf()
                plt.cla()
                plt.close()
            fig, ax = plt.subplots(figsize=(22, 7.5)) # set size frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            col_wid_scores = [0.05] * (df_metrics2.shape[1]-2)
            col_widths = [0.08, 0.08, 0.07, 0.08, 0.07] + col_wid_scores
            tabla = table(ax, df_metrics2, loc='upper right', colWidths=col_widths)  # where df_metrics2 is your data frame
            tabla.auto_set_font_size(False) # Activate set fontsize manually
            tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
            tabla.scale(1.2, 1.2) # change size table
            plt.savefig('{}.png'.format(plot_name_metrics), transparent=True)
            plt.gcf().clear()
            plt.clf()
            plt.cla()
            plt.close()



    @classmethod
    def plot_metrics_classification_table_one_series(self, df_metrics2, plot_name_metrics):
            if plt:
                plt.gcf().clear()
                plt.clf()
                plt.cla()
                plt.close()
            fig, ax = plt.subplots(figsize=(22, 7.5)) # set size frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            col_wid_scores = [0.05] * (df_metrics2.shape[1]-2)
            col_widths = [0.17, 0.07, 0.07, 0.07, 0.07, 0.07] + col_wid_scores
            tabla = table(ax, df_metrics2, loc='upper right', colWidths=col_widths)  # where df_metrics2 is your data frame
            tabla.auto_set_font_size(False) # Activate set fontsize manually
            tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
            tabla.scale(1.2, 1.2) # change size table
            plt.savefig('{}.png'.format(plot_name_metrics), transparent=True)
            plt.gcf().clear()
            plt.clf()
            plt.cla()
            plt.close()


    @classmethod
    def plot_metrics_ensemble_classifiers(self, df_metrics2, plot_name_metrics, Nmethods):
            if plt:
                plt.gcf().clear()
                plt.clf()
                plt.cla()
                plt.close()
            fig, ax = plt.subplots(figsize=(17, 3.7)) # set size frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis
            ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
            col_widths = [0.17] + Nmethods*[0.05] + [0.1, 0.1, 0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
            tabla = table(ax, df_metrics2, loc='upper right', colWidths=col_widths)  # where df_metrics2 is your data frame
            tabla.auto_set_font_size(False) # Activate set fontsize manually
            tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
            tabla.scale(1.2, 1.2) # change size table
            plt.savefig('{}.png'.format(plot_name_metrics), transparent=True)
            plt.gcf().clear()
            plt.clf()
            plt.cla()
            plt.close()


    @classmethod
    def help_save_metrics_10fold_decision_threshold(self, df_metrics_abbvie_concat, output_filename):
        """ Help function for save_metrics_10fold"""
        output_filename2 = "{}_decision_threshold_screen".format(output_filename)
        output_filename3 = "{}_best_kappa".format(output_filename)
        # save metrics for all decision thresholds
        by_row_index_abbvie = df_metrics_abbvie_concat.groupby(['fingerprint', 'decision_threshold'])
        df_abbvie_metrics_all_new2 = self.help_save_metrics_10fold(by_row_index_abbvie, output_filename2)         
        df_abbvie_metrics_all_new3 = self.help_extract_best_kappa(df_metrics_abbvie_concat, output_filename)
        if 'diff_decision_threshold' in list(df_metrics_abbvie_concat): 
            df_metrics_abbvie_concat = df_metrics_abbvie_concat.drop(columns=['diff_decision_threshold'])
        df_abbvie_metrics_all_new3 = df_abbvie_metrics_all_new3.round(2)
        self.plot_metrics_classification_table_10fold(df_abbvie_metrics_all_new3, output_filename3) 
        # save metrics for dicision threshold = 0.5
        df_metrics_abbvie_concat05 = df_metrics_abbvie_concat.loc[df_metrics_abbvie_concat['decision_threshold'] == 0.5].drop(columns=['decision_threshold'])
        by_row_index_abbvie05 = df_metrics_abbvie_concat05.groupby(['fingerprint'])
        df_abbvie_metrics05_new2 = self.help_save_metrics_10fold(by_row_index_abbvie05, output_filename)
        self.plot_metrics_classification_table_10fold(df_abbvie_metrics05_new2, output_filename) 
        return(df_abbvie_metrics_all_new2)
        
        
        
    @classmethod
    def help_save_metrics_10fold(self, by_row_index_abbvie, output_filename):
        """ Help function for save_metrics_10fold"""
        df_abbvie_gbt_means = by_row_index_abbvie.mean()
        df_abbvie_gbt_std = by_row_index_abbvie.std()
        df_abbvie_gbt_std.columns = ["Std Train Score", "Std Test Score", "Std TP", "Std TN", "Std FP", "Std FN", "Std AUC", "Std Prec", "Std SE", "Std SP", "Std Kappa", "Std F1", "Std MCC"]
        df_abbvie_gbt_metrics_new = pd.concat([df_abbvie_gbt_means, df_abbvie_gbt_std], axis = 1)
        df_abbvie_gbt_metrics_new.to_csv("{}.csv".format(output_filename), index=False)
        df_abbvie_gbt_metrics_new.to_pickle("{}.pkl".format(output_filename))
        df_abbvie_gbt_metrics_new2 = df_abbvie_gbt_metrics_new[['Train Score','Test Score','Std Test Score', 'TP', 'TN', 'FP', 'FN', 'AUC', 'Prec', 'SE', 'SP', 'Kappa', 'F1', 'MCC']]
        df_abbvie_gbt_metrics_new2 = df_abbvie_gbt_metrics_new2.round(2)
        return(df_abbvie_gbt_metrics_new2)

    @classmethod
    def save_metrics_10fold(self, df_metrics_abbvie_concat, output_filename, return_metrics = False):
        df_metrics_abbvie_concat.to_csv("Full_{}.csv".format(output_filename), index=False)
        if 'decision_threshold' in list(df_metrics_abbvie_concat):
            df_abbvie_gbt_metrics_new2 = self.help_save_metrics_10fold_decision_threshold(df_metrics_abbvie_concat, output_filename)
        else:
            by_row_index_abbvie = df_metrics_abbvie_concat.groupby(['fingerprint'])
            df_abbvie_gbt_metrics_new2 = self.help_save_metrics_10fold(by_row_index_abbvie, output_filename)
            self.plot_metrics_classification_table_10fold(df_abbvie_gbt_metrics_new2, output_filename)

        if return_metrics == True:
            df_abbvie_gbt_metrics_new2['fingerprint'] = df_abbvie_gbt_metrics_new2.index 
            df_abbvie_gbt_metrics_new2.reset_index(drop = True, inplace = True)
            if 'decision_threshold' in list(df_abbvie_gbt_metrics_new2):
                return(df_abbvie_gbt_metrics_new2[['fingerprint', 'decision_threshold', 'Train Score', 'Test Score', 'Std Test Score', 'TP', 'TN', 'FP', 'FN', 'AUC', 'Prec', 'SE', 'SP', 'Kappa', 'F1', 'MCC']])
            else:
                return(df_abbvie_gbt_metrics_new2[['fingerprint', 'Train Score', 'Test Score', 'Std Test Score', 'TP', 'TN', 'FP', 'FN', 'AUC', 'Prec', 'SE', 'SP', 'Kappa', 'F1', 'MCC']])


    @classmethod
    def reshape_metrics_decision_threshold(self, df_metrics):
        """ Function to reshaoe metrics dataframe with fingerprint and decision_threshold groups 
        such that the information is contained as pandas columns and not as groups
        """
        if "fingerprint" in list(df_metrics) and "decision_threshold" in list(df_metrics):
            by_row_index_abbvie = df_metrics.groupby(["fingerprint", "decision_threshold"])
            df_group_keys = pd.DataFrame(list(by_row_index_abbvie.groups.keys()), columns = ["fingerprint", "decision_threshold"])
            df_abbvie_gbt_metrics_reshape = pd.concat([df_group_keys, df_metrics.reset_index(level=0, drop=True).reset_index(level=0, drop=True)], axis =1)
            return(df_abbvie_gbt_metrics_reshape)
        elif 'fingerprint' in list(df_metrics) and all(isinstance(item, tuple) for item in list(df_metrics['fingerprint'])) and len(list(df_metrics['fingerprint'])[0]) == 2:
            df_group_keys = pd.DataFrame(list(df_metrics['fingerprint']), columns = ["fingerprint", "decision_threshold"])
            df_abbvie_gbt_metrics_reshape = pd.concat([df_group_keys, df_metrics.drop(columns="fingerprint")], axis =1)
            return(df_abbvie_gbt_metrics_reshape)
        else:
            try:
                by_row_index_abbvie = df_metrics.groupby(["fingerprint", "decision_threshold"])
                df_group_keys = pd.DataFrame(list(by_row_index_abbvie.groups.keys()), columns = ["fingerprint", "decision_threshold"])
                df_abbvie_gbt_metrics_reshape = pd.concat([df_group_keys, df_metrics.reset_index(level=0, drop=True).reset_index(level=0, drop=True)], axis =1)
                return(df_abbvie_gbt_metrics_reshape)
            except:    
                print("I do not know how to reshape. Implement another function?")

 
    @classmethod
    def save_metrics_10fold_rus(self, df_metrics_abbvie_concat, output_filename):
        df_metrics_abbvie_concat.to_csv("Full_{}.csv".format(output_filename), index=False)
        by_row_index_abbvie = df_metrics_abbvie_concat.groupby(['fingerprint'])
        df_abbvie_gbt_means = by_row_index_abbvie.mean()
        df_abbvie_gbt_std = by_row_index_abbvie.std()
        df_abbvie_gbt_std.columns = ["Std round", "Std Train Score", "Std Test Score", "Std Accuracy", "Std TP", "Std TN", "Std FP", "Std FN", "Std AUC", "Std Prec", "Std SE", "Std SP", "Std Kappa", "Std F1", "Std MCC"]
        df_abbvie_gbt_metrics_new = pd.concat([df_abbvie_gbt_means, df_abbvie_gbt_std], axis = 1)
        #df_abbvie_gbt_metrics_new['fingerprint'] = df_abbvie_gbt_metrics_new.index
        df_abbvie_gbt_metrics_new.to_csv("{}.csv".format(output_filename), index=False)
        df_abbvie_gbt_metrics_new.to_pickle("{}.pkl".format(output_filename))
        df_abbvie_gbt_metrics_new2 = df_abbvie_gbt_metrics_new[['Train Score','Test Score','Std Test Score', 'Accuracy', 'Std Accuracy', 'TP', 'TN', 'FP', 'FN', 'AUC', 'Prec', 'SE', 'SP', 'Kappa', 'F1', 'MCC']]
        df_abbvie_gbt_metrics_new2 = df_abbvie_gbt_metrics_new2.round(2)
        self.plot_metrics_classification_table_10fold_rus(df_abbvie_gbt_metrics_new2, output_filename)
        

class Plots:
    def __init__(self):
        pass

    @classmethod
    def barplot_hist_clusters_size(self, clusters, include_singletons, output_name = None):
        if plt:
            plt.gcf().clear()
            plt.clf()
            plt.cla()
            plt.close()

        if include_singletons == False: 
            clusters = [i for i in clusters if len(i) != 1]

        lengths = [len(x) for x in clusters]
        len_unique, len_counts = np.unique(lengths, return_counts=True)     
        len_obj = [str(x) for x in len_unique]
        x_labels = np.arange(len(len_obj))
        plt.figure(figsize = (10, 4.8))        
        plt.bar(x_labels, len_counts,  align='center', alpha=0.5)
        plt.xticks(x_labels, len_obj)
        plt.ylabel('N. of Clusters')
        plt.title('Clusters Size')
        if output_name == None:
            plt.savefig('barplot_clusters_size.png')
        else:
            plt.savefig('barplot_clusters_size_{}.png'.format(output_name))


    @classmethod
    def barplot_clusters(self, test_set, labels_column = None, class_plot_labels = None, colors_classes = None, output_name = None, bar_width = 0.5):
        """ Barplot of the clusters. Each cluster is represented as a bar whose height corresponds to the size of the cluster.
        If labels_column is specified, then the ditribution of the classes in each cluster is illustrated using different colors.
        Colors can be specified with colors_classes while class labels with class_plot_labels
        Parameters:
        -----------
        test_set: df
            dataframe of the test set. It must contain a column named "clusters" with the clustering results. 
            This is generated using the function TrainTestSplit.chemical_series for splitting the dataset.
        labels_column: str, optional
            name of the column containing the class labels or the quantity to learn. Needed only if mixed_clusters_only = True or ratio_cutoff != None.  
        class_plot_labels: list, optional
            List of labels for classes to include in the legend of the plot barplot_clusters_MURKOTYPE_BASENAME_DATASET.png. 
            If classes are numerical (0,1,...N), specify the corrisponding labels in order
        colors_classes: list, optional
            List of colors, one for each of the classes, to use in the plot barplot_clusters_MURKOTYPE_BASENAME_DATASET.png.
            If classes are numerical (0,1,...N), specify the corrisponding colors in order
        output_name: str, optional 
            Basename for the plot barplot_clusters_OUTPUT_NAME.png
        bar_width: float, optional
            Width of the bars (Default = 0.5)

        Returns:
        ----------
        barplot_clusters_OUTPUT_NAME.png: fig, optional
            Barplot of the clusters. Each cluster is shown with a bar whose size is equal to the cluster size. Outputted if plot_clustering_results = True
            If labels_column is provided as input, then also the class proprtion in the clusters are shown using different colors for the different classes. 
            OUTPUT_NAME can be specified as input (output_name), otherwise the figure is saved as barplot_clusters.png
        """
        if plt:
            plt.gcf().clear()
            plt.clf()
            plt.cla()
            plt.close()

        cluster_id, N_clusters = np.unique(test_set['clusters'], return_counts = True)
        labels = np.arange(1, len(cluster_id)+1, 1) 
        fig, ax = plt.subplots(figsize=(12,4.8))
        if labels_column == None:
            ax.bar(labels, N_clusters, bar_width, color = "gray", label='Cluster')
        else:
            class_labels = np.unique(test_set[labels_column])
            if colors_classes == None:
                colors_classes = ["orange","dodgerblue","orchid","yellowgreen","darkred","golenrod","teal","mediumslateblue","pink","silver"]  
            if class_plot_labels == None:
                class_plot_labels = list(class_labels)
            for it, class1 in enumerate(list(class_labels)):
                N_class1 = []
                for id1 in cluster_id:
                    tmp = test_set.loc[test_set.clusters == id1]
                    N_class1.append(tmp.loc[tmp[labels_column] == class1].shape[0])
                if it == 0:
                    ax.bar(labels, N_class1, bar_width, color = colors_classes[it], label=class_plot_labels[it])
                    N_bottom = N_class1
                else:
                    ax.bar(labels, N_class1, bar_width, color = colors_classes[it], bottom = N_bottom, label=class_plot_labels[it])
                    N_bottom = list(np.array(N_bottom) + np.array(N_class1))

        ax.legend()
        ax.set_ylabel('Cluster Size')
        ax.set_xlabel('Cluster')

        if output_name == None:
            plt.savefig('barplot_clusters.png', dpi=300)
        else:
            plt.savefig('barplot_clusters_{}.png'.format(output_name), dpi=300)
        return


     
class TrainTestSplit:

    def __init__(self):
        pass


    @classmethod
    def random_class_stratified(self, df_dataset, labels_column = None, test_set_size = None, random_seed = None):
        """Random Stratified Training-test split for classification models. 
        The dataset is split into a training and a test set such that the the test set has the same class proportions of the dataset.

        Parameters:
        -----------
        df_dataset: df
            dataframe of the dataset. It should contain features, a SMILES column, and classification labels
        labels_column: str, optional
            name of the column containing the binary classification labels (Default = "is_sub")
        test_set_size: int, optional
            size of the test set (Default = 20% of the dataset)
        random_seed: int, optional
            random integer number used as a seed to split the dataset  
        
        Returns:
        -----------
        df_training_set: df
            dataframe of the training set
        df_test_set: df
            dataframe of the test set
        classes_train: list
            list of classification labels of the training set
        classes_test: list
            list of classification labels of the test set
        """

        if test_set_size == None:
            test_set_size = df_dataset.shape[0]/5
            test_size1 = 0.2
        else:
            test_size1 = test_set_size/df_dataset.shape[0]

        if labels_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing binary class labels is required to perform the split".format(labels_column))
            return 

        if random_seed == None:
            random_seed = int(np.random.randint(test_set_size, size=1))

        df_dataset.reset_index(inplace = True, drop = True)
        class_labels, count_class_labels = np.unique(df_dataset[labels_column], return_counts = True)
        N_classes = len(class_labels)

        df_training_set = pd.DataFrame()
        df_test_set = pd.DataFrame()
        for label1 in class_labels: 
            df_class1 = df_dataset.loc[df_dataset[labels_column] == label1]
            df_train_class1, df_test_class1 = train_test_split(df_class1, test_size = test_size1, random_state = random_seed)
            df_training_set = pd.concat((df_training_set, df_train_class1))
            df_test_set = pd.concat((df_test_set, df_test_class1))
          
        classes_train = np.array(df_training_set[labels_column])    #list containing if it is a substrate (1) or not (0)
        classes_test = np.array(df_test_set[labels_column])     #list containing if it is a substrate (1) or not (0)
        df_training_set.reset_index(inplace = True, drop = True)
        df_test_set.reset_index(inplace = True, drop = True)
        return(df_training_set, df_test_set, classes_train, classes_test)



    @classmethod
    def max_chem_diversity(self, df_dataset, smiles_column = "smiles", test_set_size = None, random_seed = None):
        """It splits the dataset into a training and a test set such that the test set contains a maximumally diverse set of compounds.
        To use this function, the input dataset has to contain a smiles_column.
        Help function for diversity_balanced and diversity_stratified functions.
        
        The chemical similarity is quantified using the ECFP4 Tanimoto coefficient. 
        To obtain the test set, a maximally diverse subset of compounds is selected using the MaxMin algorithm in the RDKit.
        The remaining compounds are included in the training set.
 
        Parameters:
        -----------
        df_dataset: df
            dataframe of the dataset. It should contain features, a SMILES column, and classification labels
        smiles_column: str, optional
            name of the column containing the SMILES (Default = "smiles")
        test_set_size: int, optional
            size of the test set (Default = 20% of the dataset)
        random_seed: int, optional
            Integer Number to use as seed for the LazyPick function (Default = random integer) 
        
        Returns:
        -----------
        df_training_set: df
            dataframe of the training set
        df_test_set: df
            dataframe of the test set
        """

        if smiles_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing SMILES is required to compute Morgan fingerprints.".format(smiles_column))
            return

        if 'dataset' in df_dataset:
            df_dataset = df_dataset.drop(columns = ['dataset'])

        df_dataset.reset_index(inplace = True, drop = True)
        indexes = list(df_dataset.index)

        mols =  [Chem.MolFromSmiles(m) for m in df_dataset[smiles_column]]
        morgan_fps  = [AllChem.GetMorganFingerprint(x,2) for x in mols]
        nfps = len(morgan_fps)

        if test_set_size != None:
            n_to_pick = int(test_set_size)
        else:
            n_to_pick = int(df_dataset.shape[0]/5)
   
        if n_to_pick > nfps:
            print("Error: the specified test_set_size is larger than the total number of instances in the dataset. Only 20% of the dataset instances are included in the test set.")
            n_to_pick = int(df_dataset.shape[0]/5)

        if random_seed == None:
            random_seed = int(np.random.randint(n_to_pick, size=1))

        def distij(i,j,morgan_fps=morgan_fps):
            return (1-DataStructs.DiceSimilarity(morgan_fps[i],morgan_fps[j]))

        picker = SimDivFilters.MaxMinPicker()
        pickIndices = picker.LazyPick(distij, nfps, n_to_pick, seed=random_seed)
        test_indexes = list(pickIndices)
        test_indexes = list(np.take(indexes, test_indexes)) 
        train_indexes = list(set(df_dataset.index) ^ set(test_indexes)) 
        df_training_set = df_dataset.iloc[train_indexes]
        df_test_set = df_dataset.iloc[test_indexes]
        df_training_set.reset_index(inplace = True, drop = True)
        df_test_set.reset_index(inplace = True, drop = True)
        return(df_training_set, df_test_set)
    

    @classmethod
    def max_chem_diversity_class_balanced(self, df_dataset, smiles_column = "smiles", labels_column = "is_sub", test_set_size = None, random_seed = None):
        """It splits the dataset into a training and a test set such that the test set 
        contains a maximumally diverse set of compounds and a balanced number of substrates (labels_column = 1) and nonsubstrates (labels_column = 0).
        To use this function, the input dataset has to contain a smiles_column and a labels_column.
        
        To ensure that the test set is balanced, the compounds are first divided into two groups based on their class. 
        Then, the chemical similarity among the groups is quantified using the ECFP4 Tanimoto coefficient. 
        Finally, to obtain the test set, a maximally diverse subset of compounds is selected from each group using the MaxMin algorithm in the RDKit.
        The remaining compounds are included in the training set.
 
        Parameters:
        -----------
        df_dataset: df
            dataframe of the dataset. It should contain features, a SMILES column, and classification labels
        smiles_column: str, optional
            name of the column containing the SMILES (Default = "smiles")
        labels_column: str, optional
            name of the column containing the binary classification labels (Default = "is_sub")
        test_set_size: int, optional
            size of the test set (Default = 20% of the dataset)
        random_seed: int, optional
            Integer Number to use as seed for the LazyPick function (Default = random integer) 
        
        Returns:
        -----------
        df_training_set: df
            dataframe of the training set
        df_test_set: df
            dataframe of the test set
        classes_train: list
            list of classification labels of the training set
        classes_test: list
            list of classification labels of the test set
        """

        if smiles_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing SMILES is required to compute Morgan fingerprints.".format(smiles_column))
            return

        if labels_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing binary class labels is required to perform the split".format(labels_column))
            return 

        if 'dataset' in df_dataset:
            df_dataset = df_dataset.drop(columns = ['dataset'])

        df_dataset.reset_index(inplace = True, drop = True)

        df_dataset[smiles_column] = df_dataset[smiles_column].astype(str) 
        # check class labels and count instances for each class
        class_labels, count_class_labels = np.unique(df_dataset[labels_column], return_counts = True)
        N_classes = len(class_labels)

        if test_set_size != None:
            n_to_pick = int(test_set_size/N_classes)
        else:
            n_to_pick = int(df_dataset.shape[0]/5/N_classes)

        if n_to_pick > min(count_class_labels):
            n_to_pick = min(count_class_labels)
            print("Warning: The number of instances for one of the classes is smaller than 1/{} of the test_set_size. \n Only {} instances are included in the test set in order to have class balanced instances in the test set. \n Warning: In doing so, the training set will not have representative instances for one of the classes".format(N_classes,  n_to_pick*N_classes))
            
        df_training_set = pd.DataFrame()
        df_test_set = pd.DataFrame()
        for label1 in class_labels: 
            df_class1 = df_dataset.loc[df_dataset[labels_column] == label1]
            df_train_class1, df_test_class1 = TrainTestSplit.max_chem_diversity(df_class1, smiles_column = smiles_column, test_set_size = n_to_pick, random_seed = random_seed)
            df_training_set = pd.concat((df_training_set, df_train_class1))
            df_test_set = pd.concat((df_test_set, df_test_class1))

        classes_train = np.array(df_training_set[labels_column])    #list containing if it is a substrate (1) or not (0)
        classes_test = np.array(df_test_set[labels_column])     #list containing if it is a substrate (1) or not (0)
        df_training_set.reset_index(inplace = True, drop = True)
        df_test_set.reset_index(inplace = True, drop = True)
        return(df_training_set, df_test_set, classes_train, classes_test)


    @classmethod
    def max_chem_diversity_class_stratified(self, df_dataset, smiles_column = "smiles", labels_column = "is_sub", test_set_size = None, random_seed = None):
        """It splits the dataset into a training and a test set such that the test set 
        contains a maximumally diverse set of compounds. Moreover, the test set preserves the class distribution of the (eventually imbalanced) dataset.
        To use this function, the input dataset has to contain a smiles_column and a labels_column.
        
        To ensure that the test set is stratified, the compounds are first divided into two groups based on their class. 
        Then, the chemical similarity among the groups is quantified using the ECFP4 Tanimoto coefficient. 
        Finally, to obtain the test set, a maximally diverse subset of compounds is selected from each group using the MaxMin algorithm in the RDKit.
        The remaining compounds are included in the training set.
 
        Parameters:
        -----------
        df_dataset: df
            dataframe of the dataset. It should contain features, a SMILES column, and classification labels
        smiles_column: str, optional
            name of the column containing the SMILES (Default = "smiles")
        labels_column: str, optional
            name of the column containing the binary classification labels (Default = "is_sub")
        test_set_size: int, optional
            size of the test set (Default = 20% of the dataset)
        random_seed: int, optional
            Integer Number to use as seed for the LazyPick function (Default = random integer) 
        
        Returns:
        -----------
        df_training_set: df
            dataframe of the training set
        df_test_set: df
            dataframe of the test set
        classes_train: list
            list of classification labels of the training set
        classes_test: list
            list of classification labels of the test set
        """

        if smiles_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing SMILES is required to compute Morgan fingerprints.".format(smiles_column))
            return

        if labels_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing binary class labels is required to perform the split".format(labels_column))
            return 

        if 'dataset' in df_dataset:
            df_dataset = df_dataset.drop(columns = ['dataset'])

        df_dataset.reset_index(inplace = True, drop = True)

        df_dataset[smiles_column] = df_dataset[smiles_column].astype(str) 
        # check class labels and count instances for each class
        class_labels, count_class_labels = np.unique(df_dataset[labels_column], return_counts = True)

        if test_set_size != None:
            fraction = df_dataset.shape[0]/test_set_size
            n_to_pick = np.intc(count_class_labels/fraction)
        else:
            n_to_pick = np.intc(count_class_labels/5)

        df_training_set = pd.DataFrame()
        df_test_set = pd.DataFrame()
        for idx, label1 in enumerate(class_labels): 
            df_class1 = df_dataset.loc[df_dataset[labels_column] == label1]
            df_train_class1, df_test_class1 = TrainTestSplit.max_chem_diversity(df_class1, smiles_column = smiles_column, test_set_size = n_to_pick[idx], random_seed = random_seed)
            df_training_set = pd.concat((df_training_set, df_train_class1))
            df_test_set = pd.concat((df_test_set, df_test_class1))

        classes_train = np.array(df_training_set[labels_column])    #list containing if it is a substrate (1) or not (0)
        classes_test = np.array(df_test_set[labels_column])     #list containing if it is a substrate (1) or not (0)
        df_training_set.reset_index(inplace = True, drop = True)
        df_test_set.reset_index(inplace = True, drop = True)
        return(df_training_set, df_test_set, classes_train, classes_test)


    @classmethod
    def ClusterFps(self, fps, clustering_cutoff=0.2):
        """Cluster the fingerprints using the Butina algorithm

        Parameters:
        -----------
        fps: list
            list of Morgan fingerprints 
        clustering_cutoff: float, optional
            cutoff used for clustering (Default = 0.2)

        Returns:
        ----------
        cs: list
            a tuple of tuples containing information about the clusters. Output of the Butina clustering algorithm    
        """
        # first generate the distance matrix:
        dists = []
        nfps = len(fps)
        for i in range(1,nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
            dists.extend([1-x for x in sims])
        # now cluster the data:
        cs = Butina.ClusterData(dists,nfps,clustering_cutoff,isDistData=True)
        return cs


    @classmethod
    def select_testset_from_mixed_clusters(self, df_subs, selected_clusters_subs, sorted_indexes_of_mixed_clusters, test_set_size, mixed_clusters_only = None):
        """ Help function for chemical_series"""
        #training-test split
        # Select mixed series until the test set contains 20 % of the data. 
        # Clusters are picked from the more mixed to the least mixed
        if mixed_clusters_only == None:
            mixed_clusters_only = False

        test_subs1 = []
        cluster_info1 = []
        flag = 0
        for clus_id in sorted_indexes_of_mixed_clusters:
            if flag == 0:
                len_test1 = len(test_subs1)
                test_subs2 = test_subs1 + [x for x in selected_clusters_subs[clus_id]]
                cluster_info2 = cluster_info1 + [clus_id] * len(selected_clusters_subs[clus_id]) 
                len_test2 = len(test_subs2)
                if len_test2 >= test_set_size:
                    if (len_test1 - test_set_size) < (len_test2 - test_set_size):
                        test_subs = test_subs1  
                        cluster_info = cluster_info1 
                    else:
                        test_subs = test_subs2
                        cluster_info = cluster_info2
                    flag=1
                else:
                    test_subs1 = test_subs2
                    cluster_info1 = cluster_info2
            else:
                "All picked clusters contain multiple classes"
                break
 
        # if mixed_clusters_only == False and the test set does not contain yet 20% of the data, than add additional not mixed clusters.
        if flag == 0 and mixed_clusters_only == False:    
            random_numbers = np.arange(len(selected_clusters_subs))
            random_clus_id_subs = set(random_numbers) ^ set(sorted_indexes_of_mixed_clusters)
            for clus_id in random_clus_id_subs:
                if flag == 0:
                    len_test1 = len(test_subs1)
                    test_subs2 = test_subs1 + [x for x in selected_clusters_subs[clus_id]]
                    cluster_info2 = cluster_info1 + [clus_id] * len(selected_clusters_subs[clus_id]) 
                    len_test2 = len(test_subs2)
                    if len_test2 >= test_set_size:
                        if (len_test1 - test_set_size) < (len_test2 - test_set_size):
                            test_subs = test_subs1    
                            cluster_info = cluster_info1 
                        else:
                            test_subs = test_subs2
                            cluster_info = cluster_info2 
                        flag=1
                    else:
                        test_subs1 = test_subs2
                        cluster_info1 = cluster_info2
                else:
                    break
         
        elif flag == 0 and mixed_clusters_only == True:
            test_subs = test_subs2    
            cluster_info = cluster_info2        

        train_indexes = list(set(df_subs.index) ^ set(test_subs))
        df_training_set = df_subs.iloc[train_indexes]
        df_test_set = df_subs.iloc[test_subs]
        df_test_set['clusters'] = cluster_info

        return(df_training_set, df_test_set)
        


    @classmethod
    def select_testset_from_clusters(self, df_properties, selected_clusters, test_set_size = None, include_singletons_in_test = False):
        """ Help function for chemical_series"""

        def remove_large_clusters_and_shuffle_ids(selected_clusters, max_clus_size = None, include_singletons_in_test = False):
            if max_clus_size != None:
                selected_clusters_tmp = [i for i in selected_clusters if len(i) > 1 and len(i) <= max_clus_size]
            else:
                selected_clusters_tmp = [i for i in selected_clusters if len(i) > 1]
            if len(selected_clusters_tmp) > 0:
                random_clus_id = np.arange(len(selected_clusters_tmp)) 
                np.random.shuffle(random_clus_id)
            else:
                random_clus_id = []
            if include_singletons_in_test: 
                selected_clusters_singletons = [i for i in selected_clusters if len(i) == 1]
                if len(selected_clusters_singletons) > 0:
                    singletons_id = np.arange(len(selected_clusters_singletons)) + len(selected_clusters_tmp)
                    np.random.shuffle(singletons_id)
                    selected_clusters = selected_clusters_tmp + selected_clusters_singletons
                    random_clus_id = list(random_clus_id) + list(singletons_id)
            else:
                selected_clusters = selected_clusters_tmp
            return selected_clusters, random_clus_id
    
        if isinstance(test_set_size, int):
            selected_clusters, random_clus_id = remove_large_clusters_and_shuffle_ids(selected_clusters, max_clus_size = test_set_size, include_singletons_in_test = include_singletons_in_test)
            selected_clusters_tmp = [i for i in selected_clusters if len(i) > 1]
            N_compounds_in_clusters = len([item for sublist in selected_clusters_tmp for item in sublist])
            if test_set_size > N_compounds_in_clusters:
                print("The number of compounds in clusters ({}) is smaller than test_set_size ({})".format(N_compounds_in_clusters, test_set_size))
                if include_singletons_in_test:
                    print("Since include_singletons_in_test = True, {} singletons will be added to the test set such that the size of the test set = {}".format(test_set_size - N_compounds_in_clusters, test_set_size))
                else:
                    print("Since include_singletons_in_test = False, the size of the test set will not be {}, but {}".format(test_set_size, N_compounds_in_clusters))
            else:
                include_singletons_in_test = False

            #training-test split
            # Select series until the test set = test_set_size. 
            if len(selected_clusters) == 0:
                flag = 1
            else:
                flag = 0

            # select first cluster
            test_subs1 = []
            cluster_info1 = []
            iteration = 0
            while flag == 0:
                iteration += 1
                new_max = test_set_size - len(test_subs1)
                # remove clusters larger than the remaining test set size and reshuffle
                selected_clusters, random_clus_id = remove_large_clusters_and_shuffle_ids(selected_clusters, max_clus_size = new_max, include_singletons_in_test = include_singletons_in_test)
                test_subs2 = [x for x in selected_clusters[random_clus_id[0]]]
                test_subs1 = test_subs1 + test_subs2 
                cluster_info1 = cluster_info1 +  [iteration] * len(test_subs2)
                # check that test_set_size was not reached
                if (test_set_size - len(test_subs1)) == 0 or min([len(x) for x in selected_clusters]) > (test_set_size - len(test_subs1)):
                    flag = 1
                
            train_indexes = list(set(df_properties.index) ^ set(test_subs1))
            df_training_set = df_properties.iloc[train_indexes]
            df_test_set = df_properties.iloc[test_subs1]
            df_test_set['clusters'] = cluster_info1
        else:
            selected_clusters_tmp = [i for i in selected_clusters if len(i) > 1]
            cluster_info = []
            for idx in range(len(selected_clusters_tmp)):
                cluster_info = cluster_info + [idx] * len(selected_clusters_tmp[idx])

            test_indexes = [item for sublist in selected_clusters_tmp for item in sublist]
            train_indexes = list(set(df_properties.index) ^ set(test_indexes))
            df_training_set = df_properties.iloc[train_indexes]
            df_test_set = df_properties.iloc[test_indexes]
            df_test_set['clusters'] = cluster_info
        return(df_training_set, df_test_set)


    @classmethod
    def get_classes_ratio_in_clusters(self, selected_clusters_subs, properties, labels_column, flag_ratio_cutoff, plot_clustering_results, output_name = None):
        """ Help function for chemical_series"""
        extract_is_sub = []
        for cluster1 in selected_clusters_subs:
            labels_per_cluster = []
            for item1 in cluster1:
                labels_per_cluster.append(properties[labels_column].iloc[item1])
            extract_is_sub.append(labels_per_cluster)

        count_classes_per_cluster = []
        for i in extract_is_sub:
            count_classes_per_cluster.append(np.unique(i, return_counts = True)[1])

        mixed_clusters = []
        ratio = []
        baseline = []
        count_classes_per_cluster_selected = []
        for index,i  in enumerate(count_classes_per_cluster):
            if len(i) != 1:
                mixed_clusters.append(index)
                ratio.append(max(i)/min(i))
                baseline.append(max(i)/sum(i))
                if flag_ratio_cutoff == 1:
                    count_classes_per_cluster_selected.append(i)

        if flag_ratio_cutoff == 1:
            return(baseline)
        elif flag_ratio_cutoff == 0:
            return(mixed_clusters, ratio, baseline)



    @classmethod
    def chemical_series_per_class(self, df_dataset, smiles_column = "smiles", labels_column = "is_sub", remove_atom_types = False, exclude_series_larger_than = None, exclude_from_test_series_larger_than = None, test_set_size = None, include_singletons_in_test = False, clustering_cutoff = 0.2, balanced = False, stratified = False, plot_clustering_results = None, plot_basename = None):
        """ The dataset is split into a training and a test set, such that the test set contains chemical series for each of the classes. 
        For each of the classes, chemical series are identified using the following procedure: 
        (i) compounds are divided into groups based on their class
        (ii) compounds are decomposed into Murcko frameworks using RDKit. One can use either standard Murcko scaffolds, which retain the information on atom types and bond order, or generic Murcko scaffolds,  which do not retain the information on atom types and bond order. Standard Murcko scaffolds are used by default while generic Murcko scaffolds can be used by setting remove_atom_types = True
        (iii) Frameworks are represented by ECFP4 fingerprints
        (iv) Separately for each class group, ECFP4 are clustered based on the Tanimoto similarity using the Butina algorithm in the RDKit 

        Note that: 
        - If balanced = True, then clusters are selected from each class groups to form the test set such that the test set contains almost equal number of instances for each of the classes. Clusters are picked until the size if the test set is equal to test_set_size. If balanced = True and test_set_size is not specified than test_set_size is automatically set to 20% of the dataset 
        - If stratified = True, then clusters are selected from each class groups to form the test set such that the class distribution of the test set resemble the one of the whole dataset. Clusters are picked until the size if the test set is equal to test_set_size. If stratified = True and test_set_size is not specified than test_set_size is automatically set to 20% of the dataset 
        - if test_set_size is specified both both balanced = False and stratified = False, then stratified is set to True

        Parameters:
        -----------
        df_dataset: df
            dataframe of features. It must also contain a column containing SMILES. Specify the column name using smiles_column
        smiles_column: str, optional
            name of the column containing the SMILES (Default = "smiles") 
        labels_column: str, optional
            name of the column containing the class labels. Needed to perform the split. (Default = "is_sub")  
        remove_atom_types: bool, optional 
            False to use Murcko Scaffolds (retaining atom types and bond order). True to use Generic Murcko scaffolds (only skeleton). (Default = False)
        exclude_series_larger_than: int, optional
            Exclude from the dataset (both from the training and the test set) chemical series larger than the specified integer number (Default = None)
        exclude_from_test_series_larger_than: int, optional
            Exclude from the test set chemical series larger than the specified integer number (Default = None)
        test_set_size: int, optional
            Set test set size. If None is specified, all chemical series (clusters with 2 or more members) are included in the test set. (Default = None)
        include_singletons_in_test: bool, optional
            True to include singletons in the test set. However, singletons are only added to the test set if the compounds in chemical series are less than the specified value of test_set_size. If test_set_size is not specified, than only chemical series (clusters with 2 or more members) are included in the test set. (Default = False)
        clustering_cutoff: float, optional
            Cutoff for clustering (Default = 0.2)
        balanced: bool,optional
            True to have class balanced instances in the test set (Default = False)
        stratified: bool, optional
            True to ensure that the test set has the same class distribution as the whole dataset (Default = False)
        plot_clustering_results: bool, optional
            True to plot the clustering results (Default = False)
        plot_basename: str, optional 
            Basename for the plots generated if plot_clustering_results = True (Default = None)
            
        Returns:
        ----------
        df_train: df
            Dataframe of the training set
        df_test: df
            Dataframe of the test set.
            Compared to the original dataframe, this has an additional column named "clusters" which contains the clusters ID
        barplot_clusters_size_MURKOTYPE_BASENAME_DATASET_class_CLASS.png: fig, optional
            barplot of the sizes of the output clusers: counts vs cluster size. Outputted if plot_clustering_results = True
            BASENAME is the figure basename eventually specified with plot_basename,
            MURKOTYPE is the type of mucko framework: "murcko" if remove_atom_types = False (Default) and "generic_murcko" if remove_atom_types = True,
            DATASET specifies if the clustering results are for the entire dataset (DATASET = "dataset") or for the test set (DATASET = "testset"),
            and CLASS indicates the class being clustered. One plot is generated for each of the classes.
        """

        if smiles_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing SMILES is required. See the documentation, help(TrainTestSplit.chemical_series_per_class)".format(smiles_column))
            return(df_dataset) 

        if labels_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing binary class labels is required to perform the split".format(labels_column))
            return 

        if test_set_size != None and balanced == False and stratified == False:
            stratified = True
        if balanced and stratified:
            print("Error: Both balanced and stratified were set to True. Only one can be set to True. See documentation help(TrainTestSplit.chemical_series_per_class)")
            return

        if remove_atom_types == True:
            plot_name = 'generic_murcko'
        else:
            plot_name = 'murcko'
        if plot_basename != None:
            plot_name = '{}_{}'.format(plot_name, plot_basename)

        # reshape clustering data for plotting
        def reshape_clustering_results(df_test):
            clusters_test = list(df_test["clusters"])
            a = np.unique(clusters_test)
            cl = []
            for i in a:
                df_tmp = df_test.loc[df_test.clusters == i]
                cl.append(list(df_tmp.index))
            return(cl)

        df_dataset.reset_index(inplace = True, drop = True)
        class_labels, count_class_labels = np.unique(df_dataset[labels_column], return_counts = True)
        N_classes = len(class_labels)
        
        if test_set_size != None:
            if balanced:
                n_to_pick = [int(test_set_size/N_classes)]*N_classes 
            elif stratified:
                fraction = df_dataset.shape[0]/test_set_size
                n_to_pick = list(np.intc(count_class_labels/fraction))
 
        df_training_set = pd.DataFrame()
        df_test_set = pd.DataFrame()
        for idx, label1 in enumerate(class_labels): 
            df_class1 = df_dataset.loc[df_dataset[labels_column] == label1]
            df_train_class1, df_test_class1 = self.chemical_series(df_class1, smiles_column = smiles_column, remove_atom_types = remove_atom_types, exclude_series_larger_than = exclude_series_larger_than, exclude_from_test_series_larger_than = exclude_from_test_series_larger_than, include_singletons_in_test = include_singletons_in_test)
            # reshape, plot results, pick for test set, plot results
            clusters_class1 = reshape_clustering_results(df_test_class1)
            Plots.barplot_hist_clusters_size(clusters_class1, include_singletons_in_test, output_name = plot_name + "_dataset_class_{}".format(label1))  
            if test_set_size != None:
                df_train_class1_part2, df_test_class1_new = self.select_testset_from_clusters(df_test_class1, clusters_class1, test_set_size = int(n_to_pick[idx]), include_singletons_in_test = include_singletons_in_test)
                clusters_class12 = reshape_clustering_results(df_test_class1_new)
                Plots.barplot_hist_clusters_size(clusters_class1, include_singletons_in_test, output_name = plot_name + "_testset_class_{}".format(label1))  
                df_train_class1 = pd.concat((df_train_class1, df_train_class1_part2))
                df_test_class1 = df_test_class1_new
            # append to training and test dataframes
            df_training_set = pd.concat((df_training_set, df_train_class1))
            df_test_set = pd.concat((df_test_set, df_test_class1))
          
        classes_train = np.array(df_training_set[labels_column])    #list containing if it is a substrate (1) or not (0)
        classes_test = np.array(df_test_set[labels_column])     #list containing if it is a substrate (1) or not (0)
        df_training_set.reset_index(inplace = True, drop = True)
        df_test_set.reset_index(inplace = True, drop = True)
        return(df_training_set, df_test_set, classes_train, classes_test)


    @classmethod
    def chemical_series(self, df_dataset, smiles_column = "smiles", labels_column = None, remove_atom_types = False, exclude_series_larger_than = None, exclude_from_test_series_larger_than = None, test_set_size = None, include_singletons_in_test = False, mixed_clusters_only = False, ratio_cutoff = None, clustering_cutoff = 0.2, plot_clustering_results = False, plot_basename = None, class_plot_labels = None, colors_classes = None):
        """The dataset is split into a training and a test set, such that the test set contains chemical series, while the training set contains the remaining compounds.
        Chemical series are identified by clustering the dataset using the following procedure: 
        (i) compounds are decomposed into Murcko frameworks using the RDKit. One can use either standard Murcko scaffolds, which retain the information on atom types and bond order, or generic Murcko scaffolds,  which do not retain the information on atom types and bond order. Standard Murcko scaffolds are used by default while generic Murcko scaffolds can be used by setting remove_atom_types = True
        (ii) Frameworks are represented by ECFP4 fingerprints
        (iii) ECFP4 are clustered based on the Tanimoto similarity using the Butina algorithm in the RDKit

        Note that: 
        - If mixed_clusters_only = True, clusters containing only one class of molecules are excluded from the test set validation and included in the training set
        - If ratio_cutoff is specified, clusters with an imbalance ratio higher than the specified cutoff are excluded from the test set. By excluding highly imbalanced series from the test set, one avoids cases in which the accuracy does not reflect the classification ability of a model. 
        - If exclude_from_test_series_larger_than is specified than chemical series larger than the specified values are excluded from the test set while if exclude_series_larger_than is specified, large chemical series are excluded from both the training and the test set. To check the sizes of the chemical series contained in the dataset, visualize the output plot ... which is generated if plot_clustering_results = True. 

        Parameters:
        -----------
        df_dataset: df
            dataframe of features. It must also contain a column containing SMILES. Specify the column name using smiles_column
        smiles_column: str, optional
            name of the column containing the SMILES (Default = "smiles") 
        labels_column: str, optional
            name of the column containing the class labels or the quantity to learn. Needed only if mixed_clusters_only = True or ratio_cutoff != None.  
        remove_atom_types: bool, optional 
            False to use Murcko Scaffolds (retaining atom types and bond order). True to use Generic Murcko scaffolds (only skeleton). (Default = False)
        exclude_series_larger_than: int, optional
            Exclude from the dataset (both from the training and the test set) chemical series larger than the specified integer number (Default = None)
        exclude_from_test_series_larger_than: int, optional
            Exclude from the test set chemical series larger than the specified integer number (Default = None)
        test_set_size: int, optional
            Set test set size. If None is specified, all chemical series (clusters with 2 or more members) are included in the test set. (Default = None)
        include_singletons_in_test: bool, optional
            True to include singletons in the test set. However, singletons are only added to the test set if the compounds in chemical series are less than the specified value of test_set_size. If test_set_size is not specified, than only chemical series (clusters with 2 or more members) are included in the test set. (Default = False)
        mixed_clusters_only: bool, optional 
            For classification models. True to include in the test set only chemical series containing more than one class of compounds, False to also include clusters containing a single class of compounds in the test set (Default = False)
        ratio_cutoff: float, optional
            Exclude chemical series with imbalance ratio higher than the specified cutoff (Default = None)
        clustering_cutoff: float, optional
            Cutoff for clustering (Default = 0.2)
        plot_clustering_results: bool, optional
            True to plot the clustering results (Default = False)
        plot_basename: str, optional 
            Basename for the plots generated if plot_clustering_results = True (Default = None)
        class_plot_labels: list, optional
            List of labels for classes to include in the legend of the plot barplot_clusters_MURKOTYPE_BASENAME_DATASET.png. 
            If classes are numerical (0,1,...N), specify the corrisponding labels in order
        colors_classes: list, optional
            List of colors, one for each of the classes, to use in the plot barplot_clusters_MURKOTYPE_BASENAME_DATASET.png.
            If classes are numerical (0,1,...N), specify the corrisponding colors in order
            
        Returns:
        ----------
        df_train: df
            Dataframe of the training set
        df_test: df
            Dataframe of the test set.
            Compared to the original dataframe, this has an additional column named "clusters" which contains the clusters ID
        labels_train: list, optional
            list of labels of the training set. Only returned if labels_column is specified
        labels_test: list, optional
            list of labels of the test set. Only returned if labels_column is specified

        barplot_clusters_size_MURKOTYPE_BASENAME_DATASET.png: fig, optional
            barplot of the sizes of the output clusers: counts vs cluster size. Outputted if plot_clustering_results = True
            BASENAME is the figure basename eventually specified with plot_basename,
            MURKOTYPE is the type of mucko framework: "murcko" if remove_atom_types = False (Default) and "generic_murcko" if remove_atom_types = True,
            and DATASET specify if the clustering results are for the entire dataset (DATASET = "dataset") or for the test set (DATASET = "testset").
        barplot_clusters_MURKOTYPE_BASENAME_DATASET.png: fig, optional
            Barplot of the clusters. Each cluster is shown with a bar whose size is equal to the cluster size. Outputted if plot_clustering_results = True
            If labels_column is provided as input, then also the class proprtion in the clusters are shown using different colors for the different classes. BASENAME, MURKOTYPE, DATASET are defined above.
        """

        if smiles_column not in list(df_dataset):
            print("Error: the column {} is not contained in the input dataframe. A column containing SMILES is required. See the documentation, help(TrainTestSplit.chemical_series)".format(smiles_column))
            return(df_dataset) 

        flag_ratio_cutoff = 0

        df_dataset.reset_index(drop=True, inplace = True)
        smi = list(df_dataset[smiles_column])
        nfps = len(smi)
        # generate Murcko Scaffolds
        smi_murcko = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(x) for x in smi]

        # if true generate Generic Murcko Scaffolds (no atom types or bonds information)
        if remove_atom_types == True:
            plot_name = 'generic_murcko'
            smi_generic_murcko = [Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(Chem.MolFromSmiles(x))) for x in smi_murcko]
            smi_generic_murcko = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(x) for x in smi_generic_murcko]
            fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(m), 2) for m in smi_generic_murcko]
        else:
            plot_name = 'murcko'
            fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(m), 2) for m in smi_murcko]

        if plot_basename != None:
            plot_name = '{}_{}'.format(plot_name, plot_basename)

        # Cluster the structures. Outout is a list of clusters, sorted in ascending order according to the cluster size.
        # Each cluster sublist contains the indexes of the mols. Output example: [(m1, m2, m3),(m4, m5),(m6, m7), (m8)]
        cluster = sorted(self.ClusterFps(fps, clustering_cutoff = clustering_cutoff), key = len)

        # plot clustering results
        if plot_clustering_results == True:
            Plots.barplot_hist_clusters_size(cluster, include_singletons_in_test, output_name = plot_name + "_dataset") 

        #select sizes of the serie to put in the training and test set according to the specified inputs. Default no exclusion.
        if isinstance(exclude_series_larger_than, int):
            cluster = [i for i in cluster if len(i) <= exclude_series_larger_than]
        else:
            print("No chemical series is excluded from the dataset")

        # Eventually, exclude singletons or series larger than a certain size from the test set
        if include_singletons_in_test == False: 
            print("Singletons are NOT picked for the test set but are included in the training set".format(exclude_from_test_series_larger_than))
            if isinstance(exclude_from_test_series_larger_than, int):
                print("Chemical series larger than {} are NOT picked for the test set".format(exclude_from_test_series_larger_than))
                selected_clusters  = [i for i in cluster if (len(i) <= exclude_from_test_series_larger_than and len(i) > 1)]
            else:
                print("Large chemical series can also be picked for the test set".format(exclude_from_test_series_larger_than))
                selected_clusters = [i for i in cluster if len(i) > 1]
        elif include_singletons_in_test:
            print("Also singletons can be picked for the test set".format(exclude_from_test_series_larger_than))
            if isinstance(exclude_from_test_series_larger_than, int):
                print("Chemical series larger than {} are NOT picked for the test set".format(exclude_from_test_series_larger_than))
                selected_clusters  = [i for i in cluster if len(i) <= exclude_from_test_series_larger_than]
            else:
                print("Large chemical series can be picked for the test set")
                selected_clusters = cluster

        # pick only clusters than contain istances belonging to at least two classes, such that clusters with a lower imbalance ratio are picked firsr.
        # Imbalance ratio = ratio between the number of samples in the majority class and the one in the minority class. 
        # Only for  classification.if mixed_clusters_only: 
        if mixed_clusters_only: 
            if labels_column == None:
                print("Error: The labels_column was not specified")
                return
            mixed_clusters, ratio, baseline = self.get_classes_ratio_in_clusters(selected_clusters, df_dataset, labels_column, flag_ratio_cutoff, plot_clustering_results = False, output_name = plot_name)
            sorted_ratios = [ratio[i] for i in np.argsort(ratio)]
            sorted_indexes_of_mixed_clusters = [mixed_clusters[i] for i in np.argsort(ratio)]
            # Exclude from the test set chemical series with an imbalance ratio higher than ratio_cutoff
            if ratio_cutoff != None:
                flag_ratio_cutoff = 1
                clusters_indexes_new = []
                ratios_new = []
                for idx, i in enumerate(sorted_ratios):
                    if i <= ratio_cutoff:
                        ratios_new.append(i)
                        clusters_indexes_new.append(sorted_indexes_of_mixed_clusters[idx])
 
                clusters_new = []
                for clus_id in clusters_indexes_new:
                    clusters_new.append([x for x in selected_clusters[clus_id]])

                baseline = self.get_classes_ratio_in_clusters(clusters_new, df_dataset, labels_column, 1, plot_clustering_results = False)
                # pick compounds from clusters until the number of selected compounds is equal to test_set_size.
                if isinstance(test_set_size, int):
                    df_train, df_test = self.select_testset_from_mixed_clusters(df_dataset, selected_clusters, clusters_indexes_new, test_set_size, mixed_clusters_only = mixed_clusters_only)
                else:
                    df_train, df_test = self.select_testset_from_clusters(df_dataset, clusters_new, include_singletons_in_test = include_singletons_in_test)
            else:
                # pick all compounds from clusters
                clusters_new = []
                for clus_id in sorted_indexes_of_mixed_clusters:
                    clusters_new.append([x for x in selected_clusters[clus_id]])

                baseline = self.get_classes_ratio_in_clusters(clusters_new, df_dataset, labels_column, 1, plot_clustering_results = False)
                # pick compounds from clusters until the number of selected compounds is equal to test_set_size.
                if isinstance(test_set_size, int):
                    df_train, df_test = self.select_testset_from_mixed_clusters(df_dataset, selected_clusters, sorted_indexes_of_mixed_clusters, test_set_size, mixed_clusters_only = mixed_clusters_only)
                else:
                    df_train, df_test = self.select_testset_from_clusters(df_dataset, clusters_new, include_singletons_in_test = include_singletons_in_test)
        else:
            # All chemical series are included in the test set
            df_train, df_test = self.select_testset_from_clusters(df_dataset, selected_clusters, test_set_size = test_set_size, include_singletons_in_test = include_singletons_in_test)
 
        #Print Baseline accuracy
        try: 
            print("Baseline Accuracy: {}".format(sum(np.array(baseline) * 100)/len(baseline)))
        except:
            pass

        df_train.reset_index(drop = True, inplace = True)
        df_test.reset_index(drop = True, inplace = True)

        # plot clustering results
        if plot_clustering_results == True:
             Plots.barplot_clusters(df_test, labels_column = labels_column, class_plot_labels = class_plot_labels, colors_classes = colors_classes, output_name = plot_name + "_testset")
             # reshape clustering data for plotting
             clusters_test = list(df_test["clusters"])
             a, b = np.unique(clusters_test, return_counts = True)
             cl = []
             for i, j in enumerate(b):
                 cl.append([i]*j)
             Plots.barplot_hist_clusters_size(cl, include_singletons_in_test, output_name = plot_name + "_testset") 

        if labels_column != None and labels_column in list(df_dataset):
            labels_train = list(df_train[labels_column])
            labels_test = list(df_test[labels_column])
            return(df_train, df_test, labels_train, labels_test)
        else:    
            return(df_train, df_test)



    @classmethod
    def gen_train_subsets_for_random_undersampling(self, df_training_set, labels_column = "is_sub"): 
        """Generate class balanced subsets of the training set such that the entire training set is covered.
        It returns a list of lists (one for every subset). Each list contains the indexes to extract from the training set to compose the subset. 
        To extract the subsets from the dataframe of the training set, use the iloc function. 
        Warning: Before extracting the subsets from the training set, make sure that the indeces start from 0. Otherwise, reset indeces first: df_training_set.reset_index(drop=True, inplace = True)   

        Parameters:
        -----------
        df_training_set: df
            dataframe of the training set
        labels_column: str, optional
            name of the column containing the class labels

        Returns:
        ----------
        train_subset_indeces: list
            list of lists. The sublists contain the indices of the training subsets. To extract the subsets from the dataframe of the training set  
        """
        # Rebalance dataset with Random over sampling. Split training and test set after balancing
        #
        #  __________________________________
        #  |       80%           |    20%   |
        #  |_____________________|__________|
        #                  |            
        #       train-test | 	            
        #            split |	        
        #                  V	        
        #           TRAIN		         TEST
        #   ______________________       ______________________
        #   |  80%        | 20%  |       |   50%   |    50%   |
        #   |_____________|______|       |_________|__________|
        #              |     |
        #   undersampl |     | all
        #      (random |     |
        #  /diversity) V     V
        #         ______________________
        #         |   50%   |    50%   |
        #         |_________|__________|
        #

        df_training_set.reset_index(drop=True, inplace = True)
        train_indexes = df_training_set.index
        classes_train = list(df_training_set[labels_column])

        rus = RandomUnderSampler(return_indices=True)
        
        flag = 0
        count = 0
        train_subset_indeces = []
        while flag == 0:
            if count == 0:
                X_rus, y_rus, id_unique = rus.fit_sample(df_training_set, classes_train)
                train_subset_indeces.append(id_unique)
                count += 1
            else:
                count += 1
                X_rus, y_rus, id_rus2 = rus.fit_sample(df_training_set, classes_train)
                train_subset_indeces.append(id_rus2)
                id_unique = np.unique(np.concatenate([id_unique, id_rus2], axis=0))
                if len(set(id_unique) ^ set(train_indexes)) == 0:
                    print('The training set was split into {} subsets'.format(count))
                    flag = 1

        return(train_subset_indeces)


class MDFP_terms:
    """ Class that defines the terms composing the MDFPs. Definitions are stored in the dictionary descriptor_dictionary.
    Colors are also specified for each of the features and stores in the dictionary colors_dictionary (useful for the plot of the feature importances).

    To check all defined descriptors, type in the python shell    MDFP_terms.descriptor_dictionary.keys()
    This returns: dict_keys(['MDFP', 'MDFP_D', 'MDFP+', ... ]).  

    To print the features composing, for example, the MDFP+ descriptor, type in the python shell    MDFP_terms.descriptor_dictionary['MDFP+']
    To print the corresponding colors, type: MDFP_terms.colors_dictionary['MDFP+']

    The functions defined in the SelectDescriptors class are useful to extract the terms of each of the descriptors listed in a list of descriptor names. See the documentation of SelectDescriptors.
    """
    # Define Descriptors
    mdfp_2d_counts_terms = ['HA_count', 'RB_count', 'N_count', 'O_count', 'F_count', 'P_count', 'S_count', 'Cl_count', 'Br_count', 'I_count']
    mdfp_intra_terms = ['intra_ene_av_wat','intra_ene_std_wat','intra_ene_med_wat','intra_lj_av_wat', 'intra_lj_std_wat', 'intra_lj_med_wat', 'intra_crf_av_wat', 'intra_crf_std_wat', 'intra_crf_med_wat']
    mdfp_tot_terms = ['total_ene_av_wat','total_ene_std_wat','total_ene_med_wat','total_lj_av_wat', 'total_lj_std_wat', 'total_lj_med_wat', 'total_crf_av_wat', 'total_crf_std_wat', 'total_crf_med_wat']
    mdfp_other_terms = ['wat_rgyr_av', 'wat_rgyr_std','wat_rgyr_med','wat_sasa_av', 'wat_sasa_std', 'wat_sasa_med']
    psa_2d = ['2d_psa']
    psa_terms = ['3d_psa_av', '3d_psa_sd', '3d_psa_med']
    psa_pc_terms = ['3d_psa_pc_av', '3d_psa_pc_sd', '3d_psa_pc_med']
    dipole_mom_terms = ['av_mu_x', 'av_mu_y', 'av_mu_z', 'std_mu_x', 'std_mu_y', 'std_mu_z', 'med_mu_x', 'med_mu_y', 'med_mu_z', 'av_mu', 'std_mu', 'med_mu']
    other_2d_counts_terms = ['MW', 'HBD_count', 'HBA_count', 'is_zwit', '2d_shape']
    mdfp = mdfp_intra_terms + mdfp_tot_terms + mdfp_other_terms   #1
    mdfp_dipole = mdfp + dipole_mom_terms                                 #2
    mdfp_plus = mdfp + mdfp_2d_counts_terms				  #3
    mdfp_p2 = mdfp + psa_2d						  #4 
    mdfp_p3 = mdfp + psa_terms						  #5
    mdfp_pp = mdfp + psa_pc_terms					  #6
    mdfp_plus_plus = mdfp_plus + other_2d_counts_terms			  #7
    mdfp_p2_plus_plus = mdfp_plus + other_2d_counts_terms + psa_2d	  #8
    mdfp_p3_plus_plus = mdfp_plus + other_2d_counts_terms + psa_terms	  #9
    counts_2d = mdfp_2d_counts_terms					  #10
    counts_extra = other_2d_counts_terms				  #11
    counts_all = mdfp_2d_counts_terms + other_2d_counts_terms		  #12
    counts_prop_2d = counts_all + psa_2d				  #13
    dipole_mom = dipole_mom_terms					  #14
    ECFP4 = ['ECFP4']							  #15
    rdkitfp = ['RDKitFP']                                                 #16
    rdkit2d = DataPrep.get_RDKit2D_colnames()                             #17
    mdfp_rdkit2d = mdfp + rdkit2d
    # Extract Scalable Features
    scalable_features = mdfp_plus_plus + psa_2d + psa_terms + psa_pc_terms + dipole_mom_terms + ['RDKit2D'] 
 
    ECFP4_combi = [counts_all, psa_2d, counts_all + psa_2d, counts_all + psa_terms, mdfp, mdfp + psa_terms, mdfp_p3_plus_plus]
    ECFP4_combi_names = ['ECFP4++', 'ECFP4_P2', 'ECFP4_P2++', 'ECFP4_P3++', 'ECFP4_MDFP', 'ECFP4_MDFP_P3', 'ECFP4_MDFP_P3++']

    # Dictionary of Descriptors
    descriptor_dictionary = {'MDFP': mdfp, 'MDFP_D': mdfp_dipole, 'MDFP+': mdfp_plus, 'MDFP_P2': mdfp_p2, 'MDFP_P3': mdfp_p3, 'MDFP_PP': mdfp_pp, 'MDFP++': mdfp_plus_plus, 'MDFP_P2++': mdfp_p2_plus_plus, 'MDFP_P3++': mdfp_p3_plus_plus, 'C2D': counts_2d, 'P2D': counts_extra, 'CP2D': counts_all, 'CP2D_P2': counts_prop_2d, 'DIP_MOM': dipole_mom, 'RDKitFP': rdkitfp, 'RDKit2D': rdkit2d, 'MDFP_RDKit2D': mdfp_rdkit2d, 'ECFP4': ECFP4, 'ECFP4++': counts_all, 'ECFP4_P2': psa_2d, 'ECFP4_P2++': counts_all + psa_2d, 'ECFP4_P3++': counts_all + psa_terms, 'ECFP4_MDFP': mdfp, 'ECFP4_MDFP_P3': mdfp + psa_terms, 'ECFP4_MDFP_P3++': mdfp_p3_plus_plus, 'ECFP4_RDKit2D': ECFP4 + rdkit2d, 'ECFP4_MDFP_RDKit2D': ECFP4 + mdfp + rdkit2d}

    # Colors Features for Feature importance plots
    col_list = ['cornflowerblue', 'black','silver','mediumorchid','darkorange', 'brown','b','lightseagreen']  #counts, ene, lj+crf, rgyr, sasa, dipole, counts2, psa
    col_rdkit2d = ['gray'] * len(rdkit2d)
    col_dipole_mom = list(itertools.repeat(col_list[5],len(dipole_mom_terms)))
    col_counts_2d = list(itertools.repeat(col_list[0], len(counts_2d)))
    col_counts_extra = list(itertools.repeat(col_list[6],len(other_2d_counts_terms)))
    col_psa_2d = ['teal']
    col_psa_3d = list(itertools.repeat(col_list[7], 3))
    col_psa_3d_pc = list(itertools.repeat(col_list[7], 3))
    col_mdfp = list(itertools.repeat(col_list[1], 3 )) + list(itertools.repeat(col_list[2], 6)) + list(itertools.repeat(col_list[1], 3)) + list(itertools.repeat(col_list[2], 6)) + list(itertools.repeat(col_list[3],3)) + list(itertools.repeat(col_list[4], 3))
    col_mdfp_dipole = col_mdfp + col_dipole_mom
    col_mdfp_plus = col_mdfp + col_counts_2d
    col_mdfp_p2 = col_mdfp + col_psa_2d
    col_mdfp_p3 = col_mdfp + col_psa_3d
    col_mdfp_pp = col_mdfp + col_psa_3d_pc
    col_mdfp_plus_plus = col_mdfp_plus + col_counts_extra
    col_mdfp_p2_plus_plus = col_mdfp_plus + col_counts_extra + col_psa_2d
    col_mdfp_p3_plus_plus = col_mdfp_plus + col_counts_extra + col_psa_3d
    col_counts_all = col_counts_2d + col_counts_extra
    counts_prop_2d = col_counts_all + col_psa_2d
    col_mdfp_rdkit2d = col_mdfp + col_rdkit2d
    mylist_colors = [col_mdfp, col_mdfp_dipole, col_mdfp_plus, col_mdfp_p2, col_mdfp_p3, col_mdfp_pp, col_mdfp_plus_plus, col_mdfp_p2_plus_plus, col_mdfp_p3_plus_plus, col_counts_2d, col_counts_extra, col_counts_all, counts_prop_2d, col_dipole_mom] 
 
    colors_dictionary = {'MDFP': col_mdfp, 'MDFP_D': col_mdfp_dipole, 'MDFP+': col_mdfp_plus, 'MDFP_P2': col_mdfp_p2, 'MDFP_P3': col_mdfp_p3, 'MDFP_PP': col_mdfp_pp, 'MDFP++': col_mdfp_plus_plus, 'MDFP_P2++': col_mdfp_p2_plus_plus, 'MDFP_P3++': col_mdfp_p3_plus_plus, 'C2D': col_counts_2d, 'P2D': col_counts_extra, 'CP2D': col_counts_all, 'CP2D_P2': counts_prop_2d, 'DIP_MOM': col_dipole_mom, 'RDKit2D': col_rdkit2d, 'MDFP_RDKit2D': col_mdfp_rdkit2d}



class SelectDescriptors:
    """ Functions to extract the terms of each of the descriptors specified in a list of descriptor names.
    Three functions are available: MDFPFromList, ECFP4CombiFromList, and RDKitFPsFromList.

    To check all available descriptors, type in the python shell    MDFP_terms.descriptor_dictionary.keys()
    See also the documentation of MDFP_terms.

    For example, if from all available descriptors, you want to extract only the following descriptors:
    descriptor_list = ["MDFP", "MDFP+", "MDFP_P3", "ECFP4_MDFP", "MDFP_RDKit2D"]

    MDFPFromList returns a dictionary only for the MDFPs, whose keys are the descriptor names and items are the lists of features
    MDFPs_dict = SelectDescriptors.MDFPFromList(descriptor_list)
 
    ECFP4CombiFromList returns a dictionary only for the descriptors that are a combination of ECFP4 with MDFPs. Keys are the descriptor names and items are the lists of features of the MDFP component.
    ECFP4Combi_dict = SelectDescriptors.ECFP4CombiFromList(descriptor_list)

    RDKitFPsFromList returns a dictionary only for the RDKit fingerprints (RDKit2D and RDKitFP) and combination of those.
    RDKitFPs_dict = SelectDescriptors.RDKitFPsFromList(descriptor_list)
    """
    def __init__(self):
        pass

    @classmethod
    def MDFPFromList(self, descriptor_list):
        """
        It returns a dictionary only for the MDFPs, whose keys are the descriptor names and items are the lists of features.
        Usage: MDFPs_dict = SelectDescriptors.MDFPFromList(descriptor_list)

        
        Parameters:
        -----------
        descriptor_list: list
            list of descriptor names

        Returns:
        ----------
        selected_mdfp_descriptors: dict
            dictionary of descriptors. Keys are the descriptor names and items are the lists of the corresponding features. 
        """
        mdfp_list = [x for x in descriptor_list if ("RDKit" not in x and "ECFP4" not in x)]

        selected_mdfp_descriptors = {}
        if len(mdfp_list) != 0:
            for k in mdfp_list:
                selected_mdfp_descriptors[k] = MDFP_terms.descriptor_dictionary[k]

        return(selected_mdfp_descriptors)


    @classmethod
    def ECFP4CombiFromList(self, descriptor_list):
        """
        It returns a dictionary only for the descriptors that are a combination of ECFP4 with MDFPs. Keys are the descriptor names and items are the lists of features of the MDFP component.
        Usage: ECFP4Combi_dict = SelectDescriptors.ECFP4CombiFromList(descriptor_list)

        
        Parameters:
        -----------
        descriptor_list: list
            list of descriptor names

        Returns:
        ----------
        selected_mdfp_descriptors: dict
            dictionary of descriptors. Keys are the descriptor names and items are the lists of the corresponding MDFP features. 
        """
        
        ECFP4_list = [x for x in descriptor_list if ("ECFP4" in x and "RDKit" not in x)]

        selected_ECFP4_descriptors = {}
        if len(ECFP4_list) != 0:
            for k in ECFP4_list:
                selected_ECFP4_descriptors[k] = MDFP_terms.descriptor_dictionary[k]

        if 'ECFP4' in selected_ECFP4_descriptors:
            del selected_ECFP4_descriptors['ECFP4']

        return(selected_ECFP4_descriptors)


    @classmethod
    def RDKitFPsFromList(self, descriptor_list):
        """
        It returns a dictionary only for the RDKit fingerprints (RDKit2D and RDKitFP) and combination of those. Keys are the descriptor names and items are the lists of the corresponding features.
        Usage: RDKitFPs_dict = SelectDescriptors.RDKitFPsFromList(descriptor_list)

        
        Parameters:
        -----------
        descriptor_list: list
            list of descriptor names

        Returns:
        ----------
        selected_mdfp_descriptors: dict
            dictionary of descriptors. Keys are the descriptor names and items are the lists of the corresponding features. 
        """

        rdkit_list = [x for x in descriptor_list if "RDKit" in x ]

        selected_rdkit_descriptors = {}
        if len(rdkit_list) != 0:
            for k in rdkit_list:
                selected_rdkit_descriptors[k] = MDFP_terms.descriptor_dictionary[k]

        return(selected_rdkit_descriptors)



class Read_data:
    """ Class containing functions useful to read in dataframes of the datasets and to obtain a list of ECFP4 fingerprints"""

    def __init__(self):
        pass

    @classmethod
    def read_data_from_dataframe(self, filename, labels_column = None):
        """ Read pkl file of the dataset into pandas dataframe.
        If labels_column is not specified than only the dataframe is returned.
        If labels_column is specified than a list of class labels or target values is also returned.

        Usage: 
        data = Read_data.read_data_from_dataframe("dataset.pkl")
        data, labels = Read_data.read_data_from_dataframe("dataset.pkl", labels_column = "is_sub")

        Parameters:
        -----------
        filename: str
            name of the file (.pkl) containing the dataset
        labels_column: str, optional
            name of the column containing the labels 
        
        Returns:
        ----------
        df_dataset: df
            dataframe of the dataset
        list_labels: list, optional
            list of labels of the dataset. Only returned if labels_column is specified
        """
        df_dataset = pd.read_pickle(filename)
        df_dataset.reset_index(inplace = True, drop = True)
        if labels_column != None and labels_column in list(df_dataset):
            list_labels = list(df_dataset[labels_column])
            return(df_dataset, list_labels)
        else:
            return(df_dataset)

    @classmethod
    def get_ECFP4(self, df_dataset, smiles_column = "smiles", ecfp4_column = "ECFP4"):
        """ It returns a list of ECFP4 descriptors. The input dataframe has to contain either a smiles_column or already a column of ECFP4 descriptors.
        Usage: ecfp4_list = Read_data.get_ECFP4(df_dataset, smiles_column = "smiles")

        Parameters:
        -----------
        df_dataset: df
            dataframe of the dataset
        smiles_column: str, optional
            name of the column containing the SMILES (Default = "smiles")
        ecfp4_column: str, optional
            name of the column containing ECFP4 fingerprints. If not specified, ECFP4 are computed from SMILES

        Returns:
        ----------
        ecfp4_list: list
            list of ECFP4 descriptors (arrays). 
        """

        if ecfp4_column in list(df_dataset):
            return(list(df_dataset[ecfp4_column]))
        elif smiles_column in list(df_dataset):
            df_dataset = DataPrep.add_ECFP4(df_dataset)
            return(list(df_dataset[ecfp4_column]))
        else: 
            print('Warning: ECFP4 could not be read or generated. Ensure that either a ECFP4 column or a smiles smiles column is contained in the dataset')
 

