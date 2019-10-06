# Installation

This package only supports Python 3.5 or above.

The MDFPtools packages is built on various packages that is conda-installable. Therefore, either anaconda or miniconda is needed as prerequisites.

Once conda is available, execute the following commands to install all essential dependencies:

```bash
conda install -c rdkit rdkit
conda install -c omnia openforcefield, openmm, mdtraj, parmed
conda install numpy
```

*If one wishes to use the commercial software OpenEye install of RDKit during the parameterisation of the systems, then the relevant OpenEye modules need to be installed, [follow detail here.](https://docs.eyesopen.com/toolkits/python/quickstart-python/install.html)*

After all dependencies are installed, navigate to a directory where you wish to clone this repository and execute:
```
git clone https://github.com/rinikerlab/mdfptools.git
cd mdfptools
python setup.py install
```

To use the machine-learned partial charged developed in our lab during the parameterisation stage, install separately the [mlddec package](github.com/rinikierlab/mlddec).
