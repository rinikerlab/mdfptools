MDFPtools
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/hjuinj/mdfptools.png)](https://travis-ci.org/hjuinj/mdfptools)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/mdfptools/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/mdfptools/branch/master)


This is a Python implementation of the molecular dynamics fingerprints (MDFP) methodology for building predictive models for phys-chem properties as delineated in [our publication](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00778).

This toolkit is described and applied to the [SAMPL6 logP prediction challenge](https://drugdesigndata.org/about/sampl6/logp-prediction), with the results published [here](https://doi.org/10.1007/s10822-019-00252-6).

Visit our [documentation](https://mdfptools.readthedocs.io/en/latest/) to learn details about [installation](https://mdfptools.readthedocs.io/en/latest/install.html), [example workflow](https://mdfptools.readthedocs.io/en/latest/tutorial.html) and [API references](https://mdfptools.readthedocs.io/en/latest/parameterisers.html).


The openeye entry is the basic version. The rdkit handler version allow more customisation, e.g. specifying custom conformer for the molecule

### Citations
Bibtex citations for the toolkit and the method are as follows:
```
@article{esposito2020combining,
  title={Combining Machine Learning and Molecular Dynamics to Predict P-Glycoprotein Substrates},
  author={Esposito, Carmen and Wang, Shuzhe and Lange, Udo EW and Oellien, Frank and Riniker, Sereina},
  journal={Journal of Chemical Information and Modeling},
  volume={60},
  number={10},
  pages={4730--4749},
  year={2020},
  journal = {Journal of Chemical Information and Modeling}
}

@article{Wang2019,
  doi = {10.1007/s10822-019-00252-6},
  url = {https://doi.org/10.1007/s10822-019-00252-6},
  year = {2019},
  month = nov,
  publisher = {Springer Science and Business Media {LLC}},
  author = {Shuzhe Wang and Sereina Riniker},
  title = {Use of molecular dynamics fingerprints ({MDFPs}) in {SAMPL}6 octanol{\textendash}water log P blind challenge},
  journal = {Journal of Computer-Aided Molecular Design}
}

@article{Riniker2017,
  doi = {10.1021/acs.jcim.6b00778},
  url = {https://doi.org/10.1021/acs.jcim.6b00778},
  year = {2017},
  month = apr,
  publisher = {American Chemical Society ({ACS})},
  volume = {57},
  number = {4},
  pages = {726--741},
  author = {Sereina Riniker},
  title = {Molecular Dynamics Fingerprints ({MDFP}): Machine Learning from {MD} Data To Predict Free-Energy Differences},
  journal = {Journal of Chemical Information and Modeling}
}
```

### Copyright

Copyright (c) 2018, Shuzhe Wang, Carmen Esposito


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms)
