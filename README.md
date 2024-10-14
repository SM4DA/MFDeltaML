# MFDeltaML and Data Efficiency Benchmarks of Multifidelity Methods for Machine Learning 
Scripts for MFDeltaML and data efficiency benchmarks of DeltaML, MFML, o-MFML, MFDeltaML, o-MFDeltaML on the QeMFi dataset.


The scripts provided in this code repository can be used to reprodce the results of the manuscript titled 'Benchmarking Data Efficiency in Delta-ML and Multifidelity Models for Quantum Chemistry' available as a preprint at [TBA]. The python library requirements can be found in `requirements.txt` in this code repository.
The data used in this work is the QeMFi dataset which can be accessed at the following URL: [https://zenodo.org/records/12734761](https://doi.org/10.5281/zenodo.13925688)

* `Model_MFML.py` is the main python module required to run the MFML and o-MFML models.
* `SF_DeltaML.py` generates the Delta ML learning curves for different QC-baseline.
* `MFML_LCsAll.py` generates the data for MFML and o-MFML learning curves.
* `MFDeltaML_LCs.py` generates the data for the MFDeltaML and o-MFDeltaML models, two new methods introduced in this work.
* `PredictLowest.py` generates the data required for Fig.5 from the manuscript. It generates a hybrid DeltaML model where the baseline is not a QC computed one but is rather predicted using a single fidelity ML model.
* The jupyter notebook `DeltaMFML.ipnyb` contains all the plotting routines and the data split (into tes, train, and validation) codes.
