# DL_FDI

This repository contains the codes used for the main deep learning model of my Master's thesis. The deep learning model is used to detect FDI attacks of various sparsity. It also contains the codes, logs, and results of various models tested in the process.

## DATA

The data is generated using MATLAB and the matpower library following the mathematical identities of False Data Injection (FDI) attacks. The features in the data are measurements including voltages and voltage phase angles, and generator power consumption.

## RUNNING THE EXPERIMENT

Three basic classifiers are tested in classifiersTest.py

Neural networks architectures are tested in main.py and main_tester.py and the models are defined in ModelsFinal.py

The remainder of the notebooks/scripts are for data processing and individualized testings.
