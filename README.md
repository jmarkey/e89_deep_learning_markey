# e89_deep_learning_markey

## Course Description
This repository is created as a project for a Deep Learning Course.  The following documentation provides more details on the class assignment.

## Problem Statement
This project focuses on investigating approaches to detect fraudulent activity in financial transactions, using a simulated data available in Kaggle.  The project uses Deep Learning techniques (specifically Neural Networks) to detect fraudulent activity and compares these techniques to more Shallow Learning approaches.

## Data Set Examined
For this project, a synthetic dataset is used, as collected from Kaggle.  This dataset contains simulated financial transaction data based on a real dataset.  The Kaggle dataset can be downloaded here: https://www.kaggle.com/ntnu-testimon/paysim1/version/2#.
•	Large size: 471 MB, Small size: 8.4 MB, Data Format: csv

## Experiment Details
This experiment performs a grid search with a binary neural network classifier built using Keras.  It adjusts the batch size, the number of epochs, the number of units in each layer or the network and the L2 regularization weights to search for an optimal classification approach. Grid-searching for hyper-parameters has the advantages that it provides a thorough analysis of available data, however this approach is computationally expensive.
•	Hardware: Mac OS Sierra on 2.8 GHz Intel Core i7, 16GB Ram
•	Software: Python 3.6, PyCharm, Github

## Lessons Learned & Pros/Cons
•	Fraud datasets are difficult to find and obtain
•	Data transformations had a large effect on the accuracy of the model trained

## References and Acknowledgements
The data used in this experiment is part of the research project ”Scalable resource-efficient systems for big data analytics” funded by the Knowledge Foundation (grant: 20140032) in Sweden.  The researchers created this data based on original logs provided by a multi-national mobile financial service company.  Details are listed below:
1.	E. A. Lopez-Rojas , A. Elmir, and S. Axelsson. "PaySim: A financial mobile money simulator for fraud detection". In: The 28th European Modeling and Simulation Symposium-EMSS, Larnaca, Cyprus. 2016
Links

## Other Resources
•	Two Minute Summary Video: https://youtu.be/hJfso7HdHVI
•	Detailed Summary Video: https://youtu.be/ZjiFSEaqp_w
