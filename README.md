# GEP-ECOC

Genetic Expression Programming - Error Correcting Output Codes

This is the implementation for paper: [An Adaptive Error-Correcting Output Codes Algorithm Based on Gene Expression Programming and Similarity Measurement Matrix]
## Acknowledgement

- Codes about Genetic Expression Programming is modified from  [DEAP](<https://github.com/DEAP/deap>)
- Codes about Genetic Expression Programming is modified from  [geppy](<https://github.com/ShuhuaGao/geppy>)
- Codes about Base Classifiers are modified from  [scikit-learn](<https://github.com/scikit-learn/scikit-learn>)

## Environment

- **Windows 10 64 bit** 

- **python 3.7**

- **scikit-learn 1.0.1**

  Anaconda is strongly recommended, run the following command in the Powershell, all necessary python packages for this project will be installed:

  ```shell
  conda install -r requirements.txt
  ```


## Dataset Module

- **Data format**

  Raw data is put into the folder ```($root_path)/Dataset/```.
  Each dataset should be divided into ```train_data, test_data, validation_data with ratio of 2:1:1```.   
  In each dataset, each row is a sample, the last column represents the labels, and the rest are the feature space.
  Please note that invalid sample, such as value missed, will cause errors.
  
  There are two datasets ```dermatology``` and ```balance``` in the folder ```($root_path)/Dataset/``` as an example. 
  Dataset information:
    name: dermatology
    class num: 4
    feature num: 34
    sample num: 366 
    name: balance
    class num: 3
    feature num: 8
    sample num: 625

- **Data processing**

  Feature Selection and Scaling will be done automatically. 


## Algorithmic Running Module

- **Configuration**

  The configuration starts at line 133 in ```main.py```. 
  Firstly, set the number of populations ```n_pop``` and the number of iterations ```n_gen```, then choose the name of the data ```datafile```, and finally choose the base classifier ```estimators_type``` needed for ECOC.
  Note: ```datafile``` and ```estimators_type``` are both list types, so you can add the data or base classifiers you want to experiment with as needed.

- **Run the following command**

  It will traversal all datasets given by the main function in ```main.py```, each dataset will be run for 10 times.
- 
  ```shell
  python main.py
  ```

## Result Module

- **Record result**

  All result infos will be written into the folder. And the result infos will be found in ```($root_path)/Logging/```

  The iteration graphs generated when the algorithm runs will be saved in the ```($root_path)/Iteration_Chart/```.


- **Analyze result**

   In this Mode, part of the result will be printed on the terminal. You can find all result in the Results folder.
   But, there will be no automatic analyzing. 

# GEP_ECOC_test
