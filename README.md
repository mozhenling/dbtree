# DBTREE Python Package for Machinery Fault Envolope Analysis  

This repository provides the proposed dynamic bandit tree search method together with popular filters and various fault indexes to facilitate future studies of envelope analysis-based machinery fault diagnosis. More methods may be added as well in the future.

The codes are mainly based on the paper entitled _" Intelligent Fault Informative Frequency Band Search for Machinery Fault Diagnosis Assisted by A Dynamic Bandit Tree Method"_, which has been accepted for publication in the **IEEE/ASME Transactions on Mechatronics** ([Preprint](https://www.researchgate.net/publication/363469478_Intelligent_Informative_Frequency_Band_Searching_Assisted_by_A_Dynamic_Bandit_Tree_Method_for_Machine_Fault_Diagnosis), [Early Access](https://ieeexplore.ieee.org/document/9900070)). 

## 1 Key Requirements
    numpy = 1.21.5
    matplotlib = 3.5.1
    tqdm = 4.46.0
    scipy = 1.7.3
    torch = 1.7.1 (for pytorch svmd only)

## 2 Current Methods

### 1) Bandit Optimization Algorithms (dbtpy->dbtrees)
    Currently, the bandit optimization algorithms include
    (1) Multi tree bandit  algorithms (DBT2 and DBT3, multi_tree.py)
    (2) Single tree bandit algorithms (MCT and DBT1, single_tree.py)
Note that the diagnosis methods provided here can also be optimized by other optimization algorithms. On the other hand, the optimization algorithms of this part can be applied to other applications as well.

### 2) Popular Filters (dbtpy->filters)

    Currently, the filters include: 
    (1) variational model decomposition
    (2) successive variational model decomposition
    (3) empricial wavelet transform (Meyer filters)
    (4) 1/3-binary tree (fast kurtogram and its variants).  

### 3) Fault Indexes (dbtpy->findexes)
    Currently, the fault indexes include:
    (1) Kurtosis and its variants
    (2) Smoothness index
    (3) Gini index and its variants
    (4) pq_norm family
    (5) Hoyer measure
    (6) Entropy and negentropy
    (7) Fault harmonic indexes 
        ( such as vanillaSNR, 
        cyclic harmonic to noise ratio,
        harmonic L2/L1 norm, 
        and harmonic kurtosis)
Many of the indexes can be easily used to replace the kurtosis in fast kurtogram, resulting in the so-called fault-indexgram (findexgram) method. Changing the fault index and the calcuation domain is simply a matter of one line of codes. The calculation domains include the envolope, squared envolope, squared envolope spectrum, etc. In addition, the fault indexes can be utilized as the fitness function of the optimization algorithms. Finally, common test functions are also provided in fun_2dim.py and fun_ndim.py.

### 4) Useful Tools (dbtpy->tools)
    Currently, the useful tools include
    (1) Preprocessing tools (prepro_tools.py): 
        detrend, 
        pre-whiten,
        fault characteristic frequency calculations of bearing and gear. 
    (2) Time/ frequency tools (time_freq_tools.py): 
        plot different styles of figures of signal in time and frequency domains.
    (3) File tools (file_tools.py): 
        save and read important contents based on txt files. 
    (4) Visualization tools  (visual_tools.py): 
        visualize the demodulation spectrum, 
        the heatmap of findexgram,
        the fitness curve,
        the tree structure, etc.. 
	(5) Angualr resampling (order_tracking.py)
    (6) Some other tools. 

## 3 Tutorials

-For optimization problems, see [tutorial_dbt_optim.ipynb](https://github.com/mozhenling/dbtree/blob/master/tutorial_dbt_optim.ipynb)  
-For fault diagnosis based on bandit algorithms, see [tutorial_dbt_diag.ipynb](https://github.com/mozhenling/dbtree/blob/master/tutorial_dbt_diag.ipynb)  
-For fault diagnosis based on findexgram, see [tutorial_findexgram_diag.ipynb](https://github.com/mozhenling/dbtree/blob/master/tutorial_findexgram_diag.ipynb)

More tutorials may be provided in the Jupyter notebook files later. 


## 4 Citation 

We would really appreciate it if you could cite our paper once the repo is used.   

    @article{mo2022intelligent,
    title={Intelligent informative frequency band searching assisted by a dynamic bandit tree method for machine fault diagnosis},
    author={Mo, Zhenling and Zhang, Zijun and Miao, Qiang and Tsui, Kwok-Leung},
    journal={IEEE/ASME Transactions on Mechatronics},
    volume={28},
    number={2},
    pages={770--780},
    year={2022},
    publisher={IEEE}
    }

