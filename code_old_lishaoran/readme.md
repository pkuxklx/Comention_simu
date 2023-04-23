# Covariance Estimation with Network Information

- [Covariance Estimation with Network Information](#covariance-estimation-with-network-information)
  - [Workflow](#workflow)
  - [Raw Data](#raw-data)
    - [GVKeys and PERMNO Linking Table](#gvkeys-and-permno-linking-table)
    - [Hoberg's similarity score](#hobergs-similarity-score)
    - [Factor Data](#factor-data)
  - [Cleaned data](#cleaned-data)
    - [GVKEY-PERMNO and factor data](#gvkey-permno-and-factor-data)
    - [Return-Network](#return-network)
  - [Construction of Usable Dataset](#construction-of-usable-dataset)
    - [Adaptive Correlation Thresholding](#adaptive-correlation-thresholding)
    - [Adaptive Thresholding](#adaptive-thresholding)
    - [Network Banding](#network-banding)
  - [Experiments](#experiments)
    - [Hoberg With SP500](#hoberg-with-sp500)
    - [Analyst SP500](#analyst-sp500)
- [TODO](#todo)

These files estimate covariances based on observations $X_t$ along with some information about the support $Z_{ij}$.

The first major case we consider is on SP500 stocks. 



## Workflow

1. Collect [raw return and network data](#raw-data). 
2. Initial cleaning: handling NaN, 'B', 'C'; match IDs, saved as pandas dataframes in .p files 
    1. 统一格式. 

    
    1. Details can be found [here](#return-network)
3. Second cleaning: based on the experiment, extract relevant information
4. Calculation: using the following [methods](#methods):
    1. Adaptive correlation thresholding
    2. Hard thresholding
    3. Adaptive (variance) thresholding

## Raw Data



### GVKeys and PERMNO Linking Table

The source file is [linking_information.csv](data/raw/Hoberg/linking_information.csv)



### Hoberg's similarity score

- **tnic3_data.txt** contains the Hoberg network scores. It has 13808 unique GVKeys which can be expanded into a symmetric network. The score is truncated slightly above 0(for companies to be considered as linked). Downloaded from Prof. Hoberg's website
### Factor Data

- **factors_2021-01-13** contains data on the factors and risk-free rates (mktrf,smb,hml,rf,umd) from 1990-01-02 to 2019-12-31. Downloaded from CRSP. 


## Cleaned data 

Location at data/clean/. 


### GVKEY-PERMNO and factor data

- Stored in [link.p]() and [factor.p]()

Factors
```
| date                |   mktrf |     smb |     hml |      rf |     umd |
|:--------------------|--------:|--------:|--------:|--------:|--------:|
| 1990-01-02 00:00:00 |  0.0144 | -0.0068 | -0.0006 | 0.00026 | -0.0108 |
| 1990-01-03 00:00:00 | -0.0006 |  0.0075 | -0.0031 | 0.00026 | -0.0034 |
| 2019-12-30 00:00:00 | -0.0057 |  0.0017 |  0.006  | 7e-05   |  0.0004 |
| 2019-12-31 00:00:00 |  0.0028 | -0.0001 |  0.0013 | 7e-05   | -0.0048 |
```

Linking information 

```
|       |   GVKEY |   PERMNO |
|------:|--------:|---------:|
|     0 |    1000 |    25881 |
|     1 |    1001 |    10015 |
| 29784 |  331856 |    14615 |
| 29785 |  332115 |    80577 |
```

### Return-Network

| Network \ Return     | <mark>Analyst co-cov ret</mark> | SP500 | All stocks |
| ------------- | --------------------------------- | ----- | ---------- |
| Hoberg        | In process | [hoberg_sp500.p](#hoberg-sp500-gvkey) |[hoberg_all_gvkey]()
| Analyst cocov |  |analyst_sp500.p
| bwlinks       |

## Construction of Usable Dataset 

In addition to the variables <span style="color:red">HP, factor, ret, net</span>, we will **construct** 
- a rolling window of returns; 
- network corresponding to stocks in each window
- and a list of RetDF models wich contains attributes like factor cov, residual covariance, etc. 

### Adaptive Correlation Thresholding

### Adaptive Thresholding
 
### Network Banding

Use network information to guide a similar-to-banding method, where if L_ij = 1, we do nothing, and if L_ij = 0, we apply a soft thresholding. 

## Experiments

### Hoberg With SP500

### Analyst SP500

- [ ] 这些coverage mention次数多久统计一次 
- [ ] index map 是否正确 

# TODO

- [ ] Streamline data, save as pandas.df and then convert to tensor. 
- [ ] move the get_G method from utils to utils.constructor. 
- [ ] 错误observation的概率