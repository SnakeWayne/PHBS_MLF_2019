# PHBS_MLF_2019  - Final Project
phbs 2020 spring module3 machine learning

## Group Information

|Name|Student ID|
|:---|:---|
|[Shen Tingwei](https://github.com/SnakeWayne)|1901212631|
|[He Junxian](https://github.com/hejunxian256)|1901212583 |
|[Yu Xinlin](https://github.com/398563924)|1901212664|
|[Wang Zixiong](https://github.com/WangZixiong)|1901212644|

## Introduction

Our project is going to do stock selection implementing the machine learning methods. Considering using the pure trading information performance not that well, we introduce the sentiment data borrowed from [chinascope](http://finance.chinascope.com/www/) based on the news. We prepear to generate some relating features to boost the accuracy.

## Data set 
You can find the sample data in the **data** folder

### Data Summary
The sentiment data conclude from one article review the emotion torward a certain company or person where **emotionIndicator** is set to three raw values:*1 for positive, 0 for neutral, 2 for negative*. And it choose the indicator with the highest weight to be its overall sentiment. ChinaScope has clame that this classifaction accuracy is above 80%.

And other trading data is provided by the api from [joinquant](https://www.joinquant.com/), inclduing the basic ohlc data and others.






## Training
We built 9 features, including 8 other features and 1 sentiment feature.The response is 0 and 1, representing fall and rise of stock price. In order to find out the contribution of the sentiment feature, we built 3 kinds of traing set: 1) only contains 8 other features; 2) contains 8 other features and sentiment feature; 3) only contains sentiment feature.

We train on 70% of the sample and test on 30% of the sample. To increase training speed, the data are standardized. And we drop out 4 lines that contain NaN.

We applied LR, SVC, TREE methods to train. In each methods, we also applied grid search to find the best hyperparameters. We compared the results mainly through confusion matrics. 

### 1. Logistic Regression

#### Only contain other features:
The best parameter is C=1.0

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/LR%201.jpg)
Accuracy is 52.8%; PRE is 51.0%
#### Contain other features and sentiment features:
The best parameter is C=0.001

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/LR%202.jpg)
Accuracy is 54.7%; PRE is 53.8%
#### Only contain sentiment features:
The best parameter is C=0.01

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/LR%203.jpg)
Accuracy is 52.2%; PRE is 51.7%
### 2. Support Vector Machine

#### Only contain other features:
The best parameter is {'svc__C': 100.0, 'svc__gamma': 0.1, 'svc__kernel': 'rbf'}

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/SVC%201.jpg)
Accuracy is 59.9%; PRE is 57.3%
#### Contain other features and sentiment feature:
The best parameter is {'svc__C': 1.0, 'svc__gamma': 1.0, 'svc__kernel': 'rbf'}

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/SVC%202.jpg)
Accuracy is 59.2%; PRE is 57.6%
#### Only contain sentiment feature:
The best parameter is {'svc__C': 0.1, 'svc__gamma': 10.0, 'svc__kernel': 'rbf'}

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/SVC%203.jpg)
Accuracy is 52.2%; PRE is 50.6%
### 3. Decision Tree

#### Only contain other features:
The best parameter is max_depth=6

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/Tree%201.jpg)
Accuracy is 56.7%; PRE is 53.9%;
#### Contain other features and sentiment features:
The best parameter is max_depth=6

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/Tree%202.jpg)
Accuracy is 57.0%; PRE is 54.4%
#### Only contain sentiment features:
The best parameter is max_depth=3

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/Tree%203.jpg)
Accuracy is 52.7%; PRE is 52.7%

### Conclusion:
Considering that people's decision rely on the prediction result. People will buy if predict 1, people will not act if predict 0. If prediction is 1 but it's 0 in reality, people will suffer financial loss. Consequently, we want TP high and FP low. We need to measure the result based on Precision (PRE=TP/(TP+FP)). The higher PRE rate, the more profit people gain.

From the result of PRE, we find that combining other features and sentiment feature using SVC method can get the best result. The best kernel is 'rbf', so it may not perform well in linear classification. 
