# Group 2: Stock selection with ChinaScope sentiment data 
phbs 2020 spring module3 machine learning Final Project 

## Group Information

|Name|Student ID|
|:---|:---|
|[Shen Tingwei](https://github.com/SnakeWayne)|1901212631|
|[He Junxian](https://github.com/hejunxian256)|1901212583 |
|[Yu Xinlin](https://github.com/398563924)|1901212664|
|[Wang Zixiong](https://github.com/WangZixiong)|1901212644|

## Introduction

Machine learning methods are often applied in stock selection in quantitative finance, based on trading data. However, market dynamics are also greatly influenced by people's sentiment, which makes stock selection methods using pure trading information less accurate. 

Therefore, the purpose of our project is to introduce market sentiment data to stock selection with machine learning algorithms.
The sentiment data we use in this project is from [ChinaScope](http://finance.chinascope.com/www/), a financial database. We try to generate an effective feature that can reflect the dynamics of market sentiment in order to boost the accuracy in stock selection.

## Data set 
Sample data can be found in the **data** folder of our project repository.

### Data Summary
The sentiment dataset includes a number of news articles and every article is related to a certain listed company. Corresponding to each piece of news, the dataset includes the emotion regarding to that company. For a specific piece of news, the emotional indicator with the highest weight will be its overall sentiment. 

Along with the sentiment data, we also use traditional trading data, which can be obtained through Python api of [joinquant](https://www.joinquant.com/).


## Data Preparation
### Sentiment feature
#### How can we extract a useable feature from the sentiment data?
Below are the steps of extracting a sentiment feature:

1. We extract sentiment data of A-share stocks from rawdata. After preprocessing the sentiment dataset, we find that each news can effect many stocks in different ways. We extract each news article information for one company in one day as a sample. 

2. Each sample includes three elements: __sentiment type__, __sentiment weight__ and __relevance__.

* __sentiment type__ refers to **emotionIndicator** , which is set to three raw values:*1 for positive, 0 for neutral, -1 for negative*. 
* __sentiment weight__ refers to **emotionWeight**, which shows emotional intensity the company stakeholders react to the news.
* __relevance__ refers to  and **ItemRelevance**, which means the relevance between news and the stock.

3. Now we calculate sentiment score for each stock per news: 
$$ sentiment\\_score = sentiment\\_type \times sentiment\\_weight \times relevance \times 100 $$ Since there may be more than one news for a stock per day, we calculate the average senti_score to be the final sentiment feature. Then we map calendar date to trade date: cut at 15:00. It means that the sentimental data before __cut_hour: cut_time__(eg. 15:00) will be taken into current day's trading, the sentimental data after cut_hour:cut_time will be taken into next day's trading. 

### Traditional trading-related features
Rest eight traditional features related to trading infirmation are calculated by TaLib package. They are technical indicators: __MFI__, __SMA5__, __SMA10__, __MOM__, __ROC__, __ATR__, __BETA__ and __CCI__. Their meanings are as follows:
* __MFI__: Money Flow Index and Ratio，it also calls Volume Relative Strength Index，VRSI. It use four elements: days of rise, days of fall, increase of trading volume, decrease of trading volume to decide the trend of volume and energy and predict supply and demand in the market.
* __SMA5__: Simple moving average for 5 days, it indicates the average standard of price in 5 days.
* __SMA10__: Simple moving average for 10 days.
* __MOM__: Momentum. It is the volatility rate of one stock in one period.
* __ROC__: Rate of change. It compares current price with several days before to get the difference. It reflect the change rate of market.
* __ATR__: Average true range. It is the average of price fluctuation in a period. People can use it for timing trading.
* __BETA__: The β coefficient in CAPM model.
* __CCI__: Commodity Channel Index, an index created by American stock analyst Donald Lambert to judge the bias of stock price.

These traditional features are selected randomly because the our main purpose is to test the effectiveness of sentiment feature.

As for **labeling the data**, if the news_time is different from the trade date, we measure the close price of the two days, label it 1 for raise and 0 for drop. Otherwise we compare the close price and open price within the same day, also 1 for raise and 0 for drop

## Training
After data preparation, we have built 9 features, including 8 traditional trading-related features and 1 sentiment feature.

We have also built 1 label, which is 0 and 1, representing fall and rise of stock price. 

In order to find out how much contribution the sentiment feature can make to stock selection, we build 3 kinds of dataset: 
* __Set 1__: only contains 8 trading-related features
* __Set 2__: contains 8 trading-related features and the sentiment feature
* __Set 3__: only contains sentiment feature. 

By comparing __Set 1__ and __Set 2__, we want to find out whether the sentiment feature will boost the predicting performance. 

By comparing __Set 2__ and __Set 3__, we want to find out whether the 8 trading-related features serve as noise to sentiment feature or not.

We train on 70% of the sample and test on 30% of the sample. To increase training speed, we standardize the data in advance. 

We apply LR, SVC, TREE methods to train. In each methods, we also use grid search to find the best hyperparameters. We test our model on test datasets and compare the results mainly through confusion matrics. 

## Testing
### 1. Logistic Regression

#### __Set 1__: only contains 8 trading-related features
The best parameter is C=1.0

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/LR%201.jpg)

Accuracy is 52.8%; PRE is 51.0%

#### __Set 2__: contains 8 trading-related features and the sentiment feature
The best parameter is C=0.001

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/LR%202.jpg)

Accuracy is 54.7%; PRE is 53.8%
#### __Set 3__: only contains sentiment feature
The best parameter is C=0.01

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/LR%203.jpg)

Accuracy is 52.2%; PRE is 51.7%

### 2. Support Vector Machine

#### __Set 1__: only contains 8 trading-related features
The best parameter is {'svc__C': 100.0, 'svc__gamma': 0.1, 'svc__kernel': 'rbf'}

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/SVC%201.jpg)

Accuracy is 59.9%; PRE is 57.3%

#### __Set 2__: contains 8 trading-related features and the sentiment feature
The best parameter is {'svc__C': 1.0, 'svc__gamma': 1.0, 'svc__kernel': 'rbf'}

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/SVC%202.jpg)

Accuracy is 59.2%; PRE is 57.6%
#### __Set 3__: only contains sentiment feature
The best parameter is {'svc__C': 0.1, 'svc__gamma': 10.0, 'svc__kernel': 'rbf'}

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/SVC%203.jpg)

Accuracy is 52.2%; PRE is 50.6%

### 3. Decision Tree

#### __Set 1__: only contains 8 trading-related features
The best parameter is max_depth=6

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/Tree%201.jpg)

Accuracy is 56.7%; PRE is 53.9%;

#### __Set 2__: contains 8 trading-related features and the sentiment feature
The best parameter is max_depth=6

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/Tree%202.jpg)

Accuracy is 57.0%; PRE is 54.4%

#### __Set 3__: only contains sentiment feature
The best parameter is max_depth=3

The confusion matric is:
![ad](https://github.com/SnakeWayne/PHBS_MLF_2019/blob/master/image/Tree%203.jpg)

Accuracy is 52.7%; PRE is 52.7%

## Preliminary Findings:
Considering that people's trading decisions rely on the prediction result. People will buy if the prediction is 1, and will not act if the prediction is 0. If prediction is 1 but it's 0 in reality, people will suffer financial loss. Consequently, we want high TP and low FP, so we need to measure the result based on Precision __(PRE=TP/(TP+FP))__. The higher PRE rate, the more possible our model can generate profit.

From previous testing results, we find that combining 8 trading-related features and the sentiment feature using SVC method can get the best outcome. The best kernel is 'rbf', so it may not perform well in linear classification. 

There are two other surprising findings:
* Solely using the sentiment feature (__Set 3__)sometimes performs worse than our randomly picked trading-related features, which may suggest that using multiple weak features somtimes perform better than just one well designed feature. 
* Solely using the seniment feature (__Set 3__), most of the predictions are 0, which suggests future price drop. This kind of passive thinking suggest that even when a good news happen we can't have the confidence to bet the stock price will raise, but we can pretty sure the price will drop if come bad news comes out.

