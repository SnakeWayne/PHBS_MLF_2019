# PHBS_MLF_2019
phbs 2020 spring module3 machine learning
## Final Project
### Group Information

|Name|Student ID|
|:---|:---|
|Shen Tingwei|1901212631|
|He Junxian|1901212583 |
|Yu Xinlin|1901212664|
|Wang Zixiong|1901212644|

### Introduction

Our project is going to do stock selection implementing the machine learning methods. Considering using the pure trading information performance not that well, we introduce the sentiment data borrowed from [chinascope](http://finance.chinascope.com/www/) based on the news. We prepear to generate some relating features to boost the accuracy.

### Data set 
You can find the sample data in the **data** folder

### Data Summary
The sentiment data conclude from one article review the emotion torward a certain company or person where **emotionIndicator** is set to three raw values:*1 for positive, 0 for neutral, 2 for negative*. And it choose the indicator with the highest weight to be its overall sentiment. ChinaScope has clame that this classifaction accuracy is above 80%.

And other trading data is provided by the api from [joinquant](https://www.joinquant.com/), inclduing the basic ohlc data and others.