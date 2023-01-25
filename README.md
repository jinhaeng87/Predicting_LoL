# Predicting Win/Loss for League of Legend

This project is motivated by my passion towards the game League of Legend. As the game is prevalently popular, there are a lot of online sources where game-stats can be retrieved. Based on this dataset, the project is predicting whether a blue team will win a game or not (or red team losing).

## 1. Exploratory Data Analysis & Visualization
It is essential to analyze the given dataset prior to deep-diving into modeling. <br>
To initiate EDA, we start off with checking basic statistics of the datasets, displaying columns, and presence of nulls.

```python3
def data_init(df):
    print('Displaying columns:')
    print(df.columns)
    print('Checking Stats:')
    print(df.describe())
    print('Done!')
    print('Checking for nulls:')
    msno.bar(df)
```
```
Displaying columns:
Index(['gameId', 'blueWins', 'blueWardsPlaced', 'blueWardsDestroyed',
       'blueFirstBlood', 'blueKills', 'blueDeaths', 'blueAssists',
       'blueEliteMonsters', 'blueDragons', 'blueHeralds',
       'blueTowersDestroyed', 'blueTotalGold', 'blueAvgLevel',
       'blueTotalExperience', 'blueTotalMinionsKilled',
       'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
       'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
       'redFirstBlood', 'redKills', 'redDeaths', 'redAssists',
       'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',
       'redTotalGold', 'redAvgLevel', 'redTotalExperience',
       'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redGoldDiff',
       'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin'],
      dtype='object')
Checking Stats:
             gameId     blueWins  blueWardsPlaced  blueWardsDestroyed  \
count  9.879000e+03  9879.000000      9879.000000         9879.000000   
mean   4.500084e+09     0.499038        22.288288            2.824881   
std    2.757328e+07     0.500024        18.019177            2.174998   
min    4.295358e+09     0.000000         5.000000            0.000000   
25%    4.483301e+09     0.000000        14.000000            1.000000   
50%    4.510920e+09     0.000000        16.000000            3.000000   
75%    4.521733e+09     1.000000        20.000000            4.000000   
max    4.527991e+09     1.000000       250.000000           27.000000   

       blueFirstBlood    blueKills   blueDeaths  blueAssists  \
count     9879.000000  9879.000000  9879.000000  9879.000000   
mean         0.504808     6.183925     6.137666     6.645106   
std          0.500002     3.011028     2.933818     4.064520   
min          0.000000     0.000000     0.000000     0.000000   
25%          0.000000     4.000000     4.000000     4.000000   
50%          1.000000     6.000000     6.000000     6.000000   
75%          1.000000     8.000000     8.000000     9.000000   
max          1.000000    22.000000    22.000000    29.000000   

       blueEliteMonsters  blueDragons  ...  redTowersDestroyed  redTotalGold  \
count        9879.000000  9879.000000  ...         9879.000000   9879.000000   
mean            0.549954     0.361980  ...            0.043021  16489.041401   
std             0.625527     0.480597  ...            0.216900   1490.888406   
min             0.000000     0.000000  ...            0.000000  11212.000000   
25%             0.000000     0.000000  ...            0.000000  15427.500000   
50%             0.000000     0.000000  ...            0.000000  16378.000000   
75%             1.000000     1.000000  ...            0.000000  17418.500000   
max             2.000000     1.000000  ...            2.000000  22732.000000   

       redAvgLevel  redTotalExperience  redTotalMinionsKilled  \
count  9879.000000         9879.000000            9879.000000   
mean      6.925316        17961.730438             217.349226   
std       0.305311         1198.583912              21.911668   
min       4.800000        10465.000000             107.000000   
25%       6.800000        17209.500000             203.000000   
50%       7.000000        17974.000000             218.000000   
75%       7.200000        18764.500000             233.000000   
max       8.200000        22269.000000             289.000000   

       redTotalJungleMinionsKilled   redGoldDiff  redExperienceDiff  \
count                  9879.000000   9879.000000        9879.000000   
mean                     51.313088    -14.414111          33.620306   
std                      10.027885   2453.349179        1920.370438   
min                       4.000000 -11467.000000       -8348.000000   
25%                      44.000000  -1596.000000       -1212.000000   
50%                      51.000000    -14.000000          28.000000   
75%                      57.000000   1585.500000        1290.500000   
max                      92.000000  10830.000000        9333.000000   

       redCSPerMin  redGoldPerMin  
count  9879.000000    9879.000000  
mean     21.734923    1648.904140  
std       2.191167     149.088841  
min      10.700000    1121.200000  
25%      20.300000    1542.750000  
50%      21.800000    1637.800000  
75%      23.300000    1741.850000  
max      28.900000    2273.200000  

[8 rows x 40 columns]
Done!
Checking for nulls:
Processing column 0 to 4
Processing colum 5 to 9
Processing colum 10 to 14
Processing colum 15 to 19
Processing colum 20 to 24
Processing colum 25 to 29
Processing colum 30 to 34
Processing column 35 to 39
```

| <img src="/Pics/msno.png" alt="Alt text" title="Null Stats for the Dataset"> |
|:--:|
|*Checking for Nulls in the Dataset (Notice all features are without any nulls)*|

To observe how features are correlated to one another and to the response variable, bivariate analysis was conducted. <br>
Note for the eligibility sake, I have broken features down into size of 5.

| <img src="/Pics/bva1.png" alt="Alt text" title=""> |
|:--:|
|*Bivariate Analysis of five features. Note two colors represent response variable.*|

| <img src="/Pics/bva2.png" alt="Alt text" title=""> |
|:--:|
|*Bivariate Analysis of five features. Note two colors represent response variable.*|

| <img src="/Pics/bva3.png" alt="Alt text" title=""> |
|:--:|
|*Bivariate Analysis of five features. Note two colors represent response variable.*|

| <img src="/Pics/bva4.png" alt="Alt text" title=""> |
|:--:|
|*Bivariate Analysis of five features. Note two colors represent response variable.*|

| <img src="/Pics/bva5.png" alt="Alt text" title=""> |
|:--:|
|*Bivariate Analysis of five features. Note two colors represent response variable.*|

| <img src="/Pics/bva6.png" alt="Alt text" title=""> |
|:--:|
|*Bivariate Analysis of five features. Note two colors represent response variable.*|

| <img src="/Pics/bva7.png" alt="Alt text" title=""> |
|:--:|
|*Bivariate Analysis of five features. Note two colors represent response variable.*|

| <img src="/Pics/bva8.png" alt="Alt text" title=""> |
|:--:|
|*Bivariate Analysis of five features. Note two colors represent response variable.*|



