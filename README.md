# Predicting Win/Loss for League of Legend

This project is motivated by my passion towards the game League of Legend. As the game is prevalently popular, there are a lot of online sources where game-stats can be retrieved. Based on this dataset, the project is predicting whether a blue team will win a game or not (or red team losing).

## 1. Exploratory Data Analysis & Visualization
It is essential to analyze the given dataset prior to deep-diving into modeling. 
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

