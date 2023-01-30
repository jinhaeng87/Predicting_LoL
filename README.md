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

<details><summary>CLICK ME TO EXPAND and VIEW ALL PLOTS</summary><blockquote>
<p>
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

</p>
</blockquote></details>

In addition, Pearson Correlation heatmap was also plotted to view the phenomenon from different perspective.
See below for a correlation heatmap of all the features against the response variable. 

| <img src="/Pics/heatmap_0.png" alt="Alt text" title=""> |
|:--:|
|*Heatmap of features against the response variable*|

And correlation heatmaps for the features themselve. Due to the large number of features, I have broken down the heatmap into 2 parts; <br>
Red team related features and blue team related features, to improve the visibility and eligibility. 

| <img src="/Pics/heatmap_r.png" alt="Alt text" title=""> |
|:--:|
|*Heatmap for Red Team related features.*|

| <img src="/Pics/heatmap_b.png" alt="Alt text" title=""> |
|:--:|
|*Heatmap for Blue Team related features.*|

It is clear that some of the features are strongly correlated, causing potential multi-collinearity. <br>
Such can be detrimental when building a classification model and need to be dealt with. <br>
Based on the heatmap, I have setup a threshold of 0.8, hence if any two features are correlated higher than the threshold, <br>
one of them will be dropped randomly. 

```python3
# Function to drop highly correlated variables. Threshold is set to 0.8 by default, but an user can change this.

def drop_high_corr_col(df, threshold = 0.8):
    lower_tri = corr_mat(df)
    to_drop = [c for c in lower_tri.columns if any(lower_tri[c] > 0.8)]
    out = df.drop(to_drop, axis = 1)
    return out
```

We would also eliminate features that are weakly correlated to the response variable with proper thresholding, <br>
but when I go into feature engineering this will be taken care of and potentially be redundant so I will skip here.


## 2. Feature Engineering
We have already ruled out some of the highly correlated features in previous section, but the dataset still contains of too many features (potential cause of overfitting the model) and features that are less important. Normally we would have to worry about encoding categorical values, but such is not the case for this dataset. <br>

Among various regularization techniques, I have chosen to go with Elastic Net regression for choosing important features, because all the features seem to be relatable, as none of them seems unrelated. Therefore, Elastic Net would be a perfect choice to shrink some features while ruling out the others.

### Grid Search for the best parameter
There are multiple parameters that can be tuned to achieve optimal performance, two of the most prominent parameters are alpha and l1_ratio.
With RMSE as scoring method, various alpha and l1_ratio values were implemented via cross validation. <br>
Below is the validation plot for how different combination of alpha-l1_ratio performed.
```python3
# RMSE score evaluation Function
def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

X, y = data_prep(df)
# Choose matrix of alphas and l1-ratios for grid search
alphas = [0.0005, 0.001, 0.01, 0.03, 0.05, 0.1]
l1_ratios = [1, 0.9, 0.8, 0.7, 0.5, 0.1]

# Elastic Net Grid search to find best performing parameters
cv_elastic = [rmse_cv(ElasticNet(alpha = alpha, l1_ratio = l1_ratio), X, y).mean() 
            for (alpha, l1_ratio) in product(alphas, l1_ratios)]

# Find the alpha-l1_ratio where rmse is the minimum
idx = list(product(alphas, l1_ratios))
m_idx = cv_elastic.index(min(cv_elastic))
alpha, l1_ratio = idx[m_idx]

# Fit the model
elastic = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
elastic.fit(X, y)

# Plot out validation curve
plt.figure(figsize=(12, 6))
p_cv_elastic = pd.Series(cv_elastic, index = idx)
p_cv_elastic = p_cv_elastic.dropna()
p_cv_elastic.plot(title = "Validation - Elastic Net")
plt.xlabel("alpha - l1_ratio")
plt.ylabel("rmse")

```
| <img src="/Pics/enet_val.png" alt="Alt text" title=""> |
|:--:|
|*Validation score of Elastic Net Regression*|

As can be seen from the code and the plot, it will fetch the optimal combination of alpha-l1_ratio that resulted in the lowest RMSE.
These chosen values will be used to fit Elastic Net model and proceed into Feature Selection.

### Feature Selection
Upon fitting Elastic Net model, we can basically rule out any features with coefficient = 0. <br>
Depending on how many of them are there, we can either decide to further proceed into hard code the number of features desired or leave it at just ruling out 0 coefficient ones. <br>
In this case, I have deicded to go with 20 best features. Below are the plots for both cases;<br>
1. Displaying all features with their coefficients
2. Displaying top 20 features.


```python3
# Feature Selection with Elastic Net
# Layout the features based on the importance and visualize.
cf = pd.Series(elastic.coef_, index = X.columns)
imp_cf = pd.concat([cf.sort_values().head(10), cf.sort_values().tail(10)])
print("Elastic Net picked " + str(sum(cf != 0)) + " variables and eliminated the other " +  str(sum(cf == 0)) + " variables")

    
# Layout all coeff
plt.figure(figsize=(8, 10))
cf.sort_values().plot(kind = "barh", color = '#f5bc42')
plt.title("All Coefficients in the Elastic Net Model")

# Layout 20 most important 
plt.figure(figsize=(8, 10))
imp_cf.plot(kind = "barh")
plt.title("20 Important Coefficients in the Elastic Net Model")
```
| <img src="/Pics/FE1.png" alt="Alt text" title=""> |
|:--:|
|*Coefficients of all features*|

| <img src="/Pics/FE2.png" alt="Alt text" title=""> |
|:--:|
|*Coefficients of top 20 features*|

At the end of feature engineering, the cleaned version of dataframe was exported.


## 3. Model Training
Among various classification techniques out there, I have decided to go with the ones that are universally considered to be robust. <br>
1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

For each method, grid searching was performed similarly to that of Elastic Net, to find the best performing parameters. 
As building classification model with grid searching can be computationally heavy and with data being static in this case, training multiple times won't be neccasary. Therefore, these models were dumped and saved as pickle files, which can be called upon and loaded for reproducibility. 

```python3
y = df['blueWins']
X = df.drop('blueWins', axis = 1)
scaler = MinMaxScaler()
scaler.fit(X)
X_sc = scaler.transform(X)

# Split the dataset into train and test. By default the ratio is 70/30
# Ratio can be hard-coded into different values

X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.3, random_state=42)
```

### Logistic Regression
```python3
# Logistic Regression Modeling. By default, this does not save the model but an user can set save = True
# If save is True, the model result will be saved in pickle format.

def logReg(X, y, save = False):
    lm = LogisticRegression()
    
    # Setup a matrix of parameters for grid search
    param_grid = [    
        {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
        'C' : np.logspace(-4, 4, 20),
        'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
        'max_iter' : [100, 1000, 2000]
        }
    ]
    
    # Start grid searching with 3-fold cross validation. CV can be altered into different number
    # Due to the limitation of local machine with computation cost, cv is set to 3 by default.
    cv_lm = GridSearchCV(lm, param_grid = param_grid, cv = 3, verbose = True, n_jobs = -1)
    best_cvlm = cv_lm.fit(X, y)
    print (f'Accuracy - : {best_cvlm.score(X, y):.3f}')
    
    if save:
        # Save the model by dumping it into pickle
        filename = os.path.join(wDir, 'models/logReg_model.sav')
        pickle.dump(best_cvlm, open(filename, 'wb'))

    return best_cvlm
```
### Random Forest Classifier
```python3
def rfc(X, y, save = False):
    rfc=RandomForestClassifier(random_state=42)
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :['gini', 'entropy']
    }

    CV_rfc = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 5)
    CV_rfc.fit(X, y)
    print (f'Accuracy - : {CV_rfc.score(X, y):.3f}')
    
    if save:
        # Save the model by dumping it into pickle
        filename = os.path.join(wDir, 'models/rfc_model.sav')
        pickle.dump(CV_rfc, open(filename, 'wb'))

    return CV_rfc
```

### XGBoost Classifier
```python3
def xgboost(X, y, save = False):
    xgb = XGBClassifier(use_label_encoder = False, random_state = 42)
    
    param_grid = { 
        "learning_rate": [0.0001,0.001, 0.01, 0.1, 1] ,
        "max_depth": [3,8,15],
        "gamma": [i/10.0 for i in range(0,5)],
        "colsample_bytree": [i/10.0 for i in range(3,10)],
        "reg_alpha": np.logspace(-4,2,5),
        "reg_lambda": np.logspace(-4,2,5)}
    scoring = ['recall']
    
    CV_xgb = RandomizedSearchCV(estimator = xgb, param_distributions = param_grid, n_iter = 48, 
                                scoring = scoring, refit = 'recall', n_jobs = -1, cv = 3, verbose=0)
    
    CV_xgb.fit(X, y)
    print (f'Accuracy - : {CV_xgb.score(X, y):.3f}')
    
    if save:
        # Save the model by dumping it into pickle
        filename = os.path.join(wDir, 'models/XGB_model.sav')
        pickle.dump(CV_xgb, open(filename, 'wb'))

    return CV_xgb
```
## 4. Model Performance and Validation
For classification, it is often desired to observe 4 main metrics: accuracy, recall, precision and f1_score. 
The following are the 4 metrics for each model displaying the performance. Confusion Matrices are plotted to visualize type 1 & 2 errors, demonstrating the number of True/False positives & negatives.

```
______________________________________________
Classifier: Logistic Regression
Accuracy: 0.7290823211875843
Precision: 0.7283702213279678
Recall: 0.7318059299191375
F1-score: 0.7300840336134454
______________________________________________

______________________________________________
Classifier: Random Forest Classifier
Accuracy: 0.717948717948718
Precision: 0.720108695652174
Recall: 0.7142857142857143
F1-score: 0.7171853856562923
______________________________________________

______________________________________________
Classifier: XGBoost Classifier
Accuracy: 0.7074898785425101
Precision: 0.6968730057434588
Recall: 0.7358490566037735
F1-score: 0.7158308751229105
______________________________________________

 ```   
 
| <img src="/Pics/cf_lr.png" alt="Alt text" title=""> |
|:--:|
|*Confusion Matrix for Logistic Regression*|

| <img src="/Pics/cf_rf.png" alt="Alt text" title=""> |
|:--:|
|*Confusion Matrix for Random Forest*|

| <img src="/Pics/cf_xgb.png" alt="Alt text" title=""> |
|:--:|
|*Confusion Matrix for XGBoost*|
