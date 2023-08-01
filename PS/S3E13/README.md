# [PS S3E13] Simple EDA + Ensemble for Kaggle Playground Series - Classification with a Tabular Vector Borne Disease Dataset

<img src="https://github.com/Oleksiy-Zhukov/Kaggle-Competitions/assets/75014961/74161f7c-b244-4fe0-90c5-2270ba50e3d1">

## Introduction

Welcome to the README for the Kaggle Playground Series competition titled "Classification with a Tabular Vector Borne Disease Dataset." This repository contains the code and analysis for the competition entry, which achieved a top 5 result out of 934 teams with a best MAP@3 score of 0.51644.

The goal of this competition was to explore, analyze, and build an ensemble model to predict the classification task associated with a tabular dataset related to vector-borne diseases. The competition dataset is synthetically generated from real-world data, allowing participants to quickly iterate through various model and feature engineering ideas while keeping test labels private.

## Kaggle Notebook
In the Kaggle Notebook [link](https://www.kaggle.com/code/zhukovoleksiy/5-solution-ps3e13-ensemble/notebook), I followed a structured approach to tackle the competition's task. Here's an outline of the key sections:

### 1. Import & Load Data
In this initial section, we imported necessary libraries and loaded the dataset, enabling us to proceed with data exploration and analysis.

To run the Kaggle Notebook and reproduce the analysis, make sure to have the following libraries and dependencies installed. You can use the [`requirements.txt`](https://github.com/Oleksiy-Zhukov/Kaggle-Competitions/blob/main/PS/S3E13/requirements.txt) file to install the necessary packages using `pip`:

The Kaggle Notebook utilizes various libraries for data preprocessing, model building, feature engineering, and visualization. Below is a breakdown of the libraries used in the import section:

* `numpy`: A library for numerical computing in Python.
* `pandas`: A powerful library for data manipulation and analysis.
* `plotly`: A library for interactive and declarative data visualization.
* `matplotlib`: A popular plotting library in Python for creating static, interactive, and animated plots.
* `seaborn`: A data visualization library based on Matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.
* `category_encoders`: A library for encoding categorical features in various ways.
* `imbalanced-learn`: A library for dealing with imbalanced datasets using under-sampling and over-sampling techniques.
* `optuna`: An automatic hyperparameter optimization library.
* `joblib`: A library for lightweight pipelining in Python.
* `xgboost`: A gradient boosting library that provides an efficient and scalable implementation of gradient boosting.
* `lightgbm`: A gradient boosting library that is optimized for performance and handles large datasets efficiently.
* `catboost`: A gradient boosting library that is designed for categorical data and handles categorical features efficiently.
* `umap-learn`: A library for dimensionality reduction that provides manifold learning techniques.

These libraries are crucial for conducting the exploratory data analysis, feature engineering, model selection, hyperparameter tuning, and model evaluation in the Kaggle Notebook.

Please ensure you have installed the specified versions of these libraries before running the notebook.

*Note*:
In the import section, we suppress warnings to ensure a clean output during the notebook execution. Additionally, you may notice that some libraries are imported but not explicitly used in the provided notebook content. These imports may have been used in other parts of the notebook, such as commented-out code or code blocks removed from the provided content.

### 3. Basic EDA
We performed Exploratory Data Analysis (EDA) to gain insights into the dataset's characteristics. This involved visualizing distributions, examining feature correlations, and identifying potential outliers.

<img src="https://github.com/Oleksiy-Zhukov/Kaggle-Competitions/assets/75014961/354066db-9bc8-4342-96c0-8b1667db13c2">


### 4. Feature Engineering
The Feature Engineering phase aimed to prepare the data for model building. We created new features, handled missing values, and preprocessed the dataset to enhance model performance.

```python
# Implementing features
for df in [df_concat, df_test]:
    df['similar_cluster'] = df[similar_columns].sum(axis=1)
    df['chikungunya_columns'] = df[chikungunya_columns].sum(axis=1)
    df['lyme_columns'] = df[lyme_columns].sum(axis=1)
    df['red_cols'] = df[red_cols].sum(axis=1)
    df['orange_cols'] = df[orange_cols].sum(axis=1)
    df['green_cols'] = df[green_cols].sum(axis=1)

for df in [df_concat, df_test]: 
    tungiasis_columns = ['ulcers', 'toenail_loss', 'itchiness']
    df['tungiasis_cluster'] = df[tungiasis_columns].sum(axis=1)
    
    columns = [col for col in df if col != 'prognosis']
    df[columns] = df[columns].astype(int)
```

Also was Decomposition was implemented. With dimensionality reduction using various decomposition methods, such as PCA (Principal Component Analysis), NMF (Non-Negative Matrix Factorization), UMAP (Uniform Manifold Approximation and Projection), and t-SNE (t-Distributed Stochastic Neighbor Embedding).

```python
class Decomp:
    def __init__(self, n_components, method="pca", scaler_method='standard'):
        self.n_components = n_components
        self.method = method
        self.scaler_method = scaler_method
        
    def dimension_reduction(self, df):
            
        X_reduced = self.dimension_method(df)
        df_comp = pd.DataFrame(X_reduced, columns=[f'{self.method.upper()}_{_}' for _ in range(self.n_components)], index=df.index)
        
        return df_comp
    
    def dimension_method(self, df):
        
        X = self.scaler(df)
        if self.method == "pca":
            comp = PCA(n_components=self.n_components, random_state=0)
            X_reduced = comp.fit_transform(X)
        elif self.method == "nmf":
            comp = NMF(n_components=self.n_components, random_state=0)
            X_reduced = comp.fit_transform(X)
        elif self.method == "umap":
            comp = UMAP(n_components=self.n_components, random_state=0)
            X_reduced = comp.fit_transform(X)
        elif self.method == "tsne":
            comp = TSNE(n_components=self.n_components, random_state=0) # Recommend n_components=2
            X_reduced = comp.fit_transform(X)
        else:
            raise ValueError(f"Invalid method name: {method}")
        
        self.comp = comp
        return X_reduced
    
    def scaler(self, df):
        
        _df = df.copy()
            
        if self.scaler_method == "standard":
            return StandardScaler().fit_transform(_df)
        elif self.scaler_method == "minmax":
            return MinMaxScaler().fit_transform(_df)
        elif self.scaler_method == None:
            return _df.values
        else:
            raise ValueError(f"Invalid scaler_method name")
        
    def get_columns(self):
        return [f'{self.method.upper()}_{_}' for _ in range(self.n_components)]
    
    def transform(self, df):
        X = self.scaler(df)
        X_reduced = self.comp.transform(X)
        df_comp = pd.DataFrame(X_reduced, columns=[f'{self.method.upper()}_{_}' for _ in range(self.n_components)], index=df.index)
        
        return df_comp
    
    @property
    def get_explained_variance_ratio(self):
        
        return np.sum(self.comp.explained_variance_ratio_)
```

### 5. Model Building
We employed a diverse set of machine learning models for the ensemble. The models used were:

* Support Vector Classifier (SVC)
* Logistic Regression (LR)
* XGBoost (XGB)
* LightGBM (LGB)
* CatBoost (Cat)
* Balanced Random Forest (BRF)
* Random Forest (RF)
```python
  class Classifier:
    def __init__(self, n_estimators=250, device="cpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.len_models = len(self.models)
        
    def _define_model(self):
        
        xgb_params = {
            'n_estimators': 950,
            'learning_rate': 0.15639672008038652,
            'max_depth': 9,
            'subsample': 0.7154363211099006,
            'colsample_bytree': 0.1834688802646254,
            'reg_alpha': 0.00662736159424831,
            'reg_lambda': 0.392111943900896,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'tree_method': 'hist',
            'verbosity': 0,
            'random_state': self.random_state
        }
        
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
        
        
        lgb_params = {
            'n_estimators': 650,
            'max_depth': 6,
            'learning_rate': 0.09800555723996654,
            'subsample': 0.5113976179887376,
            'colsample_bytree': 0.15594697300978008,
            'reg_alpha': 0.24312642991831,
            'reg_lambda': 0.06500132210882924,
            'one_hot_max_size': 95,
            'device': self.device,
            'random_state': self.random_state}
                
        
        cb_params = {
            'n_estimators': 1000,
            'depth': 9, 
            'learning_rate': 0.45645253367049604,
            'l2_leaf_reg': 8.407202048380578,
            'random_strength': 0.1793388390086202,
            'max_bin': 225, 
            'od_wait': 58, 
            'grow_policy': 'Lossguide',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
                
        models = {
            'svc': SVC(gamma="auto", probability=True, random_state=self.random_state),
#             'svc_li': SVC(kernel="linear", gamma="auto", probability=True, random_state=self.random_state),
#             'svc_po': SVC(kernel="poly", gamma="auto", probability=True, random_state=self.random_state),
#             'svc_si': SVC(kernel="sigmoid", gamma="auto", probability=True, random_state=self.random_state),
            'lr': LogisticRegression(max_iter=150, random_state=self.random_state, n_jobs=-1),
            'xgb': xgb.XGBClassifier(**xgb_params),
            'lgb': lgb.LGBMClassifier(**lgb_params),
            'cat': CatBoostClassifier(**cb_params),
            'brf': BalancedRandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=self.random_state),
            'rf': RandomForestClassifier(n_estimators=200, random_state=self.random_state),
           # 'knn': KNeighborsClassifier(n_neighbors=5),
           # 'mlp': MLPClassifier(random_state=self.random_state, max_iter=1000),
           # 'xgb_0':xgb.XGBClassifier(random_state=self.random_state),
            #'lgb_0':lgb.LGBMClassifier(random_state=self.random_state),
            #'cat_0':CatBoostClassifier(random_state=self.random_state),
        }
        return models
```

### 5. Model Optimization
To maximize model performance, we used Optuna to optimize hyperparameters for each model. Additionally, we optimized the ensemble weights to achieve the best possible combination of individual models.
This is code I used to hypertune my models:

```python
# Load data and define target and features
data = pd.DataFrame(y_train).join(X_train)
target = y_train
features = X_train

# Define params
n_splits = 10
n_trials = 100
early_stopping_rounds = 333

# Define objective function for Optuna
def objective(trial):
    # Define hyperparameters to optimize
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=50),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.001, 10.0),
        'n_jobs': -1,
        'eval_metric': 'mlogloss',
        'objective': 'multi:softprob',
        'tree_method': 'hist',
        'verbosity': 0,
        'random_state': 42
     }

    lgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.001, 10.0),
        'one_hot_max_size': trial.suggest_int('one_hot_max_size', 10, 100),
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'device': "cpu",
        'random_state': 42
    }

    cat_params = {
        'iterations': self.n_estimators,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 0.1, 10.0),
        'random_strength': trial.suggest_loguniform('random_strength', 0.1, 1.0),
        'max_bin': trial.suggest_int('max_bin', 100, 500),
        'od_wait': trial.suggest_int('od_wait', 20, 100),
        'one_hot_max_size': 70,
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'eval_metric': 'MultiClass',
        'loss_function': 'MultiClass',
        'random_state': 42
    }

    # Initialize KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize list to store MAP@3 scores for each fold
    map3_scores = []

    # Loop over folds in parallel
    results = Parallel(n_jobs=-1)(delayed(get_fold_score)(params, features, target, train_idx, valid_idx, early_stopping_rounds) for train_idx, valid_idx in kf.split(data))
    map3_scores = [r[0] for r in results]

    # Return negative mean of MAP@3 scores (Optuna maximizes the objective)
    return np.mean(map3_scores)

# Define function to get MAP@3 score for a fold
def get_fold_score(params, features, target, train_idx, valid_idx, early_stopping_rounds):
    # Split data into training and validation sets
    X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
    y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]

    # Initialize XGBoost model
    model = xgb.XGBClassifier(**params)

#    model = lgb.LGBMClassifier(**params)

#    model = CatBoostClassifier(**params)


    # Fit model on training data
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=early_stopping_rounds, verbose=False)

    # Predict probabilities for validation data
    y_pred = model.predict_proba(X_valid)
    top_preds = np.argsort(-y_pred, axis=1)[:, :3]

    # Calculate MAP@3 score for validation data
    map3 = mapk(y_valid.values.reshape(-1, 1), top_preds, 3)

    return (map3,)

# Run Optuna to find the best hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=n_trials)

# Print the best hyperparameters and their score
print('Best hyperparameters: {}'.format(study.best_params))
print('Best score: {}'.format(study.best_value))
```
Ensemble weights optimization:

```python
class OptunaWeights:
    def __init__(self, random_state, n_trials=2000):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 1e-12, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds), axis=0, weights=weights)

        # Calculate the MAP@3 score for the weighted prediction
        top_preds = np.argsort(-weighted_pred, axis=1)[:, :3]
        score = mapk(y_true.reshape(-1, 1), top_preds, 3)
        
        return score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        self.study = optuna.create_study(sampler=sampler, study_name="OptunaWeights", direction='maximize') # minimize
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds), axis=0, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        self.fit(y_true, y_preds)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights
```

### 6. Make Submission
After training and optimizing the models, we made the final predictions and submitted the results to Kaggle for evaluation.
https://www.kaggle.com/competitions/playground-series-s3e13

https://www.kaggle.com/competitions/playground-series-s3e13/data
