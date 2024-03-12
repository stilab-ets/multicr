import sys
sys.path.append('../../')
from Config import *
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from Source.Util import Result
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
import pickle
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
best_n_estimators = 500
best_learning_rate = 0.01
seed = 2022
'''
,

    'DT' : {
        'default' : DecisionTreeClassifier(),
        'hyperparameters': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth' : [5, 10],
            'splitter' : ['best', 'random'], 
            'class_weight' : ['balanced'],
            'random_state': [seed]
        }
    },
       'RF' : {
        'default' : RandomForestClassifier(n_jobs=-1),
        'hyperparameters' : {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth' : [5, 10, None],
            'n_estimators' : [100, 500],
            'class_weight' : ['balanced'],
            'random_state': [seed]
        }
    },
 
    'LGBM' : {
        'default' : LGBMClassifier(n_jobs=-1),
        'hyperparameters' : {
           'class_weight':['balanced'],
            'n_estimators':[best_n_estimators],
            'learning_rate':[best_learning_rate],
            'subsample':[0.9], 
            'subsample_freq':[1], 
            'random_state':[np.random.randint(seed)]
        }
    },
    'ET' : {
        'default' : ExtraTreesClassifier(n_jobs=-1),
        'hyperparameters' : {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth' : [5, 10, None],
            'n_estimators' : [100, 500],
            'class_weight' : ['balanced'], 
            'random_state': [seed]
        }
    }
'''
MODELS = {
   'LR' : {
        'default' : LogisticRegression(n_jobs=-1),
        'hyperparameters' : {
            'penalty' : ['l1','l2', 'elasticnet', 'none'],
            'C' : [0.01,0.1,1.0],
            'max_iter' : [10000],
            'class_weight' : ['balanced'],
            'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'random_state': [seed]

        }
    },
        'DT' : {
        'default' : DecisionTreeClassifier(),
        'hyperparameters': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth' : [5, 10],
            'splitter' : ['best', 'random'], 
            'class_weight' : ['balanced'],
            'random_state': [seed]
        }
    },
       'RF' : {
        'default' : RandomForestClassifier(n_jobs=-1),
        'hyperparameters' : {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth' : [5, 10, None],
            'n_estimators' : [100, 500],
            'class_weight' : ['balanced'],
            'random_state': [seed]
        }
    },
    'LGBM' : {
        'default' : get_best_model(),
        'hyperparameters' : {
        }
    },
    'ET' : {
        'default' : ExtraTreesClassifier(n_jobs=-1),
        'hyperparameters' : {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth' : [5, 10, None],
            'n_estimators' : [100, 500],
            'class_weight' : ['balanced'], 
            'random_state': [seed]
        }
    }
}

GP_cross_project_path = './cross_project_data/GP_data'

os.makedirs(GP_cross_project_path,exist_ok=True)
def prepare_data_GP(df) : 
    clean_df = df.copy()
    boolean_cols = ['is_bug_fixing','is_documentation','is_feature'] 
    clean_df = clean_df.drop(columns = ['project','change_id','created','subject'])
    clean_df['status'] = 1 - clean_df['status'] 
    for col in boolean_cols: 
        clean_df[col] = clean_df[col].astype(int) 
    
    return clean_df

def get_best_model():
    return LGBMClassifier(class_weight='balanced', n_estimators=best_n_estimators, learning_rate=best_learning_rate,
                          subsample=0.9, subsample_freq=1, random_state=np.random.randint(seed))


scaler = StandardScaler()
feature_list = initial_feature_list
target = 'status'

data = {
    project: pd.read_csv(f'{data_folder}/{project}/{project}.csv', encoding='utf-8') for project in projects
}
def scenario_1_cross_project(models = MODELS,runs=runs): 
    results = []
    for current_project in projects:
        print(current_project)
        df = data[current_project]
        #train_gp = prepare_data_GP(df)
        #train_gp.to_csv(os.path.join(GP_cross_project_path,f'train_{current_project}_{other_project}.csv'))
        scaler = StandardScaler()
        df[feature_list] = scaler.fit_transform(df[feature_list], df[target])
        for other_project in projects:
            if current_project == other_project: continue

            other_df = data[other_project]
            #test_gp = prepare_data_GP(other_df)
            #test_gp.to_csv(os.path.join(GP_cross_project_path,f'test_{current_project}_{other_project}.csv'))
            other_df[feature_list] = scaler.transform(other_df[feature_list])
            print(other_df.head())
            y_true = other_df[target]
            
            
            for model_name, model_data in models.items() :
                print('Model:', model_name)
                for run in range(runs):
                    grid_search = GridSearchCV(model_data['default'], model_data['hyperparameters'],cv=10,scoring='roc_auc',refit=True, n_jobs=-1)
                    grid_search.fit(other_df.loc[:,feature_list], other_df.loc[:, target])
                    
                    model = grid_search.best_estimator_
                    model.fit(df[feature_list], df[target])

                    y_prob = model.predict_proba(other_df[feature_list])[:, 1]
                    #auc = roc_auc_score(y_true, y_prob)
                    #cost_effectiveness = Result.cost_effectiveness(y_true, y_prob, 20)

                    y_pred = np.round(y_prob)
                    f1_m, f1_a = f1_score(y_true, y_pred), f1_score(y_true, y_pred, pos_label=0)
                    mcc = matthews_corrcoef(y_true, y_pred)
                    prec_m, prec_a = precision_score(y_true, y_pred), precision_score(y_true, y_pred, pos_label=0) 
                    recall_m, recall_a = recall_score(y_true, y_pred), recall_score(y_true, y_pred, pos_label=0) 
                    new_row = {
                        'Source' : current_project,
                        'Target' : other_project, 
                        'run' : run, 
                        'algorithm' : model_name,
                        'MCC': mcc,
                        'f1_A' : f1_a,
                        'f1_M' : f1_m, 
                        'precision_m' : prec_m, 
                        'precision_a': prec_a, 
                        'recall_m': recall_m, 
                        'recall_a': recall_a
                    }
                    results.append(new_row)

            #results = np.mean(results, axis=0)
            #print(f'{other_project}: {np.round(results, 3)}')
            #print()
    return pd.DataFrame(results)

def scenario_2_cross_project(models = MODELS,runs=runs): 
    results = []
    for target_project in  projects: 

        print('Working on project',target_project)
        target_df = data[target_project]
        test_gp  = prepare_data_GP(target_df)
        test_gp.to_csv(os.path.join(GP_cross_project_path,f'test_{target_project}_all.csv'))
        

        source_dfs = []
        for other_project in projects: 
            if other_project == target_project: 
                continue 
            source_dfs.append(data[other_project])

        source_df = pd.concat(source_dfs)
        train_gp = prepare_data_GP(source_df)
        train_gp.to_csv(os.path.join(GP_cross_project_path,f'train_{target_project}_all.csv'))
        source_df[feature_list] = scaler.fit_transform(source_df[feature_list])
        target_df[feature_list] = scaler.transform(target_df[feature_list])

        for model_name, model_data in models.items() : 
             print('*** Working on model',model_name)
             for run in range(runs): 
                print(f'*** Working on model run {run}',model_name)
                grid_search = GridSearchCV(model_data['default'], model_data['hyperparameters'],cv=10,scoring='roc_auc',refit=True, n_jobs=-1)
                grid_search.fit(source_df.loc[:, feature_list], source_df.loc[:, target])
                y_pred = grid_search.best_estimator_.predict(target_df[feature_list])
                new_row = {
                    'project_name' : target_project, 'algorithm' : model_name, 
                    'run' : run, 'model_id' : 'best_model_performance', 
                    'MCC' : matthews_corrcoef(target_df[target], y_pred),
                    'f1_M' : f1_score(target_df[target], y_pred),
                    'f1_A' : f1_score(target_df[target], y_pred, pos_label=0),
                    'precision_m' : precision_score(target_df[target], y_pred),
                    'precision_a' : precision_score(target_df[target], y_pred, pos_label=0),
                    'recall_m' : recall_score(target_df[target], y_pred),
                    'recall_a' : recall_score(target_df[target], y_pred, pos_label=0)
                }
                results.append(new_row)
    return pd.DataFrame(results)
# main 
scenario_1_results = scenario_1_cross_project()
scenario_1_results.to_csv('ML_scenario_1_results_LR.csv',index=False)
scenario_2_results = scenario_2_cross_project()
scenario_2_results.to_csv('ML_scenario_2_results_LR.csv',index=False)

# Libreoffice
# Eclipse: [0.643 0.951 0.852 0.23 ]
#
# Gerrithub: [0.669 0.925 0.887 0.322]
#
# Eclipse
# Libreoffice: [0.775 0.975 0.846 0.306]
#
# Gerrithub: [0.766 0.978 0.88  0.36 ]
#
# Gerrithub
# Libreoffice: [0.792 0.983 0.844 0.307]
#
# Eclipse: [0.811 0.959 0.876 0.502]