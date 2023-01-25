from xml.parsers.expat import model
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import sys
import os
sys.path.append('../../')
from Source.Util import *
from imblearn.over_sampling import SMOTE 

# to suppress convergence warning in LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
import pickle


# achieved after hypertuning
best_n_estimators = 500
best_learning_rate = 0.01
APPLY_SMOTE = False 

MODELS = {
   
    'DT' : {
        'default' : DecisionTreeClassifier(),
        'hyperparameters': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth' : [5, 10, None],
            'splitter' : ['best', 'random'], 
            'class_weight' : ['balanced']
        }
    }
    
}

"""
'RF' : {
        'default' : RandomForestClassifier(),
        'hyperparameters' : {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth' : [5, 10, None],
            'n_estimators' : [100, 500],
            'class_weight' : ['balanced']
        }
    },
 'LR' : {
        'default' : LogisticRegression(),
        'hyperparameters' : {
            'penalty' : ['l1','l2', 'elasticnet', 'none'],
            'C' : [0.01,0.1,1.0],
            'max_iter' : [10000],
            'class_weight' : ['balanced'],
            'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

        }
    },
    'LGBM' : {
        'default' : LGBMClassifier(),
        'hyperparameters' : {
            
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth' : [5, 10, None],
            'n_estimators' : [100, 500],
            'class_weight' : ['balanced'],
            'objective' : ['binary']
        }
    },
    'ET' : {
        'default' : ExtraTreesClassifier(),
        'hyperparameters' : {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth' : [5, 10, None],
            'n_estimators' : [100, 500],
            'class_weight' : ['balanced']
        }
    }
"""
def prepare_data_GP(df) : 
    clean_df = df.copy()
    boolean_cols = ['is_bug_fixing','is_documentation','is_feature'] 
    clean_df = clean_df.drop(columns = ['project','change_id','created','subject'])
    clean_df['status'] = 1 - clean_df['status'] 
    for col in boolean_cols: 
        clean_df[col] = clean_df[col].astype(int) 
    
    return clean_df
    
model_name = 'SOTA'    
def main():
    global data_folder, root, change_directory_path, changes_root, diff_root, result_folder, result_project_folder, project
    feature_list = initial_feature_list
    for project_name in projects: 

        project = project_name
        data_folder = "../../Data"
        root = f"{data_folder}/{project}"
        change_folder = "change"
        change_directory_path = f'{root}/{change_folder}'
        changes_root = f"{root}/changes"
        diff_root = f'{root}/diff'

        result_folder = "../../Results"
        result_project_folder = f"{result_folder}/{project}"
        print(project)
        df = pd.read_csv(f'{root}/{project}.csv')
        df_copy = df.copy()

        # scaling
        scaler = StandardScaler()
        df[feature_list] = scaler.fit_transform(df[feature_list])
        #selecting_classifier(df)
        #res = concept_drift_validation(df, df_copy, scaler,models= MODELS, folds_prefix_number = 5)
        # this method does new author and effectiveness result also.
        cross_validation(df, df_copy, scaler)

        ## dimension wise results
        # dimension_validation(df)

        ## multiple revisions
        #multiple_revisions()


def get_model():
    return LGBMClassifier(class_weight='balanced', subsample=0.9, subsample_freq=1, random_state=np.random.randint(seed))


def get_best_model():
    return LGBMClassifier(class_weight='balanced',n_estimators=best_n_estimators, learning_rate=best_learning_rate,
                          subsample=0.9, subsample_freq=1, random_state=np.random.randint(seed))


def selecting_classifier(df):
    print("Selecting best classifier with AUC, F1(M) and F1(A)")
    print("RandomForest")
    for n_estimators in [100, 500]:
        for max_depth in [None, 10, 15]:
            print(n_estimators, max_depth)
            run_model(RandomForestClassifier(class_weight='balanced', n_estimators=n_estimators, max_depth=max_depth), df)

    print("GradientBoost")
    for n_estimators in [100, 500]:
        for learning_rate in [0.1, 0.01]:
            print(n_estimators, learning_rate)
            run_model(GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate), df)

    print("ExtraTrees")
    for n_estimators in [100, 500]:
        for max_depth in [None, 5, 10]:
            print(n_estimators, max_depth)
            run_model(ExtraTreesClassifier(class_weight='balanced', n_estimators=n_estimators, max_depth=max_depth), df)

    print("LogisticRegression")
    for max_iter in [50, 100, 500,1000,10000,20000]:
        print(max_iter)
        run_model(LogisticRegression(class_weight='balanced', solver='saga', max_iter=max_iter), df)

    print("LightGBM")
    for n_estimators in [100, 500]:
        for learning_rate in [0.1, 0.01]:
                print(n_estimators, learning_rate)
                model = LGBMClassifier(class_weight='balanced', n_estimators=n_estimators, random_state=np.random.randint(seed),
                                     learning_rate=learning_rate, subsample=0.9, subsample_freq=1)
                run_model(model, df)


def multiple_revisions():
    print("Multiple revisions")
    datapath_gp = f"C:/Users/Moataz/Desktop/work/code_review_delay_prediction/early_abondon_prediction/{project}"
    total_df = pd.read_csv(f'{root}/{project}_multiple_revisions.csv')
    total_df['last_revision_no'] = total_df.groupby('change_id')['number_of_revision'].transform('max').astype(int)

    total_df = total_df.sort_values(by=['created'], ascending=True).reset_index(drop=True)

    feature_list = initial_feature_list

    total_df_copy = total_df.copy()
    total_df[feature_list] = StandardScaler().fit_transform(total_df[feature_list])
    '''
    print("Approach 1: result adding only revision no")
    feature_list = feature_list + ['number_of_revision']

    for run in range(runs):
        test_result = Result()
        results = [Result() for _ in range(2)]

        for fold in range(1, folds):
            train_size = total_df.shape[0] * fold // folds
            test_size = min(total_df.shape[0] * (fold + 1) // folds, total_df.shape[0])

            x_train, y_train = total_df.loc[:train_size - 1, feature_list], \
                               total_df.loc[:train_size - 1, target]
            x_test, y_test = total_df.loc[train_size:test_size - 1, feature_list], \
                             total_df.loc[train_size:test_size - 1, target]

            clf = get_best_model()
            clf.fit(x_train, y_train)

            y_prob = clf.predict_proba(x_test)[:, 1]
            test_result.calculate_result(y_test, y_prob, fold, False)

            revision_no = total_df_copy.loc[train_size:test_size - 1, 'number_of_revision']
            for i, result in enumerate(results):
                if i + 1 == len(results):
                    index = revision_no == total_df_copy.loc[train_size:test_size - 1, 'last_revision_no']
                else:
                    index = revision_no == i + 1
                y_prob = clf.predict_proba(x_test[index])[:, 1]
                result.calculate_result(y_test[index], y_prob, fold, False)

        test_result_df = test_result.get_df()
        print(test_result_df.mean())

        for result in results:
            print(result.get_df().mean())
        print()

    print("Approach2 : result using history of prior patches")
    '''
    feature_list = initial_feature_list + late_features
    for run in range(runs):
        test_result = Result()
        results = [Result() for _ in range(2)]
  
        for fold in range(1, folds):
            train_size = total_df.shape[0] * fold // folds
            test_size = min(total_df.shape[0] * (fold + 1) // folds, total_df.shape[0])

            x_train, y_train = total_df.loc[:train_size - 1, feature_list], \
                               total_df.loc[:train_size - 1, target]
            x_test, y_test = total_df.loc[train_size:test_size - 1, feature_list], \
                             total_df.loc[train_size:test_size - 1, target]
            
            clean_df = prepare_data_GP(total_df_copy) 
            train_df = clean_df.iloc[:train_size - 1]
            test_df = clean_df.iloc[train_size:test_size - 1]
            train_df.to_csv(os.path.join(datapath_gp,f"{project}_multiple_revisions_train_{fold - 1}.csv"),index = False)
            test_df.to_csv(os.path.join(datapath_gp,f"{project}_multiple_revisions_test_{fold - 1}.csv"),index = False)
            
            #clf = get_best_model()
            clf.fit(x_train, y_train)

            y_prob = clf.predict_proba(x_test)[:, 1]
            test_result.calculate_result(y_test, y_prob, fold, False)

            revision_no = total_df_copy.loc[train_size:test_size - 1, 'number_of_revision']
            for i, result in enumerate(results):
                if i + 1 == len(results):
                    index = revision_no == total_df_copy.loc[train_size:test_size - 1, 'last_revision_no']
                else:
                    index = revision_no == i + 1
                y_prob = clf.predict_proba(x_test[index])[:, 1]
                result.calculate_result(y_test[index], y_prob, fold, False)

        test_result_df = test_result.get_df()
        print(test_result_df)
        print(test_result_df.mean())

        for result in results:
            print(result.get_df().mean())
        print()


def cross_validation(df, df_copy, scaler, models = MODELS):
    print("Cross validation")
    
    feature_list = get_initial_feature_list()
    scores = [0] * 9
    datapath_gp = f"C:/Users/Moataz/Desktop/work/code_review_delay_prediction/early_abondon_prediction/{project}"
    all_results = []
    os.makedirs(datapath_gp,exist_ok=True)
    for model_name, model_data in MODELS.items(): 
        train_results = None
        test_results = None
        new_author_results = None
        feature_importances = []
        train_time = {fold: [] for fold in range(1, folds)}
        test_time = {fold: [] for fold in range(1, folds)}
        print('working on model: ',model_name)
        indicies = [(df.shape[0] * fold // folds, min(df.shape[0] * (fold + 1) // folds, df.shape[0])) for fold in range(1,folds) ]
        #clf = GridSearchCV(model_data['default'], model_data['hyperparameters'],n_jobs=-1)
        #clf.fit(df.loc[:, feature_list], df.loc[:, target],cv=indicies)
        for run in range(runs):
            test_result = Result()
            train_result = Result()
            new_author_result = Result()
            indicies = []
            print(f'run {run} cross val')
            for fold in range(1, folds):
                train_size = df.shape[0] * fold // folds
                test_size = min(df.shape[0] * (fold + 1) // folds, df.shape[0])
                
                indicies.append((list(range(train_size - 1)),list(range(train_size,test_size - 1))))
            #indicies = [(df.shape[0] * fold // folds, min(df.shape[0] * (fold + 1) // folds, df.shape[0])) for fold in range(1,folds) ]
            grid_search = GridSearchCV(model_data['default'], model_data['hyperparameters'],cv=indicies,scoring='roc_auc')
            grid_search.fit(df.loc[:, feature_list], df.loc[:, target])
            cv_objects_path = os.path.join(result_project_folder,project,model_name)
            os.makedirs(cv_objects_path,exist_ok=True)
            pickle.dump(grid_search,file=open(f'{cv_objects_path}/{project}_{model_name}_cross_val_run_{run}.pk', 'wb'))
            clf = grid_search.best_estimator_
            for fold in range(1, folds):
                train_size = df.shape[0] * fold // folds
                test_size = min(df.shape[0] * (fold + 1) // folds, df.shape[0])
            
                x_train, y_train = df.loc[:train_size - 1, feature_list], df.loc[:train_size - 1, target]
                x_test, y_test = df.loc[train_size:test_size - 1, feature_list], \
                                df.loc[train_size:test_size - 1, target]
                
                clean_df = prepare_data_GP(df_copy) 
                train_df = clean_df.iloc[:train_size - 1]
                test_df = clean_df.iloc[train_size:test_size - 1]
                train_df.to_csv(os.path.join(datapath_gp,f"{project}_train_{fold - 1}.csv"),index = False)
                train_df.to_csv(os.path.join(datapath_gp,f"{project}_new_developer_train_{fold - 1}.csv"),index = False)
                test_df.to_csv(os.path.join(datapath_gp,f"{project}_test_{fold - 1}.csv"),index = False)
                test_new = df_copy[train_size:test_size]
                test_new = test_new[test_new['total_change_num'] < 10]
                test_new_gp = prepare_data_GP(test_new)
                test_new_gp.to_csv(os.path.join(datapath_gp,f"{project}_new_developer_test_{fold - 1}.csv"),index = False)
                print('train_size:',len(x_train))
                print('test_size:',len(x_test))
                continue 
                start = time.time()
                #clf = get_best_model()
                if APPLY_SMOTE: 
                    print('applying smote')
                    sm = SMOTE(random_state=42)
                    x_train, y_train = sm.fit_resample(x_train, y_train)
                clf.fit(x_train, y_train)
                train_time[fold].append(time.time() - start)

                y_prob = clf.predict_proba(x_train)[:, 1]
                train_result.calculate_result(y_train, y_prob, fold, False)

                start = time.time()
                y_prob = clf.predict_proba(x_test)[:, 1]
                test_time[fold].append(time.time() - start)
                test_result.calculate_result(y_test, y_prob, fold, False)

                for k in range(1, 10):
                    score = Result.cost_effectiveness(y_test, y_prob, k*10)
                    scores[k-1] += score

                test_new = df_copy[train_size:test_size]
                test_new = test_new[test_new['total_change_num'] < 10]
                y_prob = clf.predict_proba(scaler.transform(test_new[feature_list]))[:, 1]
                new_author_result.calculate_result(test_new[target], y_prob, fold, False)

                #feature_importances.append(clf.feature_importances_)

            train_result_df = train_result.get_df()
            test_result_df = test_result.get_df()
            new_author_result_df = new_author_result.get_df()
            test_result_df['algorithm'] = model_name
            test_result_df['train_or_test'] = "test"
            test_result_df['model_id'] = "best_performance_model"
            test_result_df['project'] = project
            test_result_df['run'] = run
            

            train_result_df['algorithm'] = model_name
            train_result_df['train_or_test'] = "train"
            train_result_df['model_id'] = "best_performance_model"
            train_result_df['project'] = project
            train_result_df['run'] = run

            new_author_result_df['algorithm'] = model_name
            new_author_result_df['train_or_test'] = "test"
            new_author_result_df['model_id'] = "best_performance_model"
            new_author_result_df['project'] = project
            new_author_result_df['run'] = run
            continue 
            all_results += [train_result_df,test_result_df]
            if run:
                train_results = pd.concat(train_results,train_result_df)
                test_results = pd.concat(test_results,test_result_df)
                new_author_results =pd.concat(new_author_results,new_author_result_df) 
            else:
                train_results = train_result_df
                test_results = test_result_df
                new_author_results = new_author_result_df
        continue
        print(test_results.head(30))
        result_df = pd.DataFrame({"train": train_results.median(), "test": test_results.median(),
                                "new_developers":new_author_results.median()}).reset_index()
        print(result_df)

        print("Effectiveness")
        for k in range(1, 10):
            print(k*10, scores[k-1] / (runs*(folds - 1)))

        # average time
        for fold in range(1, folds):
            train_time[fold] = np.mean(train_time[fold])
            test_time[fold] = np.mean(test_time[fold])
        train_results['time'] = train_time.values()
        test_results['time'] = test_time.values()

        train_results.to_csv(f'{result_project_folder}/{project}_{model_name}_train_result_cross.csv', index=False, float_format='%.3f')
        test_results.to_csv(f'{result_project_folder}/{project}_{model_name}_test_result_cross.csv', index=False, float_format='%.3f')
        result_df.to_csv(f'{result_project_folder}/{project}_{model_name}_result_cross.csv', index=False, float_format='%.3f')

        # process and dump feature importance
        #feature_importance_df = pd.DataFrame({'feature': feature_list, 'importance': np.mean(feature_importances, axis=0)})
        #feature_importance_df['importance'] = feature_importance_df['importance'] * 100 / feature_importance_df[
        #    'importance'].sum()
        #feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
        #feature_importance_df.to_csv(f'{result_project_folder}/{project}_feature_importance_cross.csv', index=False,
        #                            float_format='%.3f')
        print()
    return
    all_results = pd.concat(all_results)
    all_results.to_csv(f'{result_project_folder}/{project}_all_models.csv', index=False, float_format='%.3f')

def dimension_validation(df):
    print("Varying dimensions")

    for group in features_group:
        print(group)
        feature_sub_list = features_group[group]
        total_result = None

        for run in range(runs):
            result = Result()
            for fold in range(1, folds):
                train_size = df.shape[0] * fold // folds
                test_size = min(df.shape[0] * (fold + 1) // folds, df.shape[0])

                x_train, y_train = df.loc[:train_size - 1, feature_sub_list], df.loc[:train_size - 1,target]
                x_test, y_test = df.loc[train_size:test_size - 1, feature_sub_list], \
                                 df.loc[train_size:test_size - 1,target]

                clf = get_best_model()
                clf.fit(x_train, y_train)

                y_prob = clf.predict_proba(x_test)[:, 1]
                result.calculate_result(y_test, y_prob, fold, False)

            result_df = result.get_df()
            if run: total_result += result_df
            else: total_result = result_df

        total_result /= runs
        print(total_result.mean())
        print()


def concept_drift_validation(df, df_copy, scaler, models = MODELS, folds_prefix_number = 5): 
    print("Concept drift validation")
    feature_list = get_initial_feature_list()
    scores = [0] * 9
    datapath_gp = f"C:/Users/Moataz/Desktop/work/code_review_delay_prediction/early_abondon_prediction/concept_drift_val/{project}"
    all_results = []
    os.makedirs(datapath_gp,exist_ok=True)
    new_data_indicies = []
    old_data_indicies = []
    print(len(df))
    for iteration in range(1,folds_prefix_number + 1) : 
        old_data_train_end_index = df.shape[0] * folds_prefix_number  // folds
        new_data_train_start_index = df.shape[0] * iteration  // folds
        new_data_train_end_index = df.shape[0] * (iteration + folds_prefix_number )  // folds
        test_data_start_index = df.shape[0] * (folds_prefix_number + iteration ) // folds
        test_data_end_index = min(df.shape[0] * (folds_prefix_number + iteration + 1) // folds, df.shape[0])
        print('old_data_train_end_index:', old_data_train_end_index)
        print('new_data_train_start_index:', new_data_train_start_index)
        print('new_data_train_end_index:', new_data_train_end_index)
        print('test_data_start_index:', test_data_start_index)
        print('test_data_end_index:', test_data_end_index)
        old_data_indicies.append((list(range(old_data_train_end_index - 1)), list(range(test_data_start_index, test_data_end_index - 1 ))))
        new_data_indicies.append((list(range(new_data_train_start_index, new_data_train_end_index - 1)), list(range(test_data_start_index, test_data_end_index - 1 ))))
    for model_name, model_data in MODELS.items(): 
        train_results = None
        test_results = None
        new_author_results = None
        feature_importances = []
        print('working on model: ',model_name)

        for run in range(runs):
            old_test_result = Result()
            old_train_result = Result()
            new_test_result = Result()
            new_train_result = Result()

            new_author_result = Result()
            indicies = []
            print(f'run {run} cross val')
            #indicies = [(df.shape[0] * fold // folds, min(df.shape[0] * (fold + 1) // folds, df.shape[0])) for fold in range(1,folds) ]
            print('fitting model for old data')
            old_grid_search = GridSearchCV(model_data['default'], model_data['hyperparameters'],cv=old_data_indicies,scoring='roc_auc')
            old_grid_search.fit(df.loc[:, feature_list], df.loc[:, target])
            print('fitting model for new data')
            new_grid_search = GridSearchCV(model_data['default'], model_data['hyperparameters'],cv=new_data_indicies,scoring='roc_auc')
            new_grid_search.fit(df.loc[:, feature_list], df.loc[:, target])
            cv_objects_path = os.path.join(result_project_folder,project,'concept_drift',model_name)
            os.makedirs(cv_objects_path,exist_ok=True)

            pickle.dump(old_grid_search,file=open(f'{cv_objects_path}/{project}_{model_name}_old_cross_val_run_{run}.pk', 'wb'))
            pickle.dump(new_grid_search,file=open(f'{cv_objects_path}/{project}_{model_name}_new_cross_val_run_{run}.pk', 'wb'))

            old_clf= old_grid_search.best_estimator_
            new_clf= new_grid_search.best_estimator_
            for iteration in range(1,folds_prefix_number + 1) : 
                old_data_train_end_index = df.shape[0] * folds_prefix_number  // folds
                new_data_train_start_index = df.shape[0] * iteration  // folds
                new_data_train_end_index = df.shape[0] * (iteration + folds_prefix_number )  // folds
                test_data_start_index = df.shape[0] * (folds_prefix_number + iteration ) // folds
                test_data_end_index = min(df.shape[0] * (folds_prefix_number + iteration + 1) // folds, df.shape[0])
                x_train_old, y_train_old = df.loc[:old_data_train_end_index - 1, feature_list], df.loc[:old_data_train_end_index - 1, target]
                x_train_new, y_train_new = df.loc[new_data_train_start_index:new_data_train_end_index - 1, feature_list], \
                                df.loc[new_data_train_start_index:new_data_train_end_index - 1, target]
                
                x_test, y_test = df.loc[test_data_start_index:test_data_end_index - 1, feature_list], df.loc[test_data_start_index:test_data_end_index - 1, target]

                
                clean_df = prepare_data_GP(df_copy) 
                old_train_df = clean_df.iloc[:old_data_train_end_index - 1]
                new_train_df = clean_df.iloc[new_data_train_start_index:new_data_train_end_index - 1]
                test_df = clean_df.iloc[test_data_start_index:test_data_end_index - 1]

                old_train_df.to_csv(os.path.join(datapath_gp,f"{project}_old_train_{iteration - 1}.csv"),index = False)
                new_train_df.to_csv(os.path.join(datapath_gp,f"{project}_new_train_{iteration - 1}.csv"),index = False)

                test_df.to_csv(os.path.join(datapath_gp,f"{project}_old_test_{iteration - 1}.csv"),index = False)
                test_df.to_csv(os.path.join(datapath_gp,f"{project}_new_test_{iteration - 1}.csv"),index = False)
                #start = time.time()
                #clf = get_best_model()
                
                old_clf.fit(x_train_old, y_train_old)
                new_clf.fit(x_train_new, y_train_new)

                #train_time[fold].append(time.time() - start)

                old_train_result.calculate_result(y_train_old, old_clf.predict_proba(x_train_old)[:, 1], iteration, False)
                new_train_result.calculate_result(y_train_new, new_clf.predict_proba(x_train_new)[:, 1], iteration, False)

                new_test_result.calculate_result(y_test,  new_clf.predict_proba(x_test)[:, 1], iteration, False)
                old_test_result.calculate_result(y_test,  old_clf.predict_proba(x_test)[:, 1], iteration, False)

                #test_new = df_copy[train_size:test_size]
                #test_new = test_new[test_new['total_change_num'] < 10]
                #y_prob = clf.predict_proba(scaler.transform(test_new[feature_list]))[:, 1]
                #new_author_result.calculate_result(test_new[target], y_prob, fold, False)

                #feature_importances.append(clf.feature_importances_)

            new_train_result_df = new_train_result.get_df()
            old_train_result_df = old_train_result.get_df()

            new_test_result_df = new_test_result.get_df()
            old_test_result_df = old_test_result.get_df()
            print(old_train_result_df.head())
            print(old_test_result_df.head())
            #new_author_result_df = new_author_result.get_df()
            new_train_result_df['algorithm'] = model_name + '_new'
            new_train_result_df['train_or_test'] = "train"
            new_train_result_df['model_id'] = "best_performance_model"
            new_train_result_df['project'] = project
            new_train_result_df['run'] = run
            
            old_train_result_df['algorithm'] = model_name + '_old'
            old_train_result_df['train_or_test'] = "train"
            old_train_result_df['model_id'] = "best_performance_model"
            old_train_result_df['project'] = project
            old_train_result_df['run'] = run

            new_test_result_df['algorithm'] = model_name + '_new'
            new_test_result_df['train_or_test'] = "test"
            new_test_result_df['model_id'] = "best_performance_model"
            new_test_result_df['project'] = project
            new_test_result_df['run'] = run
            
            old_test_result_df['algorithm'] = model_name + "_old"
            old_test_result_df['train_or_test'] = "test"
            old_test_result_df['model_id'] = "best_performance_model"
            old_test_result_df['project'] = project
            old_test_result_df['run'] = run

           

            all_results += [new_train_result_df, new_test_result_df]
            all_results += [old_train_result_df, old_test_result_df]
            if run:
                old_train_results = pd.concat([old_train_results, old_train_result_df])
                old_test_results = pd.concat([old_test_results, old_test_result_df])
                new_train_results = pd.concat([old_train_results, old_train_result_df])
                new_test_results = pd.concat([new_test_results, new_test_result_df])
            else:
                old_train_results = old_train_result_df
                old_test_results = old_test_result_df
                new_train_results = old_train_result_df
                new_test_results = new_test_result_df
        
        
       
        old_train_results.to_csv(f'{result_project_folder}/{project}_{model_name}_old_train_result_cross.csv', index=False, float_format='%.3f')
        new_train_results.to_csv(f'{result_project_folder}/{project}_{model_name}_new_train_result_cross.csv', index=False, float_format='%.3f')
        old_test_results.to_csv(f'{result_project_folder}/{project}_{model_name}_old_test_result_cross.csv', index=False, float_format='%.3f')
        new_test_results.to_csv(f'{result_project_folder}/{project}_{model_name}_new_test_result_cross.csv', index=False, float_format='%.3f')
        #test_results.to_csv(f'{result_project_folder}/{project}_{model_name}_test_result_cross.csv', index=False, float_format='%.3f')
        #result_df.to_csv(f'{result_project_folder}/{project}_{model_name}_result_cross.csv', index=False, float_format='%.3f')

        # process and dump feature importance
        #feature_importance_df = pd.DataFrame({'feature': feature_list, 'importance': np.mean(feature_importances, axis=0)})
        #feature_importance_df['importance'] = feature_importance_df['importance'] * 100 / feature_importance_df[
        #    'importance'].sum()
        #feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
        #feature_importance_df.to_csv(f'{result_project_folder}/{project}_feature_importance_cross.csv', index=False,
        #                            float_format='%.3f')
        print()
    all_results = pd.concat(all_results)
    all_results.to_csv(f'{result_project_folder}/{project}_concept_drift_all_models.csv', index=False, float_format='%.3f')

if __name__ == '__main__':
    main()