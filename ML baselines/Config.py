projects = ['Libreoffice', 'Eclipse', 'Gerrithub']
project = projects[2]

data_folder = "../../Data"
root = f"{data_folder}/{project}"
change_folder = "change"
change_directory_path = f'{root}/{change_folder}'
changes_root = f"{root}/changes"
diff_root = f'{root}/diff'

result_folder = "../../Results"
result_project_folder = f"{result_folder}/{project}"

target = 'status'
seed = 2021
folds = 11
runs = 100

features_group = {
    'author': ['author_experience', 'author_merge_ratio', 'author_changes_per_week',
               'author_merge_ratio_in_project', 'total_change_num', 'author_review_num'],
    'text': ['description_length', 'is_documentation', 'is_bug_fixing', 'is_feature'],
    'project': ['project_changes_per_week', 'project_merge_ratio', 'changes_per_author'],
    'reviewer': ['num_of_reviewers', 'num_of_bot_reviewers', 'avg_reviewer_experience', 'avg_reviewer_review_count'],
    'code': ['lines_added', 'lines_deleted', 'files_added', 'files_deleted', 'files_modified',
             'num_of_directory', 'modify_entropy', 'subsystem_num']
}



def get_initial_feature_list() -> [str]:
    feature_list = []
    for group in features_group:
        feature_list.extend(features_group[group])
    return feature_list


initial_feature_list = get_initial_feature_list()