import os

import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, \
    ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import sys


def calc_eval(eval_mat, y_from_herustica, y_true, eval_str):
    score = eval_mat(y_true, y_from_herustica)
    print("the score for herustica :" + str(score) + " " + eval_str)
    return score


df = ""


def function_for_strucure_3(on_all, model_number, path_for_read, path_for_write,
                            type_of_dataset):  # we have 3 structures {0,1,2}, when i type "structure_2" i mean 1 when it comes to indexes
    global df
    columns = ["from_what_architecture", "feature_family", "class", "model",
               "accuracy", "precision", "recall", "f1", "jaccard", "roc_auc", "average_precision",
               "explained_variance", "max_error", "neg_mean_absolute_error", "neg_mean_squared_error",
               "neg_mean_squared_log_error", "r2",
               "heuristic_accuracy", "heuristic_precision", "heuristic_recall", "heuristic_f1", "heuristic_jaccard",
               "heuristic_roc_auc", "heuristic_average_precision",
               "heuristic_explained_variance", "heuristic_max_error", "heuristic_neg_mean_absolute_error",
               "heuristic_neg_mean_squared_error", "heuristic_neg_mean_squared_log_error", "heuristic_r2",
               "pos_diff_accuracy", "pos_diff_precision", "pos_diff_recall", "pos_diff_f1", "pos_diff_jaccard",
               "pos_diff_roc_auc", "pos_diff_average_precision",
               "pos_diff_explained_variance", "pos_diff_max_error", "pos_diff_neg_mean_absolute_error",
               "pos_diff_neg_mean_squared_error", "pos_diff_neg_mean_squared_log_error", "pos_diff_r2",
               "neg_diff_accuracy", "neg_diff_precision", "neg_diff_recall", "neg_diff_f1", "neg_diff_jaccard",
               "neg_diff_roc_auc", "neg_diff_average_precision",
               "neg_diff_explained_variance", "neg_diff_max_error", "neg_diff_neg_mean_absolute_error",
               "neg_diff_neg_mean_squared_error", "neg_diff_neg_mean_squared_log_error", "neg_diff_r2",
               "AVG_per_paper", "BEST_per_paper", "WORST_per_paper", "ORACLE_AVG_per_paper", "ORACLE_BEST_per_paper",
               "ORACLE_WORST_per_paper",
               "HEURISTIC_AVG_per_paper", "HEURISTIC_BEST_per_paper", "HEURISTIC_WORST_per_paper"]

    Rows = [x for x in range(10000)]
    results_df = pd.DataFrame(index=Rows, columns=columns)
    path_for_save = path_for_write + f"\\out_all_structures_analysis_{on_all}_{model_number}_{type_of_dataset}.csv"
    features_family_dct = {0: "all", 1: "objects", 2: "cols", 3: "paper", 4: "obj & cols", 5: "obj & paper",
                           6: "cols & paper"}
    operator_dict = {2: [0.9, 0.8, 0.7, 0.6, 0.5], 3: [1], 4: [1], 5: [1], 6: [1], 7: [0.9, 0.8, 0.7, 0.6], 8: [1]}
    operator_max = {2: 4, 3: 2, 4: 1, 5: 2, 6: 11, 7: 2, 8: 4}

    models = [AdaBoostClassifier(random_state=0, n_estimators=150),
              XGBClassifier(random_state=0),
              RandomForestClassifier(random_state=0),
              ExtraTreesClassifier(random_state=0),
              AdaBoostRegressor(random_state=0),
              RandomForestRegressor(random_state=0),
              ExtraTreesRegressor(random_state=0)]
    test_index_for_csv = 0
    os.system("echo 'here'")
    #    for on_all in range(7): #feature family types
    #
    #        for model_number in range(7): #models
    #            if (model_number == 5):
    #                continue
    #            if (model_number <= 3):
    #                b = True
    #            else:
    #                b = False
    #
    #            for archi in range(1, 3):
    archi = type_of_dataset
    if (model_number == 6):
        if (on_all == 1 and archi == 1):
            return
        elif (on_all == 4):
            return
        elif (on_all == 5):
            return
    if (model_number <= 3):
        b = True
    else:
        b = False

    os.system(f"echo 'Model: {model_number}, FF: {on_all}, Archi: {archi}'")
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    jaccard = []
    roc_auc = []
    average_precision = []  # this is 'precision_recall_curve'

    # for scores accross all folds:
    avg_accuracy_score_scores = []
    avg_precision_score_scores = []
    avg_recall_score_scores = []
    avg_f1_score_scores = []
    avg_jacard_score_scores = []
    avg_roc_auc_score_scores = []
    avg_avg_precision_score_scores = []

    accuracy_scores_heuristica = []
    precision_scores_heuristica = []
    recall_scores_heuristica = []
    f1_scores_heuristica = []
    jaccard_heuristica = []
    roc_auc_heuristica = []
    average_precision_heuristica = []  # this is 'precision_recall_curve'

    avg_accuracy_scores_heuristica_scores = []
    avg_precision_scores_heuristica_scores = []
    avg_recall_scores_heuristica_scores = []
    avg_f1_scores_heuristica_scores = []
    avg_jaccard_heuristica_scores = []
    avg_roc_auc_heuristica_scores = []
    avg_average_precision_heuristica_scores = []  # this is 'precision_recall_curve'

    # regression:
    explained_variance_scores = []
    max_error_scores = []
    neg_mean_absolute_error_scores = []
    neg_mean_squared_error_scores = []
    neg_mean_squared_log_error_scores = []
    r2_scores = []

    explained_variance_scores_scores = []
    max_error_scores_scores = []
    neg_mean_absolute_error_scores_scores = []
    neg_mean_squared_error_scores_scores = []
    neg_mean_squared_log_error_scores_scores = []
    r2_scores_scores = []

    explained_variance_scores_heuristica = []
    max_error_scores_heuristica = []
    neg_mean_absolute_error_scores_heuristica = []
    neg_mean_squared_error_scores_heuristica = []
    neg_mean_squared_log_error_scores_heuristica = []
    r2_scores_heuristica = []

    explained_variance_scores_heuristica_scores = []
    max_error_scores_heuristica_scores = []
    neg_mean_absolute_error_scores_heuristica_scores = []
    neg_mean_squared_error_scores_heuristica_scores = []
    neg_mean_squared_log_error_scores_heuristica_scores = []
    r2_scores_heuristica_scores = []
    if (archi == 0):
        all_files = [path_for_read + f"/out_concatted.csv"]  # maybe needs to change
    elif (archi == 1):
        # concatt the different dataframes with the right feature family and model
        # df_1 = "" need to do
        all_files = []
        for oper_number in range(2, 9):
            path_helper = path_for_read + f"/out_concatted_{oper_number}_{on_all}_{model_number}_{archi}.csv"
            all_files.append(path_helper)

    elif (archi == 2):
        all_files = []
        # for operator in range(1,8): un comment when we have all the data
        for operator in range(2, 9):
            if operator != 6:

                for value in operator_dict[operator]:
                    tot = operator_max[operator]
                    for i in range(1, tot + 1):
                        path = path_for_read + f"/out_operator_{operator}_{on_all}_{model_number}_{archi}_value_{value}_onNumber_{i}.csv"
                        all_files.append(path)
            else:
                for value in operator_dict[operator]:
                    tot = operator_max[operator]
                    for i in range(2, 9):
                        path = path_for_read + f"/out_operator_{operator}_{on_all}_{model_number}_{archi}_value_{value}_onNumber_{i}.csv"
                        all_files.append(path)

    concatted_df = pd.concat((pd.read_csv(f, index_col=False).dropna() for f in all_files), ignore_index=True)
    os.system(f"echo 'Model: {model_number}, FF: {on_all}, Archi: {archi} - Collecting Done!'")
    concatted_df['sort'] = concatted_df['doc_name'].str.extract('(\d+)', expand=False).astype(int)
    concatted_df['sort_2'] = concatted_df['doc_name'].str.extract('(_\d+)', expand=False)
    concatted_df['sort_2'] = concatted_df['sort_2'].replace({'_': ''}, regex=True)
    concatted_df.sort_values(['sort', 'sort_2'], inplace=True, ascending=[True, True])
    # print(x['sort'])
    concatted_df.drop('sort', axis=1, inplace=True)
    concatted_df.drop('sort_2', axis=1, inplace=True)
    avg_per_paper_per_fold = []
    best_per_paper_per_fold = []
    worst_per_paper_per_fold = []
    oracle_avg_per_paper_per_fold = []
    oracle_best_per_paper_per_fold = []
    oracle_worst_per_paper_per_fold = []
    heuristica_avg_per_paper_per_fold = []
    heuristica_best_per_paper_per_fold = []
    heuristica_worst_per_paper_per_fold = []
    os.system(f"echo 'Model: {model_number}, FF: {on_all}, Archi: {archi} - Starting folds'")
    for fold in range(5):
        names = concatted_df[concatted_df['came_from_fold'] == fold]['doc_name'].reset_index(drop=True)
        dict_for_doc_names_and_index = {}
        for i in range(len(names)):
            row = names.iloc[i]
            name = row.split('_')[0] + '_' + row.split('_')[1]
            if name not in dict_for_doc_names_and_index.keys():
                dict_for_doc_names_and_index[name] = [i]
            else:
                dict_for_doc_names_and_index[name].append(i)
        predictions = concatted_df[concatted_df['came_from_fold'] == fold]['prediction'].reset_index(drop=True)
        truth = concatted_df[concatted_df['came_from_fold'] == fold]['truth'].reset_index(drop=True).values.flatten()
        heuristica = concatted_df[concatted_df['came_from_fold'] == fold]['heuristica'].reset_index(drop=True)
        if (b):
            # binary:
            accuracy_scores.append(calc_eval(metrics.accuracy_score, predictions, truth, 'Accuracy'))
            precision_scores.append(calc_eval(metrics.precision_score, predictions, truth, 'Precision'))
            recall_scores.append(calc_eval(metrics.recall_score, predictions, truth, 'recall'))
            f1_scores.append(calc_eval(metrics.f1_score, predictions, truth, 'f1'))
            jaccard.append(calc_eval(metrics.jaccard_score, predictions, truth, 'jaccard'))
            try:
                roc_auc.append(calc_eval(metrics.roc_auc_score, predictions, truth, 'roc_auc'))
            except:
                roc_auc.append(0)
            average_precision.append(
                calc_eval(metrics.average_precision_score, predictions, truth, 'average_precision'))

            avg_accuracy_score = sum(accuracy_scores) / len(accuracy_scores)
            avg_precision_score = sum(precision_scores) / len(precision_scores)
            avg_recall_score = sum(recall_scores) / len(recall_scores)
            avg_f1_score = sum(f1_scores) / len(f1_scores)
            avg_jacard_score = sum(jaccard) / len(jaccard)
            avg_roc_auc_score = sum(roc_auc) / len(roc_auc)
            avg_avg_precision_score = sum(average_precision) / len(average_precision)

            avg_accuracy_score_scores.append(avg_accuracy_score)
            avg_precision_score_scores.append(avg_precision_score)
            avg_recall_score_scores.append(avg_recall_score)
            avg_f1_score_scores.append(avg_f1_score)
            avg_jacard_score_scores.append(avg_jacard_score)
            avg_roc_auc_score_scores.append(avg_roc_auc_score)
            avg_avg_precision_score_scores.append(avg_avg_precision_score)

            # for results from heuristica:

            accuracy_scores_heuristica.append(
                calc_eval(metrics.accuracy_score, heuristica, truth, 'Accuracy'))
            precision_scores_heuristica.append(
                calc_eval(metrics.precision_score, heuristica, truth, 'Precision'))
            recall_scores_heuristica.append(calc_eval(metrics.recall_score, heuristica, truth, 'recall'))
            f1_scores_heuristica.append(calc_eval(metrics.f1_score, heuristica, truth, 'f1'))
            jaccard_heuristica.append(calc_eval(metrics.jaccard_score, heuristica, truth, 'jaccard'))
            try:
                roc_auc_heuristica.append(calc_eval(metrics.roc_auc_score, heuristica, truth, 'roc_auc'))
            except:
                roc_auc_heuristica.append(0)
            average_precision_heuristica.append(
                calc_eval(metrics.average_precision_score, heuristica, truth, 'average_precision'))

            avg_accuracy_score_heuristica = sum(accuracy_scores_heuristica) / len(
                accuracy_scores_heuristica)
            avg_precision_score_heuristica = sum(precision_scores_heuristica) / len(
                precision_scores_heuristica)
            avg_recall_score_heuristica = sum(recall_scores_heuristica) / len(recall_scores_heuristica)
            avg_f1_score_heuristica = sum(f1_scores_heuristica) / len(f1_scores_heuristica)
            avg_jacard_score_heuristica = sum(jaccard_heuristica) / len(jaccard_heuristica)
            avg_roc_auc_score_heuristica = sum(roc_auc_heuristica) / len(roc_auc_heuristica)
            avg_avg_precision_score_heuristica = sum(average_precision_heuristica) / len(
                average_precision_heuristica)

            avg_accuracy_scores_heuristica_scores.append(avg_accuracy_score_heuristica)
            avg_precision_scores_heuristica_scores.append(avg_precision_score_heuristica)
            avg_recall_scores_heuristica_scores.append(avg_recall_score_heuristica)
            avg_f1_scores_heuristica_scores.append(avg_f1_score_heuristica)
            avg_jaccard_heuristica_scores.append(avg_jacard_score_heuristica)
            avg_roc_auc_heuristica_scores.append(avg_roc_auc_score_heuristica)
            avg_average_precision_heuristica_scores.append(avg_avg_precision_score_heuristica)

            # calulating avg_per_paper,best_per_paper,worst_per_paper,oracle_for_all_of_them,heursitica_for_all_of_them
            avg_per_paper = []
            best_per_paper = []
            worst_per_paper = []

            oracle_avg_per_paper = []
            oracle_best_per_paper = []
            oracle_worst_per_paper = []

            heuristica_avg_per_paper = []
            heuristica_best_per_paper = []
            heuristica_worst_per_paper = []
            for key, value in dict_for_doc_names_and_index.items():  # key - paper_id, value - array of indices of that paper
                temp_per_paper = []
                temp_oracle_per_paper = []
                temp_heuristica_per_paper = []
                for i in value:  # for each prediction from same paper
                    pred = predictions.loc[i]
                    true = truth[i]
                    heurstic = heuristica.loc[i]

                    temp_per_paper.append(pred)
                    temp_oracle_per_paper.append(true)
                    temp_heuristica_per_paper.append(heurstic)

                avg_per_paper.append(sum(temp_per_paper) / len(temp_per_paper))
                best_per_paper.append(max(temp_per_paper))
                worst_per_paper.append(min(temp_per_paper))

                oracle_avg_per_paper.append(sum(temp_oracle_per_paper) / len(temp_oracle_per_paper))
                oracle_best_per_paper.append(max(temp_oracle_per_paper))
                oracle_worst_per_paper.append(min(temp_oracle_per_paper))

                heuristica_avg_per_paper.append(sum(temp_heuristica_per_paper) / len(temp_heuristica_per_paper))
                heuristica_best_per_paper.append(max(temp_heuristica_per_paper))
                heuristica_worst_per_paper.append(min(temp_heuristica_per_paper))

            avg_per_paper_per_fold.append(sum(avg_per_paper) / len(avg_per_paper))
            best_per_paper_per_fold.append(sum(best_per_paper) / len(best_per_paper))
            worst_per_paper_per_fold.append(sum(worst_per_paper) / len(worst_per_paper))

            oracle_avg_per_paper_per_fold.append(sum(oracle_avg_per_paper) / len(oracle_avg_per_paper))
            oracle_best_per_paper_per_fold.append(sum(oracle_best_per_paper) / len(oracle_best_per_paper))
            oracle_worst_per_paper_per_fold.append(sum(oracle_worst_per_paper) / len(oracle_worst_per_paper))

            heuristica_avg_per_paper_per_fold.append(sum(heuristica_avg_per_paper) / len(heuristica_avg_per_paper))
            heuristica_best_per_paper_per_fold.append(sum(heuristica_best_per_paper) / len(heuristica_best_per_paper))
            heuristica_worst_per_paper_per_fold.append(
                sum(heuristica_worst_per_paper) / len(heuristica_worst_per_paper))



        else:
            explained_variance_scores.append(
                calc_eval(metrics.explained_variance_score, predictions, truth, 'explained_variance'))
            max_error_scores.append(calc_eval(metrics.max_error, predictions, truth, 'max_error'))
            neg_mean_absolute_error_scores.append(
                calc_eval(metrics.mean_absolute_error, predictions, truth, 'neg_mean_absolute_error'))
            neg_mean_squared_error_scores.append(
                calc_eval(metrics.mean_squared_error, predictions, truth, 'mean_squared_error'))
            neg_mean_squared_log_error_scores.append(
                calc_eval(metrics.mean_squared_log_error, predictions, truth, 'mean_squared_log_error'))
            r2_scores.append(calc_eval(metrics.r2_score, predictions, truth, 'r2_score'))

            avg_explained_variance_scores = sum(explained_variance_scores) / len(
                explained_variance_scores)
            avg_max_error_scores = sum(max_error_scores) / len(max_error_scores)
            avg_neg_mean_absolute_error_scores = sum(neg_mean_absolute_error_scores) / len(
                neg_mean_absolute_error_scores)
            avg_neg_mean_squared_error_scores = sum(neg_mean_squared_error_scores) / len(
                neg_mean_squared_error_scores)
            avg_neg_mean_squared_log_error_scores = sum(neg_mean_squared_log_error_scores) / len(
                neg_mean_squared_log_error_scores)
            avg_r2_scores = sum(r2_scores) / len(r2_scores)

            explained_variance_scores_scores.append(avg_explained_variance_scores)
            max_error_scores_scores.append(avg_max_error_scores)
            neg_mean_absolute_error_scores_scores.append(avg_neg_mean_absolute_error_scores)
            neg_mean_squared_error_scores_scores.append(avg_neg_mean_squared_error_scores)
            neg_mean_squared_log_error_scores_scores.append(avg_neg_mean_squared_log_error_scores)
            r2_scores_scores.append(avg_r2_scores)

            # for results from heuristica:

            explained_variance_scores_heuristica.append(
                calc_eval(metrics.explained_variance_score, heuristica, truth, 'explained_variance'))
            max_error_scores_heuristica.append(
                calc_eval(metrics.max_error, heuristica, truth, 'max_error'))
            neg_mean_absolute_error_scores_heuristica.append(
                calc_eval(metrics.mean_absolute_error, heuristica, truth, 'neg_mean_absolute_error'))
            neg_mean_squared_error_scores_heuristica.append(
                calc_eval(metrics.mean_squared_error, heuristica, truth, 'mean_squared_error'))
            neg_mean_squared_log_error_scores_heuristica.append(
                calc_eval(metrics.mean_squared_log_error, heuristica, truth, 'mean_squared_log_error'))
            r2_scores_heuristica.append(calc_eval(metrics.r2_score, heuristica, truth, 'r2_score'))

            avg_explained_variance_scores_heuristica = sum(explained_variance_scores_heuristica) / len(
                explained_variance_scores_heuristica)
            avg_max_error_scores_heuristica = sum(max_error_scores_heuristica) / len(
                max_error_scores_heuristica)
            avg_neg_mean_absolute_error_scores_heuristica = sum(
                neg_mean_absolute_error_scores_heuristica) / len(
                neg_mean_absolute_error_scores_heuristica)
            avg_neg_mean_squared_error_scores_heuristica = sum(
                neg_mean_squared_error_scores_heuristica) / len(
                neg_mean_squared_error_scores_heuristica)
            avg_neg_mean_squared_log_error_scores_heuristica = sum(
                neg_mean_squared_log_error_scores_heuristica) / len(
                neg_mean_squared_log_error_scores_heuristica)
            avg_r2_scores_heuristica = sum(r2_scores_heuristica) / len(r2_scores_heuristica)

            explained_variance_scores_heuristica_scores.append(avg_explained_variance_scores_heuristica)
            max_error_scores_heuristica_scores.append(avg_max_error_scores_heuristica)
            neg_mean_absolute_error_scores_heuristica_scores.append(
                avg_neg_mean_absolute_error_scores_heuristica)
            neg_mean_squared_error_scores_heuristica_scores.append(
                avg_neg_mean_squared_error_scores_heuristica)
            neg_mean_squared_log_error_scores_heuristica_scores.append(
                avg_neg_mean_squared_log_error_scores_heuristica)
            r2_scores_heuristica_scores.append(avg_r2_scores_heuristica)

            # calulating avg_per_paper,best_per_paper,worst_per_paper,oracle_for_all_of_them,heursitica_for_all_of_them
            avg_per_paper = []
            best_per_paper = []
            worst_per_paper = []

            oracle_avg_per_paper = []
            oracle_best_per_paper = []
            oracle_worst_per_paper = []

            heuristica_avg_per_paper = []
            heuristica_best_per_paper = []
            heuristica_worst_per_paper = []
            for key, value in dict_for_doc_names_and_index.items():  # key - paper_id, value - array of indices of that paper
                temp_per_paper = []
                temp_oracle_per_paper = []
                temp_heuristica_per_paper = []
                for i in value:  # for each prediction from same paper
                    pred = predictions.loc[i]
                    true = truth[i]
                    heurstic = heuristica.loc[i]

                    temp_per_paper.append(pred)
                    temp_oracle_per_paper.append(true)
                    temp_heuristica_per_paper.append(heurstic)

                avg_per_paper.append(sum(temp_per_paper) / len(temp_per_paper))
                best_per_paper.append(max(temp_per_paper))
                worst_per_paper.append(min(temp_per_paper))

                oracle_avg_per_paper.append(sum(temp_oracle_per_paper) / len(temp_oracle_per_paper))
                oracle_best_per_paper.append(max(temp_oracle_per_paper))
                oracle_worst_per_paper.append(min(temp_oracle_per_paper))

                heuristica_avg_per_paper.append(
                    sum(temp_heuristica_per_paper) / len(temp_heuristica_per_paper))
                heuristica_best_per_paper.append(max(temp_heuristica_per_paper))
                heuristica_worst_per_paper.append(min(temp_heuristica_per_paper))

            avg_per_paper_per_fold.append(sum(avg_per_paper) / len(avg_per_paper))
            best_per_paper_per_fold.append(sum(best_per_paper) / len(best_per_paper))
            worst_per_paper_per_fold.append(sum(worst_per_paper) / len(worst_per_paper))

            oracle_avg_per_paper_per_fold.append(sum(oracle_avg_per_paper) / len(oracle_avg_per_paper))
            oracle_best_per_paper_per_fold.append(sum(oracle_best_per_paper) / len(oracle_best_per_paper))
            oracle_worst_per_paper_per_fold.append(
                sum(oracle_worst_per_paper) / len(oracle_worst_per_paper))

            heuristica_avg_per_paper_per_fold.append(
                sum(heuristica_avg_per_paper) / len(heuristica_avg_per_paper))
            heuristica_best_per_paper_per_fold.append(
                sum(heuristica_best_per_paper) / len(heuristica_best_per_paper))
            heuristica_worst_per_paper_per_fold.append(
                sum(heuristica_worst_per_paper) / len(heuristica_worst_per_paper))

    # results from 5 folds:

    if (b):

        # binary:
        total_avg_accuracy_score_scores = sum(avg_accuracy_score_scores) / len(avg_accuracy_score_scores)
        total_avg_precision_score_scores = sum(avg_precision_score_scores) / len(avg_precision_score_scores)
        total_avg_recall_score_scores = sum(avg_recall_score_scores) / len(avg_recall_score_scores)
        total_avg_f1_score_scores = sum(avg_f1_score_scores) / len(avg_f1_score_scores)
        total_avg_jacard_score_scores = sum(avg_jacard_score_scores) / len(avg_jacard_score_scores)
        total_avg_roc_auc_score_scores = sum(avg_roc_auc_score_scores) / len(avg_roc_auc_score_scores)
        total_avg_avg_precision_score_scores = sum(avg_avg_precision_score_scores) / len(
            avg_avg_precision_score_scores)
        # heursitica:
        total_avg_accuracy_scores_heuristica_scores = sum(avg_accuracy_scores_heuristica_scores) / len(
            avg_accuracy_scores_heuristica_scores)
        total_avg_precision_scores_heuristica_scores = sum(avg_precision_scores_heuristica_scores) / len(
            avg_precision_scores_heuristica_scores)
        total_avg_recall_scores_heuristica_scores = sum(avg_recall_scores_heuristica_scores) / len(
            avg_recall_scores_heuristica_scores)
        total_avg_f1_scores_heuristica_scores = sum(avg_f1_scores_heuristica_scores) / len(
            avg_f1_scores_heuristica_scores)
        total_avg_jaccard_heuristica_scores = sum(avg_jaccard_heuristica_scores) / len(
            avg_jaccard_heuristica_scores)
        total_avg_roc_auc_heuristica_scores = sum(avg_roc_auc_heuristica_scores) / len(
            avg_roc_auc_heuristica_scores)
        total_avg_average_precision_heuristica_scores = sum(avg_average_precision_heuristica_scores) / len(
            avg_average_precision_heuristica_scores)

        # avg_per_paper and other metrics like that:
        avg_per_paper_on_5_folds = sum(avg_per_paper_per_fold) / len(avg_per_paper_per_fold)
        best_per_paper_on_5_folds = sum(best_per_paper_per_fold) / len(best_per_paper_per_fold)
        worst_per_paper_on_5_folds = sum(worst_per_paper_per_fold) / len(worst_per_paper_per_fold)

        oracle_avg_per_paper_on_5_folds = sum(oracle_avg_per_paper_per_fold) / len(oracle_avg_per_paper_per_fold)
        oracle_best_per_paper_on_5_folds = sum(oracle_best_per_paper_per_fold) / len(oracle_best_per_paper_per_fold)
        oracle_worst_per_paper_on_5_folds = sum(oracle_worst_per_paper_per_fold) / len(oracle_worst_per_paper_per_fold)

        herustic_avg_per_paper_on_5_folds = sum(heuristica_avg_per_paper_per_fold) / len(
            heuristica_avg_per_paper_per_fold)
        herustic_best_per_paper_on_5_folds = sum(heuristica_best_per_paper_per_fold) / len(
            heuristica_best_per_paper_per_fold)
        herustic_worst_per_paper_on_5_folds = sum(heuristica_worst_per_paper_per_fold) / len(
            heuristica_worst_per_paper_per_fold)

        results_df.at[Rows[test_index_for_csv], "from_what_architecture"] = archi
        results_df.at[Rows[test_index_for_csv], "feature_family"] = features_family_dct[on_all]
        results_df.at[Rows[test_index_for_csv], "class"] = "binary"
        results_df.at[Rows[test_index_for_csv], "model"] = type(models[model_number]).__name__

        results_df.at[Rows[test_index_for_csv], "accuracy"] = total_avg_accuracy_score_scores
        results_df.at[Rows[test_index_for_csv], "precision"] = total_avg_precision_score_scores
        results_df.at[Rows[test_index_for_csv], "recall"] = total_avg_recall_score_scores
        results_df.at[Rows[test_index_for_csv], "f1"] = total_avg_f1_score_scores
        results_df.at[Rows[test_index_for_csv], "jaccard"] = total_avg_jacard_score_scores
        results_df.at[Rows[test_index_for_csv], "roc_auc"] = total_avg_roc_auc_score_scores
        results_df.at[
            Rows[test_index_for_csv], "average_precision"] = total_avg_avg_precision_score_scores
        results_df.at[Rows[test_index_for_csv], "explained_variance"] = 0
        results_df.at[Rows[test_index_for_csv], "max_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_mean_absolute_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_mean_squared_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_mean_squared_log_error"] = 0
        results_df.at[Rows[test_index_for_csv], "r2"] = 0
        results_df.at[Rows[
            test_index_for_csv], "heuristic_accuracy"] = total_avg_accuracy_scores_heuristica_scores
        results_df.at[Rows[
            test_index_for_csv], "heuristic_precision"] = total_avg_precision_scores_heuristica_scores
        results_df.at[
            Rows[test_index_for_csv], "heuristic_recall"] = total_avg_recall_scores_heuristica_scores
        results_df.at[Rows[test_index_for_csv], "heuristic_f1"] = total_avg_f1_scores_heuristica_scores
        results_df.at[
            Rows[test_index_for_csv], "heuristic_jaccard"] = total_avg_jaccard_heuristica_scores
        results_df.at[
            Rows[test_index_for_csv], "heuristic_roc_auc"] = total_avg_roc_auc_heuristica_scores
        results_df.at[Rows[
            test_index_for_csv], "heuristic_average_precision"] = total_avg_average_precision_heuristica_scores
        results_df.at[Rows[test_index_for_csv], "heuristic_explained_variance"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_max_error"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_neg_mean_absolute_error"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_neg_mean_squared_error"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_neg_mean_squared_log_error"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_r2"] = 0
        acc_diff = results_df.at[Rows[test_index_for_csv], "accuracy"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_accuracy"]
        if (acc_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_accuracy"] = acc_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_accuracy"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_accuracy"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_accuracy"] = abs(acc_diff)
        precision_diff = results_df.at[Rows[test_index_for_csv], "precision"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_precision"]
        if (precision_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_precision"] = precision_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_precision"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_precision"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_precision"] = abs(precision_diff)
        recall_diff = results_df.at[Rows[test_index_for_csv], "recall"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_recall"]
        if (recall_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_recall"] = recall_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_recall"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_recall"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_recall"] = abs(recall_diff)
        f1_diff = results_df.at[Rows[test_index_for_csv], "f1"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_f1"]
        if (f1_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_f1"] = f1_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_f1"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_f1"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_f1"] = abs(f1_diff)
        jaccard_diff = results_df.at[Rows[test_index_for_csv], "jaccard"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_jaccard"]
        if (jaccard_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_jaccard"] = jaccard_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_jaccard"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_jaccard"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_jaccard"] = abs(jaccard_diff)
        roc_auc_diff = results_df.at[Rows[test_index_for_csv], "roc_auc"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_roc_auc"]
        if (roc_auc_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_roc_auc"] = roc_auc_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_roc_auc"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_roc_auc"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_roc_auc"] = abs(roc_auc_diff)
        average_precision_diff = results_df.at[Rows[test_index_for_csv], "average_precision"] - \
                                 results_df.at[
                                     Rows[test_index_for_csv], "heuristic_average_precision"]
        if (average_precision_diff > 0):  # prediction better
            results_df.at[
                Rows[test_index_for_csv], "pos_diff_average_precision"] = average_precision_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_average_precision"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_average_precision"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_average_precision"] = abs(
                average_precision_diff)
        results_df.at[Rows[test_index_for_csv], "pos_diff_explained_variance"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_max_error"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_neg_mean_absolute_error"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_neg_mean_squared_error"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_neg_mean_squared_log_error"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_r2"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_explained_variance"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_max_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_absolute_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_squared_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_squared_log_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_r2"] = 0
        # "AVG_per_paper","BEST_per_paper","WORST_per_paper","ORACLE_AVG_per_paper","ORACLE_BEST_per_paper","ORACLE_WORST_per_paper","HEURISTIC_AVG_per_paper","HEURISTIC_BEST_per_paper","HEURISTIC_WORST_per_paper"]
        results_df.at[Rows[test_index_for_csv], "AVG_per_paper"] = avg_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "BEST_per_paper"] = best_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "WORST_per_paper"] = worst_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "ORACLE_AVG_per_paper"] = oracle_avg_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "ORACLE_BEST_per_paper"] = oracle_best_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "ORACLE_WORST_per_paper"] = oracle_worst_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "HEURISTIC_AVG_per_paper"] = herustic_avg_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "HEURISTIC_BEST_per_paper"] = herustic_best_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "HEURISTIC_WORST_per_paper"] = herustic_worst_per_paper_on_5_folds
        test_index_for_csv += 1

        # NEW ROW FOR HEURISTICA:
        results_df.at[Rows[test_index_for_csv], "from_what_architecture"] = archi
        results_df.at[Rows[test_index_for_csv], "feature_family"] = features_family_dct[on_all]
        results_df.at[Rows[test_index_for_csv], "class"] = "binary"
        results_df.at[Rows[test_index_for_csv], "model"] = "heuristica"

        results_df.at[Rows[test_index_for_csv], "accuracy"] = total_avg_accuracy_scores_heuristica_scores
        results_df.at[Rows[test_index_for_csv], "precision"] = total_avg_precision_scores_heuristica_scores
        results_df.at[Rows[test_index_for_csv], "recall"] = total_avg_recall_scores_heuristica_scores
        results_df.at[Rows[test_index_for_csv], "f1"] = total_avg_f1_scores_heuristica_scores
        results_df.at[Rows[test_index_for_csv], "jaccard"] = total_avg_jaccard_heuristica_scores
        results_df.at[Rows[test_index_for_csv], "roc_auc"] = total_avg_roc_auc_heuristica_scores
        results_df.at[
            Rows[test_index_for_csv], "average_precision"] = total_avg_average_precision_heuristica_scores
        results_df.at[Rows[test_index_for_csv], "explained_variance"] = 0
        results_df.at[Rows[test_index_for_csv], "max_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_mean_absolute_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_mean_squared_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_mean_squared_log_error"] = 0
        results_df.at[Rows[test_index_for_csv], "r2"] = 0
        results_df.at[Rows[
            test_index_for_csv], "heuristic_accuracy"] = 0
        results_df.at[Rows[
            test_index_for_csv], "heuristic_precision"] = 0
        results_df.at[
            Rows[test_index_for_csv], "heuristic_recall"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_f1"] = 0
        results_df.at[
            Rows[test_index_for_csv], "heuristic_jaccard"] = 0
        results_df.at[
            Rows[test_index_for_csv], "heuristic_roc_auc"] = 0
        results_df.at[Rows[
            test_index_for_csv], "heuristic_average_precision"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_explained_variance"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_max_error"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_neg_mean_absolute_error"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_neg_mean_squared_error"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_neg_mean_squared_log_error"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_r2"] = 0
        acc_diff = results_df.at[Rows[test_index_for_csv], "accuracy"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_accuracy"]
        if (acc_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_accuracy"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_accuracy"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_accuracy"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_accuracy"] = 0
        precision_diff = results_df.at[Rows[test_index_for_csv], "precision"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_precision"]
        if (precision_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_precision"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_precision"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_precision"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_precision"] = 0
        recall_diff = results_df.at[Rows[test_index_for_csv], "recall"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_recall"]
        if (recall_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_recall"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_recall"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_recall"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_recall"] = 0
        f1_diff = results_df.at[Rows[test_index_for_csv], "f1"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_f1"]
        if (f1_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_f1"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_f1"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_f1"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_f1"] = 0
        jaccard_diff = results_df.at[Rows[test_index_for_csv], "jaccard"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_jaccard"]
        if (jaccard_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_jaccard"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_jaccard"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_jaccard"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_jaccard"] = 0
        roc_auc_diff = results_df.at[Rows[test_index_for_csv], "roc_auc"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_roc_auc"]
        if (roc_auc_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_roc_auc"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_roc_auc"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_roc_auc"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_roc_auc"] = 0
            average_precision_diff = results_df.at[Rows[test_index_for_csv], "average_precision"] - results_df.at[
                Rows[test_index_for_csv], "heuristic_average_precision"]
        if (average_precision_diff > 0):  # prediction better
            results_df.at[
                Rows[test_index_for_csv], "pos_diff_average_precision"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_average_precision"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_average_precision"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_average_precision"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_explained_variance"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_max_error"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_neg_mean_absolute_error"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_neg_mean_squared_error"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_neg_mean_squared_log_error"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_r2"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_explained_variance"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_max_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_absolute_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_squared_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_squared_log_error"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_r2"] = 0
        # "AVG_per_paper","BEST_per_paper","WORST_per_paper","ORACLE_AVG_per_paper","ORACLE_BEST_per_paper","ORACLE_WORST_per_paper","HEURISTIC_AVG_per_paper","HEURISTIC_BEST_per_paper","HEURISTIC_WORST_per_paper"]
        results_df.at[Rows[test_index_for_csv], "AVG_per_paper"] = 0
        results_df.at[Rows[test_index_for_csv], "BEST_per_paper"] = 0
        results_df.at[Rows[test_index_for_csv], "WORST_per_paper"] = 0
        results_df.at[Rows[test_index_for_csv], "ORACLE_AVG_per_paper"] = 0
        results_df.at[Rows[test_index_for_csv], "ORACLE_BEST_per_paper"] = 0
        results_df.at[Rows[test_index_for_csv], "ORACLE_WORST_per_paper"] = 0
        results_df.at[Rows[test_index_for_csv], "HEURISTIC_AVG_per_paper"] = herustic_avg_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "HEURISTIC_BEST_per_paper"] = herustic_best_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "HEURISTIC_WORST_per_paper"] = herustic_worst_per_paper_on_5_folds
        test_index_for_csv += 1
    else:
        total_explained_variance_scores_scores = sum(explained_variance_scores_scores) / len(
            explained_variance_scores_scores)
        total_max_error_scores_scores = sum(max_error_scores_scores) / len(max_error_scores_scores)
        total_neg_mean_absolute_error_scores_scores = sum(neg_mean_absolute_error_scores_scores) / len(
            neg_mean_absolute_error_scores_scores)
        total_neg_mean_squared_error_scores_scores = sum(neg_mean_squared_error_scores_scores) / len(
            neg_mean_squared_error_scores_scores)
        total_neg_mean_squared_log_error_scores_scores = sum(
            neg_mean_squared_log_error_scores_scores) / len(neg_mean_squared_log_error_scores_scores)
        total_r2_scores_scores = sum(r2_scores_scores) / len(r2_scores_scores)

        # heuristica:
        total_explained_variance_scores_heuristica_scores = sum(
            explained_variance_scores_heuristica_scores) / len(explained_variance_scores_heuristica_scores)
        total_max_error_scores_heuristica_scores = sum(max_error_scores_heuristica_scores) / len(
            max_error_scores_heuristica_scores)
        total_neg_mean_absolute_error_scores_heuristica_scores = sum(
            neg_mean_absolute_error_scores_heuristica_scores) / len(
            neg_mean_absolute_error_scores_heuristica_scores)
        total_neg_mean_squared_error_scores_heuristica_scores = sum(
            neg_mean_squared_error_scores_heuristica_scores) / len(
            neg_mean_squared_error_scores_heuristica_scores)
        total_neg_mean_squared_log_error_scores_heuristica_scores = sum(
            neg_mean_squared_log_error_scores_heuristica_scores) / len(
            neg_mean_squared_log_error_scores_heuristica_scores)
        total_r2_scores_heuristica_scores = sum(r2_scores_heuristica_scores) / len(
            r2_scores_heuristica_scores)

        # avg_per_paper and other metrics like that:
        avg_per_paper_on_5_folds = sum(avg_per_paper_per_fold) / len(avg_per_paper_per_fold)
        best_per_paper_on_5_folds = sum(best_per_paper_per_fold) / len(best_per_paper_per_fold)
        worst_per_paper_on_5_folds = sum(worst_per_paper_per_fold) / len(worst_per_paper_per_fold)

        oracle_avg_per_paper_on_5_folds = sum(oracle_avg_per_paper_per_fold) / len(
            oracle_avg_per_paper_per_fold)
        oracle_best_per_paper_on_5_folds = sum(oracle_best_per_paper_per_fold) / len(
            oracle_best_per_paper_per_fold)
        oracle_worst_per_paper_on_5_folds = sum(oracle_worst_per_paper_per_fold) / len(
            oracle_worst_per_paper_per_fold)

        herustic_avg_per_paper_on_5_folds = sum(heuristica_avg_per_paper_per_fold) / len(
            heuristica_avg_per_paper_per_fold)
        herustic_best_per_paper_on_5_folds = sum(heuristica_best_per_paper_per_fold) / len(
            heuristica_best_per_paper_per_fold)
        herustic_worst_per_paper_on_5_folds = sum(heuristica_worst_per_paper_per_fold) / len(
            heuristica_worst_per_paper_per_fold)

        results_df.at[Rows[test_index_for_csv], "from_what_architecture"] = archi
        results_df.at[Rows[test_index_for_csv], "feature_family"] = features_family_dct[on_all]
        results_df.at[Rows[test_index_for_csv], "class"] = "regression"
        results_df.at[Rows[test_index_for_csv], "model"] = type(models[model_number]).__name__

        results_df.at[Rows[test_index_for_csv], "accuracy"] = 0
        results_df.at[Rows[test_index_for_csv], "precision"] = 0
        results_df.at[Rows[test_index_for_csv], "recall"] = 0
        results_df.at[Rows[test_index_for_csv], "f1"] = 0
        results_df.at[Rows[test_index_for_csv], "jaccard"] = 0
        results_df.at[Rows[test_index_for_csv], "roc_auc"] = 0
        results_df.at[Rows[test_index_for_csv], "average_precision"] = 0
        results_df.at[
            Rows[test_index_for_csv], "explained_variance"] = total_explained_variance_scores_scores
        results_df.at[Rows[test_index_for_csv], "max_error"] = total_max_error_scores_scores
        results_df.at[Rows[
            test_index_for_csv], "neg_mean_absolute_error"] = total_neg_mean_absolute_error_scores_scores
        results_df.at[Rows[
            test_index_for_csv], "neg_mean_squared_error"] = total_neg_mean_squared_error_scores_scores
        results_df.at[Rows[
            test_index_for_csv], "neg_mean_squared_log_error"] = total_neg_mean_squared_log_error_scores_scores
        results_df.at[Rows[test_index_for_csv], "r2"] = total_r2_scores_scores

        results_df.at[Rows[test_index_for_csv], "heuristic_accuracy"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_precision"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_recall"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_f1"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_jaccard"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_roc_auc"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_average_precision"] = 0
        results_df.at[Rows[
            test_index_for_csv], "heuristic_explained_variance"] = total_explained_variance_scores_heuristica_scores
        results_df.at[
            Rows[test_index_for_csv], "heuristic_max_error"] = total_max_error_scores_heuristica_scores
        results_df.at[Rows[
            test_index_for_csv], "heuristic_neg_mean_absolute_error"] = total_neg_mean_absolute_error_scores_heuristica_scores
        results_df.at[Rows[
            test_index_for_csv], "heuristic_neg_mean_squared_error"] = total_neg_mean_squared_error_scores_heuristica_scores
        results_df.at[Rows[
            test_index_for_csv], "heuristic_neg_mean_squared_log_error"] = total_neg_mean_squared_log_error_scores_heuristica_scores
        results_df.at[Rows[test_index_for_csv], "heuristic_r2"] = total_r2_scores_heuristica_scores

        explained_variance_diff = results_df.at[Rows[test_index_for_csv], "explained_variance"] - \
                                  results_df.at[
                                      Rows[test_index_for_csv], "heuristic_explained_variance"]
        if (explained_variance_diff > 0):  # prediction better
            results_df.at[
                Rows[test_index_for_csv], "pos_diff_explained_variance"] = explained_variance_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_explained_variance"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_explained_variance"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_explained_variance"] = abs(
                explained_variance_diff)
        max_error_diff = results_df.at[Rows[test_index_for_csv], "max_error"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_max_error"]
        if (max_error_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_max_error"] = max_error_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_max_error"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_max_error"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_max_error"] = abs(max_error_diff)
        neg_mean_absolute_error_diff = results_df.at[
                                           Rows[test_index_for_csv], "neg_mean_absolute_error"] - \
                                       results_df.at[Rows[
                                           test_index_for_csv], "heuristic_neg_mean_absolute_error"]
        if (neg_mean_absolute_error_diff > 0):  # prediction better
            results_df.at[Rows[
                test_index_for_csv], "pos_diff_neg_mean_absolute_error"] = neg_mean_absolute_error_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_absolute_error"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_neg_mean_absolute_error"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_absolute_error"] = abs(
                neg_mean_absolute_error_diff)
        neg_mean_squared_error_diff = results_df.at[
                                          Rows[test_index_for_csv], "neg_mean_squared_error"] - \
                                      results_df.at[
                                          Rows[test_index_for_csv], "heuristic_neg_mean_squared_error"]
        if (neg_mean_squared_error_diff > 0):  # prediction better
            results_df.at[Rows[
                test_index_for_csv], "pos_diff_neg_mean_squared_error"] = neg_mean_squared_error_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_squared_error"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_neg_mean_squared_error"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_squared_error"] = abs(
                neg_mean_squared_error_diff)
        neg_mean_squared_log_error_diff = results_df.at[
                                              Rows[test_index_for_csv], "neg_mean_squared_log_error"] - \
                                          results_df.at[Rows[
                                              test_index_for_csv], "heuristic_neg_mean_squared_log_error"]
        if (neg_mean_squared_log_error_diff > 0):  # prediction better
            results_df.at[Rows[
                test_index_for_csv], "pos_diff_neg_mean_squared_log_error"] = neg_mean_squared_log_error_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_squared_log_error"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_neg_mean_squared_log_error"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_squared_log_error"] = abs(
                neg_mean_squared_log_error_diff)
        r2_diff = results_df.at[Rows[test_index_for_csv], "r2"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_r2"]
        if (r2_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_r2"] = r2_diff
            results_df.at[Rows[test_index_for_csv], "neg_diff_r2"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_r2"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_r2"] = r2_diff
        results_df.at[Rows[test_index_for_csv], "pos_diff_accuracy"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_precision"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_recall"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_f1"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_jaccard"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_roc_auc"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_average_precision"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_accuracy"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_precision"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_recall"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_f1"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_jaccard"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_roc_auc"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_average_precision"] = 0

        # "AVG_per_paper","BEST_per_paper","WORST_per_paper","ORACLE_AVG_per_paper","ORACLE_BEST_per_paper","ORACLE_WORST_per_paper","HEURISTIC_AVG_per_paper","HEURISTIC_BEST_per_paper","HEURISTIC_WORST_per_paper"]
        results_df.at[Rows[test_index_for_csv], "AVG_per_paper"] = avg_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "BEST_per_paper"] = best_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "WORST_per_paper"] = worst_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "ORACLE_AVG_per_paper"] = oracle_avg_per_paper_on_5_folds
        results_df.at[Rows[test_index_for_csv], "ORACLE_BEST_per_paper"] = oracle_best_per_paper_on_5_folds
        results_df.at[
            Rows[test_index_for_csv], "ORACLE_WORST_per_paper"] = oracle_worst_per_paper_on_5_folds
        results_df.at[
            Rows[test_index_for_csv], "HEURISTIC_AVG_per_paper"] = herustic_avg_per_paper_on_5_folds
        results_df.at[
            Rows[test_index_for_csv], "HEURISTIC_BEST_per_paper"] = herustic_best_per_paper_on_5_folds
        results_df.at[
            Rows[test_index_for_csv], "HEURISTIC_WORST_per_paper"] = herustic_worst_per_paper_on_5_folds

        test_index_for_csv += 1

        # FOR HEURSITCA LINE:
        results_df.at[Rows[test_index_for_csv], "from_what_architecture"] = archi
        results_df.at[Rows[test_index_for_csv], "feature_family"] = features_family_dct[on_all]
        results_df.at[Rows[test_index_for_csv], "class"] = "regression"
        results_df.at[Rows[test_index_for_csv], "model"] = "heuristica"

        results_df.at[Rows[test_index_for_csv], "accuracy"] = 0
        results_df.at[Rows[test_index_for_csv], "precision"] = 0
        results_df.at[Rows[test_index_for_csv], "recall"] = 0
        results_df.at[Rows[test_index_for_csv], "f1"] = 0
        results_df.at[Rows[test_index_for_csv], "jaccard"] = 0
        results_df.at[Rows[test_index_for_csv], "roc_auc"] = 0
        results_df.at[Rows[test_index_for_csv], "average_precision"] = 0
        results_df.at[
            Rows[test_index_for_csv], "explained_variance"] = total_explained_variance_scores_heuristica_scores
        results_df.at[Rows[test_index_for_csv], "max_error"] = total_max_error_scores_heuristica_scores
        results_df.at[Rows[
            test_index_for_csv], "neg_mean_absolute_error"] = total_neg_mean_absolute_error_scores_heuristica_scores
        results_df.at[Rows[
            test_index_for_csv], "neg_mean_squared_error"] = total_neg_mean_squared_error_scores_heuristica_scores
        results_df.at[Rows[
            test_index_for_csv], "neg_mean_squared_log_error"] = total_neg_mean_squared_log_error_scores_heuristica_scores
        results_df.at[Rows[test_index_for_csv], "r2"] = total_r2_scores_heuristica_scores

        results_df.at[Rows[test_index_for_csv], "heuristic_accuracy"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_precision"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_recall"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_f1"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_jaccard"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_roc_auc"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_average_precision"] = 0
        results_df.at[Rows[
            test_index_for_csv], "heuristic_explained_variance"] = 0
        results_df.at[
            Rows[test_index_for_csv], "heuristic_max_error"] = 0
        results_df.at[Rows[
            test_index_for_csv], "heuristic_neg_mean_absolute_error"] = 0
        results_df.at[Rows[
            test_index_for_csv], "heuristic_neg_mean_squared_error"] = 0
        results_df.at[Rows[
            test_index_for_csv], "heuristic_neg_mean_squared_log_error"] = 0
        results_df.at[Rows[test_index_for_csv], "heuristic_r2"] = 0

        explained_variance_diff = results_df.at[Rows[test_index_for_csv], "explained_variance"] - \
                                  results_df.at[
                                      Rows[test_index_for_csv], "heuristic_explained_variance"]
        if (explained_variance_diff > 0):  # prediction better
            results_df.at[
                Rows[test_index_for_csv], "pos_diff_explained_variance"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_explained_variance"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_explained_variance"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_explained_variance"] = 0
        max_error_diff = results_df.at[Rows[test_index_for_csv], "max_error"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_max_error"]
        if (max_error_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_max_error"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_max_error"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_max_error"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_max_error"] = 0
        neg_mean_absolute_error_diff = results_df.at[
                                           Rows[test_index_for_csv], "neg_mean_absolute_error"] - \
                                       results_df.at[Rows[
                                           test_index_for_csv], "heuristic_neg_mean_absolute_error"]
        if (neg_mean_absolute_error_diff > 0):  # prediction better
            results_df.at[Rows[
                test_index_for_csv], "pos_diff_neg_mean_absolute_error"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_absolute_error"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_neg_mean_absolute_error"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_absolute_error"] = 0
        neg_mean_squared_error_diff = results_df.at[
                                          Rows[test_index_for_csv], "neg_mean_squared_error"] - \
                                      results_df.at[
                                          Rows[test_index_for_csv], "heuristic_neg_mean_squared_error"]
        if (neg_mean_squared_error_diff > 0):  # prediction better
            results_df.at[Rows[
                test_index_for_csv], "pos_diff_neg_mean_squared_error"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_squared_error"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_neg_mean_squared_error"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_squared_error"] = 0
        neg_mean_squared_log_error_diff = results_df.at[
                                              Rows[test_index_for_csv], "neg_mean_squared_log_error"] - \
                                          results_df.at[Rows[
                                              test_index_for_csv], "heuristic_neg_mean_squared_log_error"]
        if (neg_mean_squared_log_error_diff > 0):  # prediction better
            results_df.at[Rows[
                test_index_for_csv], "pos_diff_neg_mean_squared_log_error"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_squared_log_error"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_neg_mean_squared_log_error"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_neg_mean_squared_log_error"] = 0
        r2_diff = results_df.at[Rows[test_index_for_csv], "r2"] - results_df.at[
            Rows[test_index_for_csv], "heuristic_r2"]
        if (r2_diff > 0):  # prediction better
            results_df.at[Rows[test_index_for_csv], "pos_diff_r2"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_r2"] = 0
        else:
            results_df.at[Rows[test_index_for_csv], "pos_diff_r2"] = 0
            results_df.at[Rows[test_index_for_csv], "neg_diff_r2"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_accuracy"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_precision"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_recall"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_f1"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_jaccard"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_roc_auc"] = 0
        results_df.at[Rows[test_index_for_csv], "pos_diff_average_precision"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_accuracy"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_precision"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_recall"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_f1"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_jaccard"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_roc_auc"] = 0
        results_df.at[Rows[test_index_for_csv], "neg_diff_average_precision"] = 0

        # "AVG_per_paper","BEST_per_paper","WORST_per_paper","ORACLE_AVG_per_paper","ORACLE_BEST_per_paper","ORACLE_WORST_per_paper","HEURISTIC_AVG_per_paper","HEURISTIC_BEST_per_paper","HEURISTIC_WORST_per_paper"]
        results_df.at[Rows[test_index_for_csv], "AVG_per_paper"] = 0
        results_df.at[Rows[test_index_for_csv], "BEST_per_paper"] = 0
        results_df.at[Rows[test_index_for_csv], "WORST_per_paper"] = 0
        results_df.at[Rows[test_index_for_csv], "ORACLE_AVG_per_paper"] = 0
        results_df.at[Rows[test_index_for_csv], "ORACLE_BEST_per_paper"] = 0
        results_df.at[
            Rows[test_index_for_csv], "ORACLE_WORST_per_paper"] = 0
        results_df.at[
            Rows[test_index_for_csv], "HEURISTIC_AVG_per_paper"] = herustic_avg_per_paper_on_5_folds
        results_df.at[
            Rows[test_index_for_csv], "HEURISTIC_BEST_per_paper"] = herustic_best_per_paper_on_5_folds
        results_df.at[
            Rows[test_index_for_csv], "HEURISTIC_WORST_per_paper"] = herustic_worst_per_paper_on_5_folds

        test_index_for_csv += 1

    results_df.to_csv(path_for_save, index=False)


if __name__ == "__main__":
    # batch_operator = int(sys.argv[1])
    on_all = int(sys.argv[2])
    model_number = int(sys.argv[3])
    path_for_read = sys.argv[4]
    path_for_write = sys.argv[5]
    type_of_dataset = int(sys.argv[6])

    function_for_strucure_3(on_all, model_number, path_for_read, path_for_write, type_of_dataset)
