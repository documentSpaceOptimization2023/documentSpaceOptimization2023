from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, \
    ExtraTreesRegressor, RandomForestRegressor
import sklearn

print('The scikit-learn version is {}.'.format(sklearn.__version__))

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from kfoldgroup import StratifiedGroupKFold
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import copy
import sys
import os


# read csv here


def clean_data(target, model_type):  # here remove df
    global df
    global y
    global Y_df
    global x_name
    os.system(f"echo A")
    print(df.columns)
    print("8b")
    # df.fillna(df.mean(numeric_only=True), inplace=True)  # replace missing values with mean of the feature
    #df = df[df[target] >= 0]  # remove rows with -1 in target
    rows_to_drop = y.index[y[target] == -1].tolist()
    # print(rows_to_drop)
    # df = df[df[target] >= 0]  # remove rows with -1 in target
    y.drop(rows_to_drop, inplace=True)
    df.drop(rows_to_drop, inplace=True)
    print(df.columns)
    print("8c")
    y.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df.head(3))
    print(df.columns)
    print("8d")
    Y_df.drop(rows_to_drop, inplace=True)
    Y_df.reset_index(drop=True, inplace=True)

    os.system(f"echo B")
    # if (operator != 0):
    #     df = df[df['type'] == operator]
    #df.drop(columns=[target], axis=1, inplace=True)
    os.system(f"echo C")
    groups = []
    # print(df.head(3))
    # print(df.to_numpy())
    #os.system(f"echo {df.head(3)}")
    groupcol = df[x_name[0]]
    for row in groupcol:
        #print(row)
        filename = row.split('_')[0] + '_' + row.split('_')[1]
        groups.append(filename)
    #    for row in df.to_numpy():
    #        #
    #        filename = row[0].split('_')[0] + '_' + row[0].split('_')[1]
    #        groups.append(filename)
    os.system(f"echo D")
    groups = np.array(groups)
    os.system(f"echo E")
    #    for index,row in df.iterrows():  # get all latex file names to support split by document
    #        print(row)
    #        print(index)
    #        #filename = row[0].split('_')[0] + '_' + row[0].split('_')[1]
    #        filename = row["Unnamed 0.1.1"].split('_')[0] + '_' + row["Unnamed 0.1.1"].split('_')[1]
    #        groups.append(filename)
    #    groups = np.array(groups)

    # df_with_all_columns = copy.deepcopy(df)

    os.system(f"echo E1")
    #y.drop(columns=['y_gained', 'herustica', "lines_we_gained"], axis=1, inplace=True)
    os.system(f"echo E2")
    y = y.to_numpy()

    os.system(f"echo F")

    os.system(f"echo G")
    df = df.select_dtypes(['number'])  # remove non-numeric columns
    os.system(f"echo H")
    # y_reg = df[target_reg].values
    # y = df[target].values  # separate target values

    # if to_drop:
    #     df.drop(columns=[col for col in to_drop], axis=1, inplace=True)  # remove irrelevant features

    # if features_family == 1:
    #     df = df[features_family_cols_to_use[1]]
    # elif features_family == 2:
    #     df = df[features_family_cols_to_use[2]]
    # elif features_family == 3:
    #     df = df[features_family_cols_to_use[3]]
    # elif features_family == 4:
    #     df.drop(columns=[col for col in features_family_cols_to_use[2]], axis=1, inplace=True)

    # if target in df.columns:
    #     X = df.drop(columns=[target], axis=1)  # create a dataframe with all training data except the target column #here reolace X with df inplace
    # else:
    #     X = df #here replace
    if model_type == "classification":
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    elif model_type == "regression":
        sgkf = GroupKFold(n_splits=5)
    else:
        sgkf = None

    print("here")
    os.system(f"echo I")
    return sgkf, groups  # here remove X


import pickle


######################### add this functions (write_list, read_list, cv_generator) #############################
# write list to binary file
def write_list(path, a_list):
    with open(path, 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')


# Read list to memory
def read_list(path):
    with open(path, 'rb') as fp:
        n_list = pickle.load(fp)
        # print(n_list)
        return n_list


def cv_generator(folds, groups):  # generate splits based on the big split
    for i in range(5):
        train, test = [], []
        curr_fold = folds[i]
        for j in range(len(groups)):
            if groups[j] in curr_fold:
                if curr_fold[groups[j]] == True:
                    train.append(j)
                elif curr_fold[groups[j]] == False:
                    test.append(j)
                else:
                    print("err")
            else:
                train.append(j)

        yield np.array(train), np.array(test)


def simpleModel(estimator, scoring, folds, groups,
                params=None):  ################################ #here remove X
    global df
    global y

    # cv = folds.split(X, y, groups)
    cv = cv_generator(folds, groups)
    cv_results = cross_validate(estimator, df, y, cv=cv, scoring=scoring, return_estimator=True)  # replace X with df
    fitted_models = cv_results.pop('estimator')
    for key in cv_results:
        cv_results[key] = np.average(abs(cv_results[key]))
    # print("\tResults (average of the cross validation): ", cv_results)

    return fitted_models, cv_results


def get_best_prediction(prediction):
    return max(prediction)


def get_worst_prediction(prediction):
    return min(prediction)


def calc_eval(eval_mat, y_from_herustica, y_true, eval_str):
    score = eval_mat(y_true, y_from_herustica)
    print("the score for herustica :" + str(score) + " " + eval_str)
    return score


cols = [
    'num_of_pars_in_col_1', 'num_of_pars_in_col_2',
    'num_of_paragraphs_in_col_1', 'num_of_paragraphs_in_col_2',
    'num_of_algorithms_in_col_1', 'num_of_algorithms_in_col_2',
    'num_of_formulas_in_col_1', 'num_of_formulas_in_col_2',
    'num_of_figures_in_col_1', 'num_of_figures_in_col_2',
    'num_of_tables_in_col_1', 'num_of_tables_in_col_2']
total_pdf = ['ending_y_of_doc', 'max_lines_par', 'min_lines_par', 'max_lines_enum', 'min_lines_enum',
             'max_lines_caption', 'min_lines_caption', 'max_figure_y_space', 'min_figure_y_space',
             'max_table_y_space', 'min_table_y_space',
             'max_height_object', 'min_height_object', 'num_of_elements',
             'num_of_pars', 'num_of_pars_in_col_1', 'num_of_pars_in_col_2',
             'num_of_paragraphs',
             'num_of_algorithms',
             'num_of_formulas',
             'last_element',
             'sum_space_taken',
             'sum_open_space', 'sum_space_taken_by_figures', 'sum_space_taken_by_tables',
             'sum_of_chars_across_doc',
             'sum_of_words_from_pars', 'sum_of_chars_from_pars', 'avg_num_of_words_from_pars',
             'avg_num_of_chars_from_pars', 'num_of_paragraphs_with_1_word_at_the_end',
             'num_of_figures_with_captions', 'num_of_tables_with_captions']
objects = ['page0', 'column0', 'start_y0', 'end_y0', 'spread_on_more_than_1_column0', 'height0',
           'number_of_lines0', 'num_of_chars0', 'num_of_words0', 'last_line_length_chars0', 'last_line_length_words0',
           'single_word_in_last_line0', 'space_between_this_object_and_last_object0',
           'space_between_this_object_and_the_next_object0', 'page1', 'column1', 'start_y1', 'end_y1',
           'spread_on_more_than_1_column1', 'height1', 'number_of_lines1', 'num_of_chars1', 'num_of_words1',
           'last_line_length_chars1', 'last_line_length_words1', 'single_word_in_last_line1',
           'space_between_this_object_and_last_object1', 'space_between_this_object_and_the_next_object1', 'page2',
           'column2', 'start_y2', 'end_y2', 'spread_on_more_than_1_column2', 'height2', 'number_of_lines2',
           'num_of_chars2', 'num_of_words2', 'last_line_length_chars2', 'last_line_length_words2',
           'single_word_in_last_line2', 'space_between_this_object_and_last_object2',
           'space_between_this_object_and_the_next_object2', 'page3', 'column3', 'start_y3', 'end_y3',
           'spread_on_more_than_1_column3', 'height3', 'number_of_lines3', 'num_of_chars3', 'num_of_words3',
           'last_line_length_chars3', 'last_line_length_words3', 'single_word_in_last_line3',
           'space_between_this_object_and_last_object3', 'space_between_this_object_and_the_next_object3', 'page4',
           'column4', 'start_y4', 'end_y4', 'spread_on_more_than_1_column4', 'height4', 'number_of_lines4',
           'num_of_chars4', 'num_of_words4', 'last_line_length_chars4', 'last_line_length_words4',
           'single_word_in_last_line4', 'space_between_this_object_and_last_object4',
           'space_between_this_object_and_the_next_object4', 'page5', 'column5', 'start_y5', 'end_y5',
           'spread_on_more_than_1_column5', 'height5', 'number_of_lines5', 'num_of_chars5', 'num_of_words5',
           'last_line_length_chars5', 'last_line_length_words5', 'single_word_in_last_line5',
           'space_between_this_object_and_last_object5', 'space_between_this_object_and_the_next_object5', 'page6',
           'column6', 'start_y6', 'end_y6', 'spread_on_more_than_1_column6', 'height6', 'number_of_lines6',
           'num_of_chars6', 'num_of_words6', 'last_line_length_chars6', 'last_line_length_words6',
           'single_word_in_last_line6', 'space_between_this_object_and_last_object6',
           'space_between_this_object_and_the_next_object6', 'page7', 'column7', 'start_y7', 'end_y7',
           'spread_on_more_than_1_column7', 'height7', 'number_of_lines7', 'num_of_chars7', 'num_of_words7',
           'last_line_length_chars7', 'last_line_length_words7', 'single_word_in_last_line7',
           'space_between_this_object_and_last_object7', 'space_between_this_object_and_the_next_object7',
           'page8', 'column8', 'start_y8', 'end_y8', 'spread_on_more_than_1_column8', 'height8',
           'number_of_lines8', 'num_of_chars8', 'num_of_words8', 'last_line_length_chars8',
           'last_line_length_words8', 'single_word_in_last_line8', 'space_between_this_object_and_last_object8',
           'space_between_this_object_and_the_next_object8', 'page9', 'column9', 'start_y9', 'end_y9',
           'spread_on_more_than_1_column9', 'height9', 'number_of_lines9', 'num_of_chars9', 'num_of_words9',
           'last_line_length_chars9', 'last_line_length_words9', 'single_word_in_last_line9',
           'space_between_this_object_and_last_object9', 'space_between_this_object_and_the_next_object9',
           'page10', 'column10', 'start_y10', 'end_y10', 'height10', 'figure_has_caption10', 'figure_position_param10',
           'space_between_this_object_and_last_object10', 'space_between_caption_and_figure10',
           'space_between_this_object_and_the_next_object10', 'page11', 'column11', 'start_y11', 'end_y11', 'height11',
           'figure_has_caption11', 'figure_position_param11', 'space_between_this_object_and_last_object11',
           'space_between_caption_and_figure11', 'space_between_this_object_and_the_next_object11', 'page12',
           'column12', 'start_y12', 'end_y12', 'height12', 'figure_has_caption12', 'figure_position_param12',
           'space_between_this_object_and_last_object12', 'space_between_caption_and_figure12',
           'space_between_this_object_and_the_next_object12', 'page13', 'column13', 'start_y13', 'end_y13', 'height13',
           'figure_has_caption13', 'figure_position_param13', 'space_between_this_object_and_last_object13',
           'space_between_caption_and_figure13', 'space_between_this_object_and_the_next_object13', 'page14',
           'column14', 'start_y14', 'end_y14', 'height14', 'number_of_lines14', 'last_line_length_chars14',
           'last_line_length_words14', 'space_between_this_object_and_last_object14',
           'space_between_this_object_and_the_next_object14', 'page15', 'column15', 'start_y15', 'end_y15', 'height15',
           'number_of_lines15', 'last_line_length_chars15', 'last_line_length_words15',
           'space_between_this_object_and_last_object15', 'space_between_this_object_and_the_next_object15', 'page16',
           'column16', 'start_y16', 'end_y16', 'height16', 'number_of_lines16', 'last_line_length_chars16',
           'last_line_length_words16', 'space_between_this_object_and_last_object16',
           'space_between_this_object_and_the_next_object16', 'page17', 'column17', 'start_y17', 'end_y17', 'height17',
           'number_of_lines17', 'last_line_length_chars17', 'last_line_length_words17',
           'space_between_this_object_and_last_object17', 'space_between_this_object_and_the_next_object17', 'page18',
           'column18', 'start_y18', 'end_y18', 'height18', 'table_has_caption18', 'table_position_param18',
           'space_between_this_object_and_last_object18', 'space_between_caption_and_table18',
           'space_between_this_object_and_the_next_object18', 'page19', 'column19', 'start_y19', 'end_y19', 'height19',
           'table_has_caption19', 'table_position_param19', 'space_between_this_object_and_last_object19',
           'space_between_caption_and_table19', 'space_between_this_object_and_the_next_object19', 'page20', 'column20',
           'start_y20', 'end_y20', 'height20', 'number_of_lines20', 'last_line_length_chars20',
           'last_line_length_words20', 'space_between_this_object_and_last_object20',
           'space_between_this_object_and_the_next_object20', 'page21', 'column21', 'start_y21', 'end_y21', 'height21',
           'number_of_lines21', 'last_line_length_chars21', 'last_line_length_words21',
           'space_between_this_object_and_last_object21', 'space_between_this_object_and_the_next_object21', 'page22',
           'column22', 'start_y22', 'end_y22', 'spread_on_more_than_1_column22', 'height22',
           'space_between_this_object_and_last_object22', 'space_between_this_object_and_the_next_object22', 'page23',
           'column23', 'start_y23', 'end_y23', 'spread_on_more_than_1_column23', 'height23',
           'space_between_this_object_and_last_object23', 'space_between_this_object_and_the_next_object23', 'page24',
           'column24', 'start_y24', 'end_y24', 'spread_on_more_than_1_column24', 'height24',
           'space_between_this_object_and_last_object24', 'space_between_this_object_and_the_next_object24', 'page25',
           'column25', 'start_y25', 'end_y25', 'spread_on_more_than_1_column25', 'height25',
           'space_between_this_object_and_last_object25', 'space_between_this_object_and_the_next_object25', 'page26',
           'column26', 'start_y26', 'end_y26', 'spread_on_more_than_1_column26', 'height26',
           'space_between_this_object_and_last_object26', 'space_between_this_object_and_the_next_object26', 'page27',
           'column27', 'start_y27', 'end_y27', 'spread_on_more_than_1_column27', 'height27',
           'space_between_this_object_and_last_object27', 'space_between_this_object_and_the_next_object27', 'page28',
           'column28', 'start_y28', 'end_y28', 'spread_on_more_than_1_column28', 'height28', 'number_of_lines28',
           'num_of_chars28', 'num_of_words28', 'last_line_length_chars28', 'last_line_length_words28',
           'single_word_in_last_line28', 'space_between_this_object_and_last_object28',
           'space_between_this_object_and_the_next_object28', 'page29', 'column29', 'start_y29', 'end_y29',
           'spread_on_more_than_1_column29', 'height29', 'number_of_lines29', 'num_of_chars29', 'num_of_words29',
           'last_line_length_chars29', 'last_line_length_words29', 'single_word_in_last_line29',
           'space_between_this_object_and_last_object29', 'space_between_this_object_and_the_next_object29', 'page30',
           'column30', 'start_y30', 'end_y30', 'spread_on_more_than_1_column30', 'height30', 'number_of_lines30',
           'num_of_chars30', 'num_of_words30', 'last_line_length_chars30', 'last_line_length_words30',
           'single_word_in_last_line30', 'space_between_this_object_and_last_object30',
           'space_between_this_object_and_the_next_object30', 'page31', 'column31', 'start_y31', 'end_y31',
           'spread_on_more_than_1_column31', 'height31', 'number_of_lines31', 'num_of_chars31', 'num_of_words31',
           'last_line_length_chars31', 'last_line_length_words31', 'single_word_in_last_line31',
           'space_between_this_object_and_last_object31', 'space_between_this_object_and_the_next_object31', 'page32',
           'column32', 'start_y32', 'end_y32', 'spread_on_more_than_1_column32', 'height32', 'number_of_lines32',
           'num_of_chars32', 'num_of_words32', 'last_line_length_chars32', 'last_line_length_words32',
           'single_word_in_last_line32', 'space_between_this_object_and_last_object32',
           'space_between_this_object_and_the_next_object32', 'page33', 'column33', 'start_y33', 'end_y33', 'height33',
           'bad_detection_formula33', 'space_between_this_object_and_last_object33',
           'space_between_this_object_and_the_next_object33', 'page34', 'column34', 'start_y34', 'end_y34', 'height34',
           'bad_detection_formula34', 'space_between_this_object_and_last_object34',
           'space_between_this_object_and_the_next_object34', 'page35', 'column35', 'start_y35', 'end_y35', 'height35',
           'bad_detection_formula35', 'space_between_this_object_and_last_object35',
           'space_between_this_object_and_the_next_object35', 'page36', 'column36', 'start_y36', 'end_y36', 'height36',
           'bad_detection_formula36', 'space_between_this_object_and_last_object36',
           'space_between_this_object_and_the_next_object36', 'page37', 'column37', 'start_y37', 'end_y37', 'height37',
           'bad_detection_formula37', 'space_between_this_object_and_last_object37',
           'space_between_this_object_and_the_next_object37', 'page38', 'column38', 'start_y38', 'end_y38',
           'spread_on_more_than_1_column38', 'height38', 'algorithm_position_param38',
           'space_between_this_object_and_last_object38', 'space_between_this_object_and_the_next_object38', 'page39',
           'column39', 'start_y39', 'end_y39', 'spread_on_more_than_1_column39', 'height39',
           'algorithm_position_param39', 'space_between_this_object_and_last_object39',
           'space_between_this_object_and_the_next_object39']

batch_operator = int(sys.argv[1])
on_all = int(sys.argv[2])
model_number = int(sys.argv[3])
path_for_read = sys.argv[4]
path_for_write = sys.argv[5]
type_of_dataset = int(sys.argv[6])

os.system(f"echo 1")
columns = []  # automatically decides on the feature_Fam
if on_all == 0:  # all
    type_of_family = "all"
    columns = cols + total_pdf + objects
elif on_all == 1:  # objects
    type_of_family = "objects"
    columns = objects
elif on_all == 2:  # columns
    type_of_family = "columns"
    columns = cols
elif on_all == 3:  # doc
    type_of_family = "doc"
    columns = total_pdf
elif on_all == 4:  # objects + cols
    type_of_family = "objects + cols"
    columns = objects + cols
elif on_all == 5:  # objects + doc
    type_of_family = "objects + doc"
    columns = objects + total_pdf
elif on_all == 6:  # columns + doc
    type_of_family = "columns + doc"
    columns = cols + total_pdf

if(type_of_dataset == 0):
    x_name = ['Unnamed: 0.1']
elif(type_of_dataset == 1):
  if (batch_operator==1 or batch_operator==2):
    x_name = ['Unnamed: 0.1']
  else:
    x_name = ['Unnamed: 0.1.1']
elif(type_of_dataset == 2):
    x_name = ['Unnamed: 0.1.1'] # for final


columns += x_name

os.system(f"echo 2")
# if binary==0:
#     y_cols=["binary_class"]
# else:
#     y_cols=["y_gained"]
# y = pd.read_csv(path, usecols=y_cols)
# ydf_cols = ['herustica']
# Y_df= pd.read_csv(path,usecols=ydf_cols)
os.system(f"echo 3")

operator_dict = {2: [0.9, 0.8, 0.7, 0.6, 0.5], 3: [1], 4: [1], 5: [1], 6: [1], 7: [0.9, 0.8, 0.7, 0.6], 8: [1]}
operator_max = {2: 4, 3: 2, 4: 1, 5: 2, 6: 11, 7: 2, 8: 4}


features_family_dct = {0: "all", 1: "objects", 2: "cols", 3: "paper", 4: "obj & cols", 5: "obj & paper",
                       6: "cols & paper"}

if type_of_dataset == 0:
    columns += ['type', 'value', 'object_used_on', 'num_of_object']
    path = path_for_read + f"/concatted_{batch_operator}.csv"
    os.system(f"echo {path}")
    path_to_save = path_for_write + f"/out_concatted_{on_all}_{model_number}_{type_of_dataset}.csv"
elif type_of_dataset == 1:  # 2nd framework
    # path = path_for_read + f"/concatted_{batch_operator}.csv"
    columns += ['value', 'object_used_on', 'num_of_object']
    path = path_for_read + f"/concatted_{batch_operator}.csv"
    os.system(f"echo {path}")
    path_to_save = path_for_write + f"/out_concatted_{batch_operator}_{on_all}_{model_number}_{type_of_dataset}.csv"
elif type_of_dataset == 2:  # 3rd framework
    # path = path_for_read + f"/operator_{batch_operator}_value_{val}_onNumber_{i}.csv"
    # path = path_for_read + f"\\df_3.csv" #for now
    columns += ['object_used_on']
    for val in operator_dict[batch_operator]:
        tot = operator_max[batch_operator]
        for i in range(1, tot + 1):
            print("WE ARE IN " + str(batch_operator) + " WITH VALUE " + str(val) + " ON NUMBER " + str(i))
            path = path_for_read + f'/operator_{batch_operator}_value_{val}_onNumber_{i}.csv'
            os.system(f"echo {path}")
            path_to_save = path_for_write + f'/out_operator_{batch_operator}_{on_all}_{model_number}_{type_of_dataset}_value_{val}_onNumber_{i}.csv'
            operator_dict_new = {0: "all_operators", 1: "vspace", 2: "change_figure_size", 3: "change_algorithm_size",
                             4: "convert_enum",
                             5: "remove_par_tag", 6: "combine_two_paragraphs", 7: "change_table_size",
                             8: "remove_special_positional_chars", 9: "remove_last_2_words"}

            df = pd.read_csv(path,
                             usecols=columns)  # dataframe from the right csv, we will filter by the feature_family (columns)
            # creating the csv:
            #os.system(f"echo {df.head(3)}")
            print(df.columns)
            print("8a")
            columns_for_csv = ["doc_name", "operator", "feature_family", "model", "class", "value_of_operator",
                               "object_used_on", "num_of_object",
                               "came_from_structure", "came_from_fold", "prediction", "truth", "heuristica"]

            Rows = [x for x in range(len(df.index))]  # maybe add more but idk
            results_df = pd.DataFrame(index=Rows, columns=columns_for_csv)
            # y_cols = ['y_gained', 'lines_we_gained', 'binary_class', 'herustica']
            # Y_df = pd.read_csv(path, usecols=y_cols)

            if model_number <= 3:
                y_cols = ["binary_class"]
            else:
                y_cols = ["y_gained"]
            y = pd.read_csv(path, usecols=y_cols)

            ydf_cols = ['herustica','binary_class',"y_gained",x_name[0]]
            Y_df = pd.read_csv(path, usecols=ydf_cols)
            #os.system(f"echo {Y_df.head(3)}")
            print("before the split")
            print("loaded df:")
            #print(df)
            print("loaded y_df:")
            #print(Y_df)

            type_of_model = ''
            if (model_number > 3):  # regression
                type_of_model = 'regression'
                reg_target = 'y_gained'
                # reg_columns_to_drop = ['lines_we_gained', 'binary_class', 'herustica']
                scoring = ('explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error',
                           'neg_mean_squared_log_error',
                           'r2')
                # rows_to_drop = Y_df.index[Y_df[reg_target] == -1].tolist()
                # print(rows_to_drop)
                # # df = df[df[target] >= 0]  # remove rows with -1 in target
                # Y_df = Y_df.drop(rows_to_drop)
                # df = df.drop(rows_to_drop)
                # Y_df = Y_df.reset_index(drop=True)
                # df = df.reset_index(drop=True)

                folds, groups = clean_data(target=reg_target, model_type="regression")

                # y = Y_df.loc[:, ['y_gained']].values.reshape(-1)
            else:
                type_of_model = 'binary'
                target = 'binary_class'
                # columns_to_drop = ['y_gained', 'lines_we_gained', 'herustica']
                scoring = ('accuracy', 'precision', 'recall', 'f1', 'jaccard', 'roc_auc', 'average_precision')

                # rows_to_drop = Y_df.index[Y_df[target] == -1].tolist()
                # print(rows_to_drop)
                # df = df[df[target] >= 0]  # remove rows with -1 in target
                # Y_df = Y_df.drop(rows_to_drop)
                # df = df.drop(rows_to_drop)
                # Y_df = Y_df.reset_index(drop=True)
                # df = df.reset_index(drop=True)

                folds, groups = clean_data(target=target, model_type="classification")
                # y = Y_df.loc[:, ['binary_class']].values.reshape(-1)

            print("after the split")
            print("loaded X:")
            # print(X)
            print("loaded y:")
            # print(y)

            print("df:")
            # print(df)
            print("Y_df:")
            # print(Y_df)
            # we got the X and y, and their corresponding split, now we create the right model:

            models = [(AdaBoostClassifier(random_state=0, n_estimators=150), {"n_estimators": [50, 100, 150]}),
                      (XGBClassifier(random_state=0), {"eta": [0.1, 0.2, 0.3]}),
                      (RandomForestClassifier(random_state=0), {"criterion": ['entropy']}),
                      (ExtraTreesClassifier(random_state=0), {"n_estimators": [100, 150, 200]}),
                      (AdaBoostRegressor(random_state=0), {"n_estimators": [50, 100, 150]}),
                      (RandomForestRegressor(random_state=0), {"criterion": ['mse']}),
                      (ExtraTreesRegressor(random_state=0), {"n_estimators": [100, 150, 200]})]

            # if (len(X) < 40): maybe use
            #     #writing blank csv

            name_for_index = {}
            index_for_name = {}
            name = ""
            last_name = ""
            for i in range(len(groups)):
                name = groups[i]
                if (last_name != name):  # found new name
                    last_name = name
                    index_for_name[i] = []
                    index_for_name[i].append(last_name)
                    name_for_index[last_name] = []
                    name_for_index[last_name].append(i)
                else:
                    name_for_index[last_name].append(i)
                    index_for_name[i] = []
                    index_for_name[i].append(last_name)

            big_dataset_splits = read_list(f'list_of_folds_new')
            all_models_from_all_folds = []
            results_from_all_models_from_all_folds = []
            models_from_each_fold, results_from_all_folds_binary = simpleModel(
                                                                               estimator=models[model_number][0],
                                                                               scoring=scoring,
                                                                               folds=big_dataset_splits,
                                                                               groups=groups)
            all_models_from_all_folds.append(models_from_each_fold)
            results_from_all_models_from_all_folds.append(
                (type(models[model_number][0]).__name__, models_from_each_fold[0], results_from_all_folds_binary))

            print("all models from all folds array:")
            # print(all_models_from_all_folds)
            print("results from all model from all folds array:")
            # print(results_from_all_models_from_all_folds)

            test = []
            train = []
            for i, (train_index, test_index) in enumerate(cv_generator(big_dataset_splits, groups)):
                # print(f"Fold {i}:")
                # print(f"  Train: index={train_index}")
                # print(f"  Test: index={test_index}")
                # print(len(train_index))
                # print(len(test_index))
                test.append(test_index)
                train.append(train_index)

            # getting the information from the learning process and testing process:
            name = ""
            last_name = ""

            dict_for_results = {}
            dict_for_best_results = {}
            dict_for_min_results = {}

            dict_for_regression_results = {}
            dict_for_best_regression_results = {}
            dict_for_min_regression_results = {}

            list_for_results = []
            list_for_best_results = []
            list_for_min_results = []

            list_for_results_regression = []
            list_for_best_results_regression = []
            list_for_min_results_regression = []
            dict_for_name_X_index_prediction_y_true_heuristica = {}
            fold_index = 0
            for fold in test:
                start_index = -1
                end_index = -1
                for index in range(len(fold) + 1):
                    if (index == len(fold)):
                        end_index = fold[index - 1] + 1
                        df_to_predict = df.iloc[start_index].to_frame().T
                        #print(start_index)
                        #print(end_index)
                        #print(last_last_name[0])  # this is the name of the doc that we predict
                        # print(name_for_index[last_last_name[0]])
                        # df_to_predict_for_reg = X[start_index:end_index]
                        # print(last_last_name)
                        # print(all_models_from_all_folds)
                        # print(all_regression_models_from_all_folds)
                        for model in all_models_from_all_folds:  # model is a list of all models in different folds
                            sub_model = model[fold_index]
                            prediction = sub_model.predict(df_to_predict)
                            index_index = start_index
                            small_index = 0
                            if (type_of_model == "binary"):
                                if (Y_df.at[index_index, 'herustica'] > 0):
                                    binary_class = 1
                                else:
                                    binary_class = 0
                                dict_for_name_X_index_prediction_y_true_heuristica[
                                    Y_df.at[index_index, x_name[0]]] = (
                                    prediction[small_index], Y_df.at[index_index, 'binary_class'], binary_class,
                                    val, df.at[index_index, 'object_used_on'],
                                    i, fold_index)
                            else:
                                dict_for_name_X_index_prediction_y_true_heuristica[
                                    Y_df.at[index_index, x_name[0]]] = (
                                    prediction[small_index], Y_df.at[index_index, 'y_gained'],
                                    Y_df.at[index_index, 'herustica'], val,
                                    df.at[index_index, 'object_used_on'], i,
                                    fold_index)
                            small_index += 1
                        break
                    name = index_for_name[fold[index]]
                    if (last_name != name):
                        last_last_name = last_name
                        last_name = name
                        if (end_index == -1 and start_index == -1):
                            start_index = fold[index]
                        else:
                            end_index = fold[index - 1] + 1
                            df_to_predict = df[start_index:end_index]
                            #print(start_index)
                            #print(end_index)
                            #print(last_last_name[0])  # this is the name of the doc that we predict
                            # print(name_for_index[last_last_name[0]])
                            # df_to_predict_for_reg = X[start_index:end_index]
                            # print(last_last_name)
                            # print(all_models_from_all_folds)
                            # print(all_regression_models_from_all_folds)
                            for model in all_models_from_all_folds:  # model is a list of all models in different folds
                                sub_model = model[fold_index]
                                prediction = sub_model.predict(df_to_predict)
                                for index_index in range(start_index, end_index + 1):
                                    small_index = 0
                                    if (type_of_model == "binary"):
                                        if (Y_df.at[index_index, 'herustica'] > 0):
                                            binary_class_for_heurstica = 1
                                        else:
                                            binary_class_for_heurstica = 0
                                        dict_for_name_X_index_prediction_y_true_heuristica[
                                            Y_df.at[index_index, x_name[0]]] = (
                                        prediction[small_index], Y_df.at[index_index, 'binary_class'],
                                        binary_class_for_heurstica, val, df.at[index_index, 'object_used_on'], i,
                                        fold_index)
                                    else:
                                        dict_for_name_X_index_prediction_y_true_heuristica[
                                            Y_df.at[index_index, x_name[0]]] = (
                                        prediction[small_index], Y_df.at[index_index, 'y_gained'],
                                        Y_df.at[index_index, 'herustica'], val, df.at[index_index, 'object_used_on'], i,
                                        fold_index)
                                    small_index += 1
                            start_index = fold[index]
                fold_index += 1

            #print(dict_for_name_X_index_prediction_y_true_heuristica)

            # create the csv:
            test_index_for_csv = 0
            for key, value in dict_for_name_X_index_prediction_y_true_heuristica.items():
                results_df.at[Rows[test_index_for_csv], "doc_name"] = key
                results_df.at[Rows[test_index_for_csv], "operator"] = operator_dict_new[batch_operator]
                results_df.at[Rows[test_index_for_csv], "feature_family"] = type_of_family
                results_df.at[Rows[test_index_for_csv], "model"] = type(models[model_number][0]).__name__
                results_df.at[Rows[test_index_for_csv], "class"] = type_of_model
                # check what we do, do we include value of operators even when we dont check for specific operators
                results_df.at[Rows[test_index_for_csv], "value_of_operator"] = value[3]
                results_df.at[Rows[test_index_for_csv], "object_used_on"] = value[4]
                results_df.at[Rows[test_index_for_csv], "num_of_object"] = value[5]
                results_df.at[Rows[test_index_for_csv], "came_from_structure"] = type_of_dataset
                results_df.at[Rows[test_index_for_csv], "came_from_fold"] = value[6]
                results_df.at[Rows[test_index_for_csv], "prediction"] = value[0]
                results_df.at[Rows[test_index_for_csv], "truth"] = value[1]
                results_df.at[Rows[test_index_for_csv], "heuristica"] = value[2]
                test_index_for_csv += 1
            results_df.to_csv(path_to_save, index=False)

if(type_of_dataset == 0 or type_of_dataset == 1):
    operator_dict_new2 = {0: "all_operators", 1: "vspace", 2: "change_figure_size", 3: "change_algorithm_size",
                        4: "convert_enum",
                        5: "remove_par_tag", 6: "combine_two_paragraphs", 7: "change_table_size",
                        8: "remove_special_positional_chars", 9: "remove_last_2_words"}

    df = pd.read_csv(path, usecols=columns)  # dataframe from the right csv, we will filter by the feature_family (columns)
    # creating the csv:
    #print(df.columns)
    print("hi")

    columns_for_csv = ["doc_name", "operator", "feature_family", "model", "class", "value_of_operator",
                       "object_used_on", "num_of_object",
                       "came_from_structure","came_from_fold", "prediction", "truth", "heuristica"]

    Rows = [x for x in range(len(df.index))]  # maybe add more but idk
    results_df = pd.DataFrame(index=Rows, columns=columns_for_csv)
    # y_cols = ['y_gained', 'lines_we_gained', 'binary_class', 'herustica']
    # Y_df = pd.read_csv(path, usecols=y_cols)

    if model_number <= 3:
        y_cols = ["binary_class"]
    else:
        y_cols = ["y_gained"]
    y = pd.read_csv(path, usecols=y_cols)

    ydf_cols = ['herustica', 'binary_class', "y_gained",x_name[0]]
    Y_df = pd.read_csv(path, usecols=ydf_cols)

    print("before the split")
    print("loaded df:")
    #print(df)
    print("loaded y_df:")
    #print(Y_df)

    type_of_model = ''
    if (model_number > 3):  # regression
        type_of_model = 'regression'
        reg_target = 'y_gained'
        # reg_columns_to_drop = ['lines_we_gained', 'binary_class', 'herustica']
        scoring = ('explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error',
                   'neg_mean_squared_log_error',
                   'r2')
        # rows_to_drop = Y_df.index[Y_df[reg_target] == -1].tolist()
        # print(rows_to_drop)
        # # df = df[df[target] >= 0]  # remove rows with -1 in target
        # Y_df = Y_df.drop(rows_to_drop)
        # df = df.drop(rows_to_drop)
        # Y_df = Y_df.reset_index(drop=True)
        # df = df.reset_index(drop=True)

        folds, groups = clean_data(target=reg_target, model_type="regression")

        # y = Y_df.loc[:, ['y_gained']].values.reshape(-1)
    else:
        type_of_model = 'binary'
        target = 'binary_class'
        # columns_to_drop = ['y_gained', 'lines_we_gained', 'herustica']
        scoring = ('accuracy', 'precision', 'recall', 'f1', 'jaccard', 'roc_auc', 'average_precision')

        # rows_to_drop = Y_df.index[Y_df[target] == -1].tolist()
        # print(rows_to_drop)
        # df = df[df[target] >= 0]  # remove rows with -1 in target
        # Y_df = Y_df.drop(rows_to_drop)
        # df = df.drop(rows_to_drop)
        # Y_df = Y_df.reset_index(drop=True)
        # df = df.reset_index(drop=True)

        folds, groups = clean_data(target=target, model_type="classification")
        # y = Y_df.loc[:, ['binary_class']].values.reshape(-1)

    print("after the split")
    print("loaded X:")
    #print(X)
    print("loaded y:")
    #print(y)

    print("df:")
    #print(df)
    print("Y_df:")
    #print(Y_df)
    # we got the X and y, and their corresponding split, now we create the right model:

    models = [(AdaBoostClassifier(random_state=0, n_estimators=150), {"n_estimators": [50, 100, 150]}),
             (XGBClassifier(random_state=0), {"eta": [0.1, 0.2, 0.3]}),
             (RandomForestClassifier(random_state=0), {"criterion": ['entropy']}),
             (ExtraTreesClassifier(random_state=0), {"n_estimators": [100, 150, 200]}),
             (AdaBoostRegressor(random_state=0), {"n_estimators": [50, 100, 150]}),
             (RandomForestRegressor(random_state=0), {"criterion": ['mse']}),
             (ExtraTreesRegressor(random_state=0), {"n_estimators": [100, 150, 200]})]

    # if (len(X) < 40): maybe use
    #     #writing blank csv

    name_for_index = {}
    index_for_name = {}
    name = ""
    last_name = ""
    for i in range(len(groups)):
       name = groups[i]
       if (last_name != name):  # found new name
           last_name = name
           index_for_name[i] = []
           index_for_name[i].append(last_name)
           name_for_index[last_name] = []
           name_for_index[last_name].append(i)
       else:
           name_for_index[last_name].append(i)
           index_for_name[i] = []
           index_for_name[i].append(last_name)

    big_dataset_splits = read_list(f'list_of_folds_new')
    all_models_from_all_folds = []
    results_from_all_models_from_all_folds = []
    models_from_each_fold, results_from_all_folds_binary = simpleModel(
        estimator=models[model_number][0],
        scoring=scoring,
        folds=big_dataset_splits,
        groups=groups)

    all_models_from_all_folds.append(models_from_each_fold)
    results_from_all_models_from_all_folds.append(
       (type(models[model_number][0]).__name__, models_from_each_fold[0], results_from_all_folds_binary))

    print("all models from all folds array:")
    #print(all_models_from_all_folds)
    print("results from all model from all folds array:")
    #print(results_from_all_models_from_all_folds)

    test = []
    train = []
    for i, (train_index, test_index) in enumerate(cv_generator(big_dataset_splits, groups)):
       #print(f"Fold {i}:")
       #print(f"  Train: index={train_index}")
       #print(f"  Test: index={test_index}")
       #print(len(train_index))
       #print(len(test_index))
       test.append(test_index)
       train.append(train_index)

    # getting the information from the learning process and testing process:
    name = ""
    last_name = ""

    dict_for_results = {}
    dict_for_best_results = {}
    dict_for_min_results = {}

    dict_for_regression_results = {}
    dict_for_best_regression_results = {}
    dict_for_min_regression_results = {}

    list_for_results = []
    list_for_best_results = []
    list_for_min_results = []

    list_for_results_regression = []
    list_for_best_results_regression = []
    list_for_min_results_regression = []
    dict_for_name_X_index_prediction_y_true_heuristica = {}
    fold_index = 0
    for fold in test:
       start_index = -1
       end_index = -1
       for index in range(len(fold)+1):
           if(index == len(fold)):
                end_index = fold[index-1]+1
                df_to_predict = df.iloc[start_index].to_frame().T
                #print(start_index)
                #print(end_index)
                #print(last_last_name[0])  # this is the name of the doc that we predict
                # print(name_for_index[last_last_name[0]])
                # df_to_predict_for_reg = X[start_index:end_index]
                # print(last_last_name)
                # print(all_models_from_all_folds)
                # print(all_regression_models_from_all_folds)
                for model in all_models_from_all_folds:  # model is a list of all models in different folds
                    sub_model = model[fold_index]
                    prediction = sub_model.predict(df_to_predict)
                    index_index = start_index
                    small_index = 0
                    if (type_of_model == "binary"):
                        if (Y_df.at[index_index, 'herustica'] > 0):
                            binary_class = 1
                        else:
                            binary_class = 0
                        dict_for_name_X_index_prediction_y_true_heuristica[
                            Y_df.at[index_index, x_name[0]]] = (
                        prediction[small_index], Y_df.at[index_index, 'binary_class'], binary_class,
                        df.at[index_index, 'value'], df.at[index_index, 'object_used_on'],
                        df.at[index_index, 'num_of_object'],fold_index)
                    else:
                        dict_for_name_X_index_prediction_y_true_heuristica[
                            Y_df.at[index_index, x_name[0]]] = (
                        prediction[small_index], Y_df.at[index_index, 'y_gained'],
                        Y_df.at[index_index, 'herustica'], df.at[index_index, 'value'],
                        df.at[index_index, 'object_used_on'], df.at[index_index, 'num_of_object'],fold_index)
                    small_index += 1
                break
           name = index_for_name[fold[index]]
           if (last_name != name):
               last_last_name = last_name
               last_name = name
               if (end_index == -1 and start_index == -1):
                   start_index = fold[index]
               else:
                   end_index = fold[index - 1] + 1
                   df_to_predict = df[start_index:end_index]
                   #print(start_index)
                   #print(end_index)
                   #print(last_last_name[0])  # this is the name of the doc that we predict
                   # print(name_for_index[last_last_name[0]])
                   # df_to_predict_for_reg = X[start_index:end_index]
                   # print(last_last_name)
                   # print(all_models_from_all_folds)
                   # print(all_regression_models_from_all_folds)
                   for model in all_models_from_all_folds:  # model is a list of all models in different folds
                       sub_model = model[fold_index]
                       prediction = sub_model.predict(df_to_predict)
                       for index_index in range(start_index, end_index + 1):
                           small_index = 0
                           if (type_of_model == "binary"):
                               if (Y_df.at[index_index, 'herustica'] > 0):
                                   binary_class = 1
                               else:
                                   binary_class = 0
                               dict_for_name_X_index_prediction_y_true_heuristica[Y_df.at[index_index, x_name[0]]] = (prediction[small_index], Y_df.at[index_index, 'binary_class'], binary_class,df.at[index_index,'value'],df.at[index_index,'object_used_on'],df.at[index_index,'num_of_object'],fold_index)
                           else:
                               dict_for_name_X_index_prediction_y_true_heuristica[Y_df.at[index_index, x_name[0]]] = (prediction[small_index], Y_df.at[index_index, 'y_gained'],Y_df.at[index_index, 'herustica'],df.at[index_index,'value'],df.at[index_index,'object_used_on'],df.at[index_index,'num_of_object'],fold_index)
                           small_index += 1
                   start_index = fold[index]
       fold_index += 1

    #print(dict_for_name_X_index_prediction_y_true_heuristica)
    #print(len(dict_for_name_X_index_prediction_y_true_heuristica))

    # create the csv:
    test_index_for_csv = 0
    for key, value in dict_for_name_X_index_prediction_y_true_heuristica.items():
       results_df.at[Rows[test_index_for_csv], "doc_name"] = key
       results_df.at[Rows[test_index_for_csv], "operator"] = operator_dict_new2[batch_operator]
       results_df.at[Rows[test_index_for_csv], "feature_family"] = type_of_family
       results_df.at[Rows[test_index_for_csv], "model"] = type(models[model_number][0]).__name__
       results_df.at[Rows[test_index_for_csv], "class"] = type_of_model
       # check what we do, do we include value of operators even when we dont check for specific operators
       results_df.at[Rows[test_index_for_csv], "value_of_operator"] = value[3]
       results_df.at[Rows[test_index_for_csv], "object_used_on"] = value[4]
       results_df.at[Rows[test_index_for_csv], "num_of_object"] = value[5]
       results_df.at[Rows[test_index_for_csv], "came_from_structure"] = type_of_dataset
       results_df.at[Rows[test_index_for_csv], "came_from_fold"] = value[6]
       results_df.at[Rows[test_index_for_csv], "prediction"] = value[0]
       results_df.at[Rows[test_index_for_csv], "truth"] = value[1]
       results_df.at[Rows[test_index_for_csv], "heuristica"] = value[2]
       test_index_for_csv += 1
    results_df.to_csv(path_to_save, index=False)
        # print(dict_for_results)