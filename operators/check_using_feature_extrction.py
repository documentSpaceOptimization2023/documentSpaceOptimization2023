import copy
import pickle
from using_operators import omit,is_par

import main_parsing

latex_path=""
with open(latex_path, encoding='UTF-8') as file:
    # doc = file.read()
    latex_clean_lines = []
    with open(latex_path, encoding='UTF-8') as file:
        # doc = file.read()
        original_lines=[]
        latex_clean_lines=[]
        foundHeader = False
        foundBottom = False
        for line in file:
            latex_clean_lines.append(line)
            if foundHeader == False:
                if line.startswith("\\begin{document}"):
                    foundHeader = True
                original_lines.append("\n")
            else:
                if foundBottom == False and line.startswith("\\end{document}"):
                    foundBottom = True
                else:
                    if foundBottom == False:
                        original_lines.append(line)


list_of_starts, tags = main_parsing.parse2(latex_path, original_lines)


index_for_object = {'Par': 1, 'Figure': 2, 'CaptionFigure': 3, 'Table': 4, 'CaptionTable': 5, 'Section': 6,
                    'SubSection': 7, 'Matrix': 8, 'Enum': 9, 'Formula': 10, 'Algorithm': 11}
# mapping:
mapping_dict = {}
par_index = 0
caption_index = 0
matrix_index = 0
title_index = 0
abstractsec_index = 0
abstractpar_index = 0
section_index = 0
subsection_index = 0
subsubsection_index = 0
enum_index = 0
algo_index = 0
undetected_index = 0
formula_index = 0
caption_table_index = 0
figure_index = 0
table_index = 0
for i in range(len(list_of_starts)):
    if (tags[i][0][0].startswith("Par")):
        par_index += 1
        index = par_index
    elif (tags[i][0][0] == 'Title'):
        title_index += 1
        index = title_index
    elif (tags[i][0][0] == 'AbstractSection'):
        abstractsec_index += 1
        index = abstractsec_index
    elif (tags[i][0][0] == 'AbstractPar'):
        abstractpar_index += 1
        index = abstractpar_index
    elif (tags[i][0][0] == 'Section'):
        section_index += 1
        index = section_index
    elif (tags[i][0][0] == 'SubSection'):
        subsection_index += 1
        index = subsection_index
    elif (tags[i][0][0] == 'SubSubSection'):
        subsubsection_index += 1
        index = subsubsection_index
    elif (tags[i][0][0] == 'Enum'):
        enum_index += 1
        index = enum_index
    elif (tags[i][0][0] == 'Algorithm'):
        algo_index += 1
        index = algo_index
    elif (tags[i][0][0] == 'CaptionFigure'):
        caption_index += 1
        index = caption_index
    elif (tags[i][0][0] == 'CaptionTable'):
        caption_table_index += 1
        index = caption_table_index
    elif (tags[i][0][0] == 'Formula'):
        formula_index += 1
        index = formula_index
    else:
        undetected_index += 1
        index = undetected_index
    if (tags[i][0][0] == 'Figure'):
        mapping_dict[tags[i][0][0] + str(tags[i][0][1])] = (list_of_starts[i], latex_clean_lines[list_of_starts[i]])
    elif (tags[i][0][0] == 'Table'):
        mapping_dict[tags[i][0][0] + str(int(tags[i][0][1]) + 1)] = (
            list_of_starts[i], latex_clean_lines[list_of_starts[i]])
    else:
        mapping_dict[tags[i][0][0] + str(index)] = (list_of_starts[i], latex_clean_lines[list_of_starts[i]])

for k in mapping_dict:
    print(f"{k}: {mapping_dict[k]}")

with open("", 'rb') as dct_file:
    objects = pickle.load(dct_file)

items_seen = []
item_num = 1
occurrence_num=1
enum_list=[]
pdf_order = list(objects.keys())
latex_order = list(mapping_dict.keys())
pdf_pairs = [(pdf_order[i], pdf_order[i + 1]) for i in range(len(pdf_order) - 1)]
latex_pairs = [(latex_order[i], latex_order[i + 1]) for i in range(len(latex_order) - 1)]
pair_to_check = []

for key, value in objects.items():  # in this loop we will make all the operators

    if (key.startswith('Enum')):  # convert enum to paragraph operator
        if (int(key[4:]) not in items_seen):
            chosen_index_to_insert = mapping_dict[key][0]
            new_list = copy.deepcopy(latex_clean_lines)
            number = 1
            for i in range(chosen_index_to_insert, len(new_list)):
                if new_list[i] == '\\end{enumerate}\n':
                    break
                if "\\item" in new_list[i]:
                    new_list[i] = new_list[i].replace("\\item", f"{number}.")
                    items_seen.append(item_num)
                    item_num += 1
                    number += 1
            new_list = omit(new_list, "\\begin{enumerate}\n", occurrence_num)
            new_list = omit(new_list, "\\end{enumerate}\n", occurrence_num)
            occurrence_num += 1
            heuristic = number - 2
            heuristic = heuristic * 3.3
            enum_list.append((new_list, key, heuristic))

# for f in enum_list:
print(enum_list)
for line in enum_list:
    print(line)
# print(enum_list)
