from pdfMiner_new_checking_length import run
from pdf_extraction import pictures,tables
import sys
import numpy as np

def order_page(path,page_num,tables_dict,pictures_dict,frompdf):

    pdf_p0=frompdf[page_num]
    tables_p0=[]
    try:
        tables_p0=tables_dict[page_num]
    except:
        tables_p0=[]
    pictures_p0=[]
    try:
        pictures_p0 = pictures_dict[page_num]
    except:
        pictures_p0=[]

    left_column= {}
    right_column = {}
    for line in pdf_p0:
        if line[1][0]<300: #left_column
            left_column[(line[1][1],line[1][3])]=line[0]
        else:
            right_column[(line[1][1],line[1][3])] = line[0]

    for key_picture in pictures_p0:

        if key_picture["x0"]<300:
            if (key_picture["y0"],key_picture["y1"]) in left_column: #it is possible that two figures have the exact same height
                left_column[(key_picture["y0"],key_picture["y1"])]+="_FIGUREFIGURE"
            else:
                left_column[(key_picture["y0"],key_picture["y1"])]="FIGUREFIGURE"
        else:
            if (key_picture["y0"],key_picture["y1"]) in right_column: #it is possible that two figures have the exact same height
                right_column[(key_picture["y0"],key_picture["y1"])]+="_FIGUREFIGURE"
            else:
                right_column[(key_picture["y0"],key_picture["y1"])]="FIGUREFIGURE"

    for key_table in tables_p0:
        if len(tables_p0[key_table])!=0:
            for table in tables_p0[key_table]:
                if table[0]<300:
                    left_column[(table[1],table[3])] = "TABLETABLE"
                else:
                    right_column[(table[1],table[3])] = "TABLETABLE"

    left_column_sorted_dict=sorted(left_column.items(),key=lambda x:x[0][0])
    right_column_sorted_dict=sorted(right_column.items(),key=lambda x:x[0][0])
    return left_column_sorted_dict,right_column_sorted_dict

def order(path):
    pdf_total_order={}
    frompdf = run(path)

    tables_dict = tables(path)
    pictures_dict = pictures(path)


    for i in range(len(frompdf)-1):
        pdf_total_order[i]=order_page(path,i,tables_dict,pictures_dict,frompdf)


    last_obj=0
    first_obj=0
    if len(pdf_total_order)==1:
        first_obj = pdf_total_order[0][0][0][0][0]
        last_obj = pdf_total_order[0][0][-1][0][1]

        if len(pdf_total_order[0][1]) != 0:
            first_obj = pdf_total_order[0][1][0][0][0]
            last_obj = pdf_total_order[0][1][-1][0][1]
            return 792 - last_obj
        else:
            return 792+ 792-last_obj
    else:
        if len(pdf_total_order[1][0]) > 0 and len(pdf_total_order[1][1]) == 0:
            return 0
    if len(pdf_total_order[1][0]) == 0:
        first_obj = pdf_total_order[0][0][0][0][0]
        last_obj = pdf_total_order[0][0][-1][0][1]

        if len(pdf_total_order[0][1]) != 0:
            first_obj = pdf_total_order[0][1][0][0][0]
            last_obj = pdf_total_order[0][1][-1][0][1]
            return 792 - last_obj
        else:
            return 792 + 792 - last_obj
    else:
        first_obj = pdf_total_order[1][1][0][0][0]
        last_obj=pdf_total_order[1][1][-1][0][1]
        return -(last_obj-first_obj)


if __name__ == '__main__':

    # pdf_path=sys.argv[1]
    # index = sys.argv[2]
    # permutation_num = sys.argv[3]
    # created_pdf_path=pdf_path+"/"+index+f"_{permutation_num}.pdf"
    created_pdf_path=""
    total_pdf_length=order(created_pdf_path)
    print(total_pdf_length)
