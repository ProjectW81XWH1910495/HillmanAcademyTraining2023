import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

def plot_trend(filename,X_groupname1,plottitle = False,X_groupname2 = False,Y_name='mean_test_auc',method="mean",condition=False,conditionname=False,ifshowdata=False,):
    """
This function is help to plot line chart between two different variable,
if you have two experiments with same x, then you will have 2 different y, you can use the average of y to plot
    Args:
        filename: the name of data file
        X_groupname1: the name of x such like epochs lrate, must same with the name in parameter set
        X_groupname2: if you want to group by two x parameters you can add X_groupname2, for example you want to plot time performance according to different tables and computername
        if you want to plot time performance with different datasets then using "full_dataset_name",if different models then using "ml_classifier_name"
        Y_name:the name of y such like mean test AUC or percent_auc_diff or "running_time1(average sec)"
        method: what kind of method you want use "mean":calculate the average of y with same x;median means median; all:assume you want to plot all the points;
        condition: if you want to select a part of data to plot, for example you want to see all the data with layer number==1, you can make conditionname="layer number" and condition = 1
        ifshowdata: if you want to add data label to your plot
    """
    print(Y_name)
    df = pd.read_csv(filename, sep=',')

    print(conditionname)
    if conditionname:
        df = df[df[conditionname] == condition]
        print("The size of current dataset is "+ str(df.shape))
    if method == "mean":
        if X_groupname2:
            result_table = df.groupby([X_groupname1, X_groupname2]).mean()
        else:
            result_table = df.groupby([X_groupname1]).mean()
        result_table.sort_values(by=X_groupname1, axis=0, ascending=True, inplace=True)
        #print(result_table)
        #print(result_table[Y_name])
        series = result_table[Y_name]
        #print(series)
        y = series.tolist()
        print(y)
        x = list(series.index)
        print(x)
        if isinstance(x[0], float):
            x = [int(j) for j in x]
        else:
            x = [str(j) for j in x]
    elif method == "median":
        result_table = df.groupby([X_groupname1]).median()
        result_table.sort_values(by=X_groupname1, axis=0, ascending=True, inplace=True)
        #print(result_table)
        series = result_table[Y_name]
        #print(series)
        y = series.tolist()
        x = list(series.index)
        if isinstance(x[0], float):
            x = [int(j) for j in x]
        else:
            x = [str(j) for j in x]
    elif method == "all":
        df.sort_values(by=X_groupname1, axis=0, ascending=True, inplace=True)
        x = df[X_groupname1].values.tolist()
        y = df[Y_name].values.tolist()
        print("+++++++++++")
        #print(x)
    plt.plot(x, y)
    #result_table.plot(x="epochs",y="mean_test_auc")
    #print(type(result_table))
    if ifshowdata:
        for a, b in zip(x, y):
            plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=12)
    if X_groupname1 == "layer_number" and not X_groupname2 :
        plt.xticks(ticks=[0,1,2,3],labels=["1 layer", "2 layers", "3 layers", "4 layers"])

    plt.grid(linestyle='-.')
    plt.xlabel(X_groupname1,fontsize=15)
    plt.ylabel(Y_name,fontsize=15)

    if plottitle:
        plt.title(plottitle, fontsize=17)
    elif X_groupname1 == "full_dataset_name":
        print(X_groupname1)
        plt.title("Time performance with different year dataset", fontsize=17)
    elif X_groupname1 == "ml_classifier_name":
        plt.title("Time performance with different models", fontsize=17)
    else:
        plt.title("Relationship between " + Y_name + " and " + X_groupname1, fontsize=17)
    print("Plot trend plot successfully, it will not be saved automatically, please save it by yourself")
    plt.show()

def plot_one_background_setting(data_path, x_label, y_label,title):
    """
    all other hyperparameter values are the same, to see the relationship between x_label and y_label
    """
    df = pd.read_csv(data_path)
    fig=px.scatter(x=df[x_label], y=df[y_label], title=title+"-scatter",
                        labels=dict(x=x_label, y=y_label))
    fig.update_traces(
        textposition='top center',
        textfont={'color': '#bebebe', 'size': 10},
    )
    """put background information(value of other hyperparameters in the title, but the text size is too big, couldn't store too much information"""
    # title_text = ''
    # for column in df.columns:
    #     if column == x_label or column == y_label: continue
    #     title_text += f'{column}: {df[column].unique()}\n'
    title_text = f'{x_label}: {round(df[x_label].min(),5)} -- {round(df[x_label].max(),5)}'

    fig.update_layout(
        height=800,
        title_text=title_text
    )
    fig.show()

def plot_many_background_settings(data_path, x_label, y_label,title,time):
    """
    there are many numbers of background settings, to see the relationship between x_label and y_label based on all groups of values
    """
    df = pd.read_csv(data_path)
    number = 0 #number of groups
    for column in df.columns:
        if column == x_label or column == y_label: continue
        number = len(df[column].unique()) if len(df[column].unique()) > number else number
    print(number)
    #find the facet_col
    facet_col = ''
    for column in df.columns:
        if column == x_label or column == y_label: continue
        if len(df[column].unique()) == number:
            facet_col = column
            break
    # change the facet col to be number
    unique_facet_col = df[facet_col].unique()
    print(len(unique_facet_col),123)
    print(unique_facet_col)
    facet_col_dict = {unique_facet_col[i]: i + 1 for i in range(len(unique_facet_col))}
    new_facet_col = []
    for value in df[facet_col]:
        new_facet_col.append(facet_col_dict[value])

    df[facet_col] = new_facet_col
    fig=px.scatter(df, x=x_label, y=y_label, title=title+"-scatter",
                        labels=dict(x=x_label, y=y_label),facet_col= facet_col,facet_col_wrap=3)
    fig.update_traces(
        textposition='top center',
        textfont={'color': '#bebebe', 'size': 10},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    try:
        pd.to_numeric(df[x_label])
        title_text = f'{x_label}: {round(df[x_label].min(), 5)} -- {round(df[x_label].max(), 5)}'
    except:
        title_text=f'{x_label}: ['
        for value in df[x_label].unique():
            title_text+=value+', '
        title_text += ']'

    fig.update_layout(
        height = 800,
        title_text = title_text
    )
    return fig
    # # 设置输出图像的分辨率（dpi）
    # dpi = 400  # 设置为所需的 DPI 值
    # scale_factor = dpi / 96  # 计算 DPI 缩放因子
    # # 导出图像
    # pio.write_image(fig, f'x_{x_label}_y_{y_label}_trend_plot.png', scale=scale_factor)  # 保存为PNG图像，并设置缩放因子
    # fig.show()
def expand_dictionary_column(data_path, dictionary_column, result_path):
    df = pd.read_csv(data_path)
    dictionary_column = df[dictionary_column]
    for row in dictionary_column:
        row = eval(row)
        keys = row.keys()
        break
    with open(result_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        for row in dictionary_column:
            row = eval(row)
            dict_writer.writerow(row)
def add_new_column_tocsv(data_path, new_column,new_column_name):
    df = pd.read_csv(data_path)
    df[new_column_name] = new_column
    df.to_csv(data_path,index=False)
def return_unique_background_parameters(data_path,return_path,delete_column):
    df = pd.read_csv(data_path)
    for column in delete_column:
        del df[column]
    df=df.drop_duplicates()
    df.to_csv(return_path,index=False)
def retrieve_points_based_on_steps(parameter_path,target_hyperparameter,step):
    df = pd.read_csv(parameter_path)
    # find the facet_col of group(the first column which is not the target hyperparameter
    for column in df.columns:
        if column != target_hyperparameter:
            break
    groups = df.groupby([column])
    result = None
    for group_id, group_df in groups:
        temp = group_df[::step]
        if result is None:
            result = temp
        else:
            result = pd.concat([result, temp], axis=0)
    result.to_csv(parameter_path,index=False)
if __name__ == "__main__":
    flag = int(sys.argv[1])
    if flag == 1: #expand the dictionary column
        data_path = '../DNM/multistage_stage1_SHGS/15year_stage1version2_batch_size_desktop.csv'
        dictionary_column = 'parameters_and_values'
        result_path = '../DNM/multistage_stage1_SHGS/15year_stage1version2_batch_size_desktop_parameters.csv'
        #unique=False #if unique=True, could return the unique values
        expand_dictionary_column(data_path,dictionary_column,result_path)
    elif flag == 2: #Plot the trend
        data_path =  '../DNM/multistage_stage1_SHGS/SHGS_version2/15year_stage1version2_dec_mac_version2_parameters.csv'
        x_label = 'dec'
        y_label = 'test_auc'
        title = 'test'
        plot_one_background_setting(data_path, x_label, y_label,title)
    elif flag == 3:
        data_path = '../DNM/multistage_stage1_SHGS/15year_stage1version2_batch_size_desktop_parameters.csv'
        x_label = 'batch_size'
        y_label = 'mean_test_auc'
        title = 'test'
        time = True
        plot_many_background_settings(data_path, x_label, y_label, title, time)
    elif flag == 4: #add new column to a file
        data_path = '../DNM/multistage_stage1_SHGS/15year_stage1version2_batch_size_desktop_parameters.csv'
        temp_path = '../DNM/multistage_stage1_SHGS/15year_stage1version2_batch_size_desktop.csv'
        df = pd.read_csv(temp_path)
        new_column = df['mean_test_auc']
        new_column_name='mean_test_auc'
        add_new_column_tocsv(data_path,new_column,new_column_name)
    elif flag == 5: #Only return the unique background parameters
        parameter_path = '../DNM/multistage_stage1_SHGS/15year_stage1version2_batch_size_desktop_parameters.csv'
        return_path = '../DNM/multistage_stage1_SHGS/15year_stage1version2_dec_batch_size_desktop_parameters_unique.csv'
        delete_column = ['batch_size','test_auc']
        return_unique_background_parameters(parameter_path,return_path,delete_column)
    elif flag == 6: #the target hyperparameter is hidden_layer
        """
        (1) retrieve the number of layers from the mstruct
        (2) create a new column called number of layers and add it to the parameter file
        """
        parameter_path = "../DNM/multistage_stage1_SHGS/15year_stage1version2_hidden_layer_desktop_parameters.csv"
        df = pd.read_csv(parameter_path)
        new_column_values = []
        new_column_name = 'number_of_hidden_layer'
        if 'mstruct' not in df.columns:
            print('Wrong')
        else:
            for mstruct in df['mstruct']:
                new_column_values.append(len(eval(mstruct))-1)
        df[new_column_name] = new_column_values
        del df['mstruct']
        df.to_csv(parameter_path,index=False)
    elif flag == 7: #retrieve points based on steps
        parameter_path = '../DNM/multistage_stage1_SHGS/15year_stage1version2_batch_size_desktop_parameters.csv'
        target_hyperparameter = 'batch_size'
        step = 2
        retrieve_points_based_on_steps(parameter_path, target_hyperparameter, step)


"""
1=>expand the parameter values
4=>add test_auc to the parameter file
3=>plot
5=>return the index table
if the target hyperparmaeter is "number_of_hidden_layer":
    the order is 1,4,6,3
"""

"""
for 15year_dataset, we need to retrieve points based on the step first
1=>expand the parameter values
4=>add test_auc to the hyperparameter file
7=>retrieve points to get small figure
3=>plot
5=>return the index table
"""



