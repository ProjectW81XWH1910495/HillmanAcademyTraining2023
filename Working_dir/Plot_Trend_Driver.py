import sys 
import os
import pandas as pd

from plottrend import *

filePath = r"H:\My Drive\Education\Internships\Hillman\2023\HillmanAcademyTraining2023\experiment_results\15year_stage1version2_momen_version2_merge_SGD.csv"
resultPath = r"H:\My Drive\Education\Internships\Hillman\2023\HillmanAcademyTraining2023\Working_dir\15year_stage1version2_momen_version2_merge_SGD_parameters.csv"

dictionary_column = 'parameters_and_values'

x_label = 'momen'
y_label = 'mean_test_auc'
title = 'test'
time = True

expand_dictionary_column(filePath,dictionary_column,resultPath)

df = pd.read_csv(filePath)
new_column = df[y_label]
new_column_name=y_label
add_new_column_tocsv(resultPath,new_column,new_column_name)

fig = plot_many_background_settings(resultPath, x_label, y_label, title, time)

dpi = 400
scale_factor = dpi / 96
pio.write_image(fig, f'x_{x_label}_y_{y_label}_trend_plot.png', scale=scale_factor)
fig.show()