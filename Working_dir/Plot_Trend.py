import sys 
import os
import pandas as pd

sys.path.append(os.path.abspath(r"H:\My Drive\Education\Internships\Hillman\2023\keras\utils"))
from plottrend import *

filePath = r"H:\My Drive\Education\Internships\Hillman\2023\HillmanAcademyTraining2023\experiment_results\15year_stage1version2_momen_version2_merge_SGD.csv"
resultPath = r"H:\My Drive\Education\Internships\Hillman\2023\HillmanAcademyTraining2023\Working_dir\15year_stage1version2_momen_version2_merge_SGD_parameters.csv"

dictionary_column = 'parameters_and_values'

x_label = 'momen'
y_label = 'test_auc'
title = 'Momemtum vs AUC'

expand_dictionary_column(filePath,dictionary_column,resultPath)

df = pd.read_csv(filePath)
new_column = df[y_label]
new_column_name=y_label
add_new_column_tocsv(resultPath,new_column,new_column_name)

plot_one_background_setting(resultPath, x_label, y_label, title)