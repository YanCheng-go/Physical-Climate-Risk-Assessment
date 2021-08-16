"""
Generate figures in the working paper.

1. % reduced capacity by turbine-cool types
2. Exceedance probability of design air and water temperatures.
3. Time series of climatic variables.

Note: the waterfall chart will be generated when executing thermal assessment (i.e., exe_thermal_assessment.py)
"""

import os

work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))  # work directory
report_folder_name = 'reports'
report_folder = os.path.join(work_directory, report_folder_name)
if not os.path.exists(report_folder):
    os.mkdir(report_folder)
assessment_report_folder = os.path.join(report_folder, 'assessment')
if not os.path.exists(assessment_report_folder):
    os.mkdir(assessment_report_folder)

# Visualize % reduced capacity by turbine-cool types as boxplots.
# Please run boxplots.py and remember to change the parameters accordingly.

# Visualize exceedance probability of design air and water temperatures.
print('...Start to visualize exceedance probability of design air and water temperatures...')
fp = os.path.join(work_directory, 'scripts', 'visualizations', 'probability_changes.py')
exec(open(fp).read())

# Visualize time series of climatic variables.
print('...Start to visualize time series of climatic variables...')
fp = os.path.join(work_directory, 'scripts', 'visualizations', 'climate_trajectories.py')
exec(open(fp).read())
