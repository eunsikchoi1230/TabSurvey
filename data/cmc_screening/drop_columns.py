import pandas as pd
import os

# run script in current file directory
os.chdir(os.path.dirname(__file__))

# Load the dataset
path = "cmc_screening_raw.csv"  
df = pd.read_csv(path)

# drop the columns that are not needed
drop_columns = ['SBP', 'DBP', 'HT_dx', # For Hypertension 'HT_DM_drug' is also dropped
                'DM_dx', 'HT_DM_drug', # For Diabetes 
                'CAD_ECG', 'CAD_TMT', 'stroke_dx', 'AAA_dx', 'carotid_dx', 'CAD_dx', 'CAD_drug', # For Cardiovascular Disease
                'LDL_sum', # For Dyslipidemia
                'fw_days']

df = df.drop(drop_columns, axis=1)

# Save the modified dataset
df.to_csv("cmc_screening.csv", index=False)