import pandas as pd
import os

# Run script in current file directory
os.chdir(os.path.dirname(__file__))

# Load the dataset
path = "cmc_screening.csv"  
df = pd.read_csv(path)

# Print column names of df
print(df.columns)

# Get indices of categorical columns
cat_columns = ['sex', 'Prot_Test', 'CAD_echo', 'PAOD_sono', 'CAD_CT',
       'stroke_MR', 'PAOD_dx', 'IFG', 'dyslipid', 'glaucoma', 'cancer_dx', 'alcohol', 'smoke', 'marriage', 'education', 'job',
       'income', 'menopause', 'chest_pain', 'CAD_fx', 'IPSS',
       'exercise_intensity']
cat_columns_idx = [df.columns.get_loc(col) for col in cat_columns]
print(cat_columns_idx)

# Print unique values for each categorical columns
for col in cat_columns:
    unique_values = df[col].unique()
    print(f"Unique values for column {col}: {unique_values}")

# Print percentage of missing values for each column
missing_values = df.isnull().mean() * 100
print(missing_values)
