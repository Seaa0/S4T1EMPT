import numpy as np
import pandas as pd

dataset = pd.read_csv("rawDataset.csv")
print("Dataset (First 5 Values Transposed):\n", dataset.head().T)  # look at dataset
print()


dataset.drop(columns=["Gender"], inplace=True) # not using that variable
dataset.drop(columns=["Person ID"], inplace=True) # not using that variable

# Checking Column Names
print(
    "Dataset Column Names:\n", dataset.columns
)  # words are capitalised
print()

dataset = dataset.rename(str.lower, axis="columns")

print(
    "Processed Dataset Column Names:\n", dataset.columns
)  # All columns are uncapitalised
print()

# Checking for missing values
print(
    "No. of missing values per column in dataset:\n", dataset.isnull().sum()
)  # sleep disorder has missing values that mean there is no sleep disorder
print()

for column_name in dataset.columns:
    dataset[column_name] = dataset[column_name].apply(
        lambda x: 'None' if pd.isna(x) else x
    )

print(
    "No. of missing values per column in dataset after replacing no sleep disorders with 'None':\n",
    dataset.isnull().sum(),
)  # No more missing values
print()

print("Dataset (First 5 Values Transposed):\n", dataset.head().T)  # values are capitalised
print()

# Converting all values in dataset to lowercase
for column_name in dataset.columns:
    if dataset[column_name].dtype == 'object':
        dataset[column_name] = dataset[column_name].str.lower()

print("Dataset (First 5 Values Transposed) after making all values lowercase:\n", dataset.head().T)  # values are no longer capitalised

print()

dataset[['systolic blood pressure', 'diastolic blood pressure']] = dataset['blood pressure'].str.split('/', expand=True).astype('int') # splitting blood pressure into systolic and diastolic
dataset.drop(columns=["blood pressure"], inplace=True) # don't need any more
1
print(dataset.head().T)

# Binding Targets 
uniqueTargetValues = {}
for (
    column_name
) in dataset.columns:  # Finding all the unique column values in the dataset
    if dataset[column_name].dtype == 'object':
        uniqueTargetValues[column_name] = list(set(dataset.loc[:, column_name].tolist()))


print("All unique values among the values that are not numeric:\n", uniqueTargetValues)
print()

for column_name in uniqueTargetValues:  # Creating a new column in the dataset
    dataset[column_name] = dataset[column_name].apply(
        lambda x: uniqueTargetValues[column_name].index(x)*100000 # multiplied by 100000 to space them so far apart that clustering algorithm does not notice them being together, because they are separate factors altogether
    )

print('Final preprocessed dataset:\n',dataset.head().T) # look at final dataset

dataset.to_csv("preprocessedDataset.csv",index=False)
