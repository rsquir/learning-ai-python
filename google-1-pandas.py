import numpy as np
import pandas as pd

# Create and populate a 5x2 NumPy array.
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])


# Create a Python list that holds the names of the two columns.
my_column_names = ['tempurature', 'activity']

# Create a DataFrame
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print entire DataFrame
print(my_dataframe)

# Create a new column named adjusted.
my_dataframe["adjusted"] = my_dataframe["activity"] + 2
print(my_dataframe)

print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), "\n")

print("Row #2:")
print(my_dataframe.iloc[[2]], "\n")

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], "\n")

print("Column tempurature:")
print(my_dataframe["tempurature"])


# Task 1
task_data = np.random.randint(0, 101, size=(3, 4))
task_column_names = ['Eleanor', 'Chidi', 'Tahani', "Jason"]
task_dataframe = pd.DataFrame(data=task_data, columns=task_column_names)

task_dataframe["Janet"] = task_dataframe["Tahani"] + task_dataframe["Jason"]
print(task_dataframe)

# Create a reference by assigning my_dataframe to a new variable.
print("Experiment with a reference:")
reference_to_df = task_dataframe

# Print the starting value of a particular cell.
print("  Starting value of task_df: %d" % task_dataframe['Jason'][1])
print("  Starting value of reference_to_df: %d\n" % reference_to_df['Jason'][1])

# Modify a cell in df.
task_dataframe.at[1, 'Jason'] = task_dataframe['Jason'][1] + 5
print("  Updated task_dataframe: %d" % task_dataframe['Jason'][1])
print("  Updates reference_to_df: %d" % reference_to_df['Jason'][1])

# then there's copy



