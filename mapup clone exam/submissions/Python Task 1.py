#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


def generate_car_matrix(df):

    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    #load the data
    
    df= pd.read_csv(r"C:\dhaval phy\mapup exam+\MapUp-Data-Assessment-F-main\MapUp-Data-Assessment-F-main\datasets\dataset-1.csv")
    
    # Pivot the DataFrame to create a matrix with id_1 as index, id_2 as columns, and car values as data
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    for col in car_matrix.columns:
        car_matrix.loc[col, col] = 0

    return car_matrix


# In[5]:


df= pd.read_csv(r"C:\dhaval phy\mapup exam+\MapUp-Data-Assessment-F-main\MapUp-Data-Assessment-F-main\datasets\dataset-1.csv")
   
result_matrix = generate_car_matrix(df)

print(result_matrix)


# In[6]:


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Add new categorical column 'car_type' based on 'car' values
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],labels=['low', 'medium', 'high'], right=False)

    # Calculate count of occurrences for each car_type category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts


# In[8]:


result_dict = get_type_count(df)
print(result_dict)


# In[9]:


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
        # Calculate the mean value of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes


# In[10]:


result_indices = get_bus_indexes(df)
print(result_indices)


# In[11]:


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
        # Group by 'route' and calculate the mean value of the 'truck' column
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average value of 'truck' is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    return selected_routes


# In[12]:


result_routes = filter_routes(df)
print(result_routes)


# In[13]:


def multiply_matrix(input_matrix):
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Create a deep copy to avoid modifying the original DataFrame
    modified_matrix = input_matrix.copy()

    # Apply the specified logic to modify each value
    for col in modified_matrix.columns:
        for idx in modified_matrix.index:
            if modified_matrix.at[idx, col] > 20:
                modified_matrix.at[idx, col] *= 0.75
            else:
                modified_matrix.at[idx, col] *= 1.25

    # Round the values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix


# In[14]:


# Generate car matrix using the first function
car_matrix = generate_car_matrix(df)

# Multiply and modify the values using the second function
modified_car_matrix = multiply_matrix(car_matrix)

# Print the modified matrix
print(modified_car_matrix)


# In[19]:


import pandas as pd

df= pd.read_csv(r"C:\dhaval phy\mapup exam+\MapUp-Data-Assessment-F-main\MapUp-Data-Assessment-F-main\datasets\dataset-2.csv")

def verify_time_completeness(df):
    # Convert startDay and endDay to categorical types to ensure correct ordering
    df['startDay'] = pd.Categorical(df['startDay'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
    df['endDay'] = pd.Categorical(df['endDay'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)

    # Convert categorical columns to strings before addition
    df['start_datetime'] = pd.to_datetime(df['startDay'].astype(str) + ' ' + df['startTime'], errors='coerce')
    df['end_datetime'] = pd.to_datetime(df['endDay'].astype(str) + ' ' + df['endTime'], errors='coerce')

    # Calculate the time duration for each entry
    df['duration'] = (df['end_datetime'] - df['start_datetime']).dt.total_seconds()

    # Check if each (id, id_2) pair has incorrect timestamps
    result = df.groupby(['id', 'id_2']).apply(lambda group: not (
        group['duration'].sum() >= 24 * 60 * 60 and
        group['start_datetime'].min().time() == pd.Timestamp('00:00:00').time() and
        group['end_datetime'].max().time() == pd.Timestamp('23:59:59').time() and
        len(group) == 7
    ))

    return result


# In[20]:


result = verify_time_completeness(df)

print(result)

