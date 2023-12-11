#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install networkx')


# In[4]:


df= pd.read_csv(r"C:\dhaval phy\mapup exam+\MapUp-Data-Assessment-F-main\MapUp-Data-Assessment-F-main\datasets\dataset-3.csv")
df


# In[8]:


import pandas as pd
import networkx as nx

def calculate_distance_matrix(df):
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges with distances to the graph
    for _, row in df.iterrows():
        G.add_edge(row['id_start'], row['id_end'], distance=row['distance'])
        G.add_edge(row['id_end'], row['id_start'], distance=row['distance'])  # Bidirectional

    # Calculate shortest paths between all pairs of nodes
    all_pairs_shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G))

    # Create a DataFrame to store the distance matrix
    nodes = sorted(G.nodes())
    distance_matrix = pd.DataFrame(index=nodes, columns=nodes)

    # Populate the distance matrix
    for node1 in nodes:
        for node2 in nodes:
            # If the nodes are the same, set distance to 0
            if node1 == node2:
                distance_matrix.at[node1, node2] = 0
            else:
                # Get the shortest path distance between node1 and node2
                distance_matrix.at[node1, node2] = all_pairs_shortest_paths[node1][node2]

    return distance_matrix


result_matrix = calculate_distance_matrix(df)

print(result_matrix)


# In[9]:


def unroll_distance_matrix(distance_matrix):
    # Initialize an empty list to store rows of the resulting DataFrame
    result_rows = []

    # Iterate through the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Skip combinations where id_start is equal to id_end
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                result_rows.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a DataFrame from the list of rows
    result_df = pd.DataFrame(result_rows)

    return result_df


result_df = unroll_distance_matrix(result_matrix)

print(result_df)


# In[11]:


def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Filter the DataFrame for rows with the specified reference value
    reference_rows = df[df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    reference_average_distance = reference_rows['distance'].mean()

    # Calculate the lower and upper thresholds within 10% of the average
    lower_threshold = reference_average_distance - (reference_average_distance * 0.10)
    upper_threshold = reference_average_distance + (reference_average_distance * 0.10)

    # Filter the DataFrame for rows within the 10% threshold
    result_df = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]

    # Extract unique values from the 'id_start' column and sort them
    result_ids = sorted(result_df['id_start'].unique())

    return result_ids

reference_value = 223 
result_ids = find_ids_within_ten_percentage_threshold(result_df, reference_value)

print(result_ids)


# In[12]:


def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df


result_df_with_toll = calculate_toll_rate(result_df)

print(result_df_with_toll)


# In[25]:


import datetime

df= pd.read_csv(r"C:\dhaval phy\mapup exam+\MapUp-Data-Assessment-F-main\MapUp-Data-Assessment-F-main\datasets\dataset-2.csv")
df


# In[50]:


import datetime

def calculate_time_based_toll_rates(df):
    
    df= pd.read_csv(r"C:\dhaval phy\mapup exam+\MapUp-Data-Assessment-F-main\MapUp-Data-Assessment-F-main\datasets\dataset-2.csv")
    df
    
    # Convert startDay, startTime, endDay, and endTime to datetime
    df['startDay'] = pd.to_datetime(df['startDay'])
    df['startTime'] = pd.to_datetime(df['startTime'], format='%H:%M:%S').dt.time
    df['endDay'] = pd.to_datetime(df['endDay'])
    df['endTime'] = pd.to_datetime(df['endTime'], format='%H:%M:%S').dt.time

    # Define time ranges for weekdays and weekends
    weekday_time_ranges = [
        {'start': datetime.time(0, 0, 0), 'end': datetime.time(10, 0, 0), 'discount_factor': 0.8},
        {'start': datetime.time(10, 0, 0), 'end': datetime.time(18, 0, 0), 'discount_factor': 1.2},
        {'start': datetime.time(18, 0, 0), 'end': datetime.time(23, 59, 59), 'discount_factor': 0.8}
    ]
    weekend_time_ranges = [
        {'start': datetime.time(0, 0, 0), 'end': datetime.time(23, 59, 59), 'discount_factor': 0.7}
    ]

    # Initialize new columns for start_day, start_time, end_day, and end_time
    df['start_day'] = df['startDay'].dt.strftime('%A')
    df['start_time'] = df['startTime']
    df['end_day'] = df['endDay'].dt.strftime('%A')
    df['end_time'] = df['endTime']

    # Apply time-based toll rates
    for _, row in df.iterrows():
        if row['start_day'] in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            time_ranges = weekday_time_ranges
        else:
            time_ranges = weekend_time_ranges

        for time_range in time_ranges:
            if time_range['start'] <= row['start_time'] <= time_range['end'] and \
                    time_range['start'] <= row['end_time'] <= time_range['end']:
                df.loc[_, 'moto':'truck'] *= time_range['discount_factor']
                break  # Apply only the first matching time range

    # Drop intermediate columns used for time-based calculations
    df.drop(['start_day', 'start_time', 'end_day', 'end_time'], axis=1, inplace=True)

    return df



# In[51]:


result_df_with_time_based_toll = calculate_time_based_toll_rates(result_df_with_toll)

print(result_df_with_time_based_toll)


# In[ ]:





# In[ ]:




