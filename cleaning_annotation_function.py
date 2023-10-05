#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[4]:


def process_csv_file(csv_file_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import os
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from ipywidgets import Dropdown, interactive, widgets, Output
    from scipy.interpolate import make_interp_spline
    
    # Define the column headers
    headers = ['frame', 'classid', 'id', 'x1', 'y1', 'width', 'height', 'a', 'b', 'c', 'd']

    # Read the CSV file into a DataFrame and assign the headers
    df_in = pd.read_csv(csv_file_path, header=None, names=headers)

    unique_individuals = df_in[['classid', 'id']].drop_duplicates()

    # Calculate the total number of unique individuals
    total_individuals = len(unique_individuals)
    total_boxes = len(df_in)

    # Print the total number of unique individuals
    print("Total Number of Unique Individuals (Male and Female):", total_individuals)
    print("Total Number of bounding boxes:", total_boxes)

    df_in['area'] = df_in['width'] * df_in['height']

    
    #CLASS ID ERROR
    # Finding classid error
    # Create a new column 'classid_error' and initialize it with 0
    df_in['classid_error'] = 0

    # Function to update 'classid_error' column based on 'classid' column
    def update_classid_error(row):
        if row['classid'] == -1:
            return 1
        else:
            return row['classid_error']

    # Apply the update_classid_error function to each row
    df_in['classid_error'] = df_in.apply(update_classid_error, axis=1)

    df_in['classid_error_frame'] = 0

    # Find frames with 'classid_error' entry of 1
    frames_with_classid_error = df_in[df_in['classid_error'] == 1]['frame'].unique()

    # Update 'classid_error_frame' column for the identified frames
    df_in.loc[df_in['frame'].isin(frames_with_classid_error), 'classid_error_frame'] = 1

    # Print frames with duplicates
    print("Frames with classid errors:", frames_with_classid_error)

    # Cleaning classid error
    # Define a threshold for closeness when comparing coordinates and area
    threshold = 20  # Adjust this value as needed
    area_threshold = 500
    # Delete the entries in the 'classid' and 'id' columns of frames with classid errors
    for frame in frames_with_classid_error:
        df_in.loc[(df_in['frame'] == frame) & (df_in['classid_error'] == 1), ['classid', 'id']] = None

    # Iterate through frames with classid error
    for frame in frames_with_classid_error:
        # Get the rows for the current frame with classid error
        error_frame_rows = df_in[(df_in['frame'] == frame) & (df_in['classid_error'] == 1)]

        # Find the previous frame
        previous_frame = df_in[df_in['frame'] == frame - 1]

        # Iterate through the rows with classid error in the current frame
        for index, row in error_frame_rows.iterrows():
            x = row['x1']
            y = row['y1']
            area = row['area']

            # Iterate through the rows of the previous frame
            for prev_index, prev_row in previous_frame.iterrows():
                prev_x = prev_row['x1']
                prev_y = prev_row['y1']
                prev_area = prev_row['area']
                prev_classid = prev_row['classid']
                prev_individual_id = prev_row['id']

                # Check if the coordinates and area are close enough
                if abs(x - prev_x) < threshold and abs(y - prev_y) < threshold and abs(area - prev_area) < area_threshold:
                    # Fill the missing entries in the current row from the previous row
                    df_in.at[index, 'classid'] = prev_classid
                    df_in.at[index, 'id'] = prev_individual_id

                    
    #DUPLICATE ERROR                
    # Finding duplicate frames
    # Create a new column 'duplicates' indicating if a row is a duplicate
    df_in['duplicates'] = df_in.groupby(['frame', 'classid'])['id'].transform(lambda x: x.duplicated(keep=False).astype(int))

    # Find unique frames with duplicates
    frames_with_duplicates = df_in.loc[df_in['duplicates'] == 1, 'frame'].unique()

    # Create 'duplicate_frame' column and set values based on 'frame' and 'frames_with_duplicates'
    df_in['duplicate_frame'] = df_in['frame'].apply(lambda x: 1 if x in frames_with_duplicates else 0)

    # Print frames with duplicates
    print("Frames with Duplicates:", frames_with_duplicates)

    # Remove specified columns
    columns_to_drop = ['duplicates', 'duplicate_frame', 'classid_error', 'classid_error_frame', 'area']
    df_in = df_in.drop(columns=columns_to_drop)

    
    #SAVE CSV 
    # Determine the path to save the edited CSV file in the same directory
    directory = os.path.dirname(csv_file_path)
    file_name_without_extension = os.path.splitext(os.path.basename(csv_file_path))[0]
    edited_csv_file_path = os.path.join(directory, f'Edited_{file_name_without_extension}.csv')

    # Save the edited DataFrame to CSV without headers in the same directory
    df_in.to_csv(edited_csv_file_path, index=False, header=False)
    print(f"Edited CSV file saved at: {edited_csv_file_path}")
    
    
    




# In[ ]:




