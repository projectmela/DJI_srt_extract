#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Author : Dipin
#Function : Cleans the duplication and class error from the annotation

import pandas as pd
import os
import math

def main():
    file_path = input("Enter the file path for the annotation file: ")
    df = process_csv(file_path)
    df = class_error(df)
    df = duplicates(df)
    save_dataframe_to_csv(df, file_path)

def process_csv(file_path):
    
    """
    Reads the csv file and changes it into a dataframe.
    Args:
        file path: Location of the csv
        
    Return:
        df_in: data frame containing details of the annotations
 
        
    """
    
    # Define the column headers
    headers = ['frame', 'classid', 'id', 'x1', 'y1', 'width', 'height', 'a', 'b', 'c', 'd']

    # Read the CSV file into a DataFrame and assign the headers
    df_in = pd.read_csv(file_path, header=None, names=headers)

    df_in.sort_values(by='frame', inplace=True)
    
    # Calculate 'area' column
    df_in['area'] = df_in['width'] * df_in['height']
    return df_in

#find classid 
def class_error(df):
    """
    Takes the data frame and cleans the class error by comparing the iou value of the the error box with the one in the
    prevoius frame
  Args:
      df_in: data frame of the annotation
      
    Return:
        df_in: updated data frame
    
    """
    
    # Create a new column 'classid_error' and initialize it with 0
    df['classid_error'] = 0

    # Function to update 'classid_error' column based on 'classid' column
    def update_classid_error(row):
        if row['classid'] == -1:
            return 1
        else:
            return row['classid_error']

    # Apply the update_classid_error function to each row
    df['classid_error'] = df.apply(update_classid_error, axis=1)

    df['classid_error_frame'] = 0

    # Find frames with 'classid_error' entry of 1
    frames_with_classid_error = df[df['classid_error'] == 1]['frame'].unique()

    # Update 'classid_error_frame' column for the identified frames
    df.loc[df['frame'].isin(frames_with_classid_error), 'classid_error_frame'] = 1


    # Print frames with classid errors
    print("Frames with classid errors:", frames_with_classid_error)
    # Define a function to calculate IoU between two bounding boxes
    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        intersection_x1 = max(x1, x2)
        intersection_y1 = max(y1, y2)
        intersection_x2 = min(x1 + w1, x2 + w2)
        intersection_y2 = min(y1 + h1, y2 + h2)

        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
        union_area = w1 * h1 + w2 * h2 - intersection_area

        return intersection_area / union_area

    # Find frames with 'classid_error' entry of 1
    frames_with_classid_error = df[df['classid_error'] == 1]['frame'].unique()

    for frame in frames_with_classid_error:
        error_frame_rows = df[(df['frame'] == frame) & (df['classid_error'] == 1)]
        previous_frame = df[df['frame'] == frame - 1]

        for index, error_row in error_frame_rows.iterrows():
            max_iou = -1  # Initialize max IoU to a non-positive value
            corrected_classid = None
            corrected_id = None

            # Extract coordinates of the error row
            error_box = (error_row['x1'], error_row['y1'], error_row['width'], error_row['height'])

            # Iterate through rows of the previous frame
            for prev_index, prev_row in previous_frame.iterrows():
                prev_box = (prev_row['x1'], prev_row['y1'], prev_row['width'], prev_row['height'])

                # Calculate IoU between the error row and the previous row
                iou = calculate_iou(error_box, prev_box)

                if iou > max_iou:
                    max_iou = iou
                    corrected_classid = prev_row['classid']
                    corrected_id = prev_row['id']

            # Update the error row with the classid and id from the row with maximum IoU
            df.at[index, 'classid'] = corrected_classid
            df.at[index, 'id'] = corrected_id

    return df



#Finding duplicate frames
def duplicates(df):
    
    """
    Identifies the duplicates in each frame and cleans it using IOU value with the same individual in the previous frame

    Args:
        df_in: data frame of the annotation
        
    Return:
        df_in: Updated data frame
    """
    
    # Create a new column 'duplicates' indicating if a row is a duplicate
    df['duplicates'] = df.groupby(['frame', 'classid'])['id'].transform(lambda x: x.duplicated(keep=False).astype(int))
    duplicate_statements = []
    # Find unique frames with duplicates
    frames_with_duplicates = df.loc[df['duplicates'] == 1, 'frame'].unique()

    # Create 'duplicate_frame' column and set values based on 'frame' and 'frames_with_duplicates'
    df['duplicate_frame'] = df['frame'].apply(lambda x: 1 if x in frames_with_duplicates else 0)

    for index, row in df[df['duplicates'] == 1].iterrows(): 
        statement = f"ID {row['id']} (Class {row['classid']}) has duplicates in frame {row['frame']}"
        duplicate_statements.append(statement)


    # Print frames with duplicates
    print("Frames with Duplicates:", frames_with_duplicates)

    frames_with_duplicates = df.loc[df['duplicates'] == 1, 'frame'].unique()

    # Step 2: Create a list to store the corresponding unique entries of the frame column
    unique_frames = []

    # Iterate through frames_with_duplicates and append unique entries to unique_frames list
    for frame in frames_with_duplicates:
        unique_frame = df.loc[(df['frame'] == frame) & (df['duplicates'] == 1), 'frame'].iloc[0]
        unique_frames.append(unique_frame)

    for frame in unique_frames:
        frame_data = df[df['frame'] == frame]

        # Check if there are more than one duplicate rows in the frame
        if len(frame_data) > 1:
            min_distance = float('inf')
            min_distance_duplicate_index = None

            # Iterate through all combinations of duplicate rows
            for index1, duplicate_row1 in frame_data.iterrows():
                for index2, duplicate_row2 in frame_data.iterrows():
                    if index1 != index2:
                        x1 = duplicate_row1['x1']
                        y1 = duplicate_row1['y1']
                        x2 = duplicate_row2['x1']
                        y2 = duplicate_row2['y1']

                        # Calculate the Euclidean distance between the coordinates of the two duplicates
                        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                        if distance < min_distance:
                            min_distance = distance
                            min_distance_duplicate_index = index1

            # Check if the minimum distance is greater than 100
            if min_distance > 100:
                #remove dupliactes based on the distance
                import pandas as pd

                # Read the DataFrame from your CSV file or use the existing DataFrame df_in
                # df_in = pd.read_csv('your_csv_file.csv')

                # Step 1: Find unique frames with duplicates
                frames_with_duplicates = df.loc[df['duplicates'] == 1, 'frame'].unique()

                # Step 2: Create a list to store the corresponding unique entries of the frame column
                unique_frames = []

                # Iterate through frames_with_duplicates and append unique entries to unique_frames list
                for frame in frames_with_duplicates:
                    unique_frame = df.loc[(df['frame'] == frame) & (df['duplicates'] == 1), 'frame'].iloc[0]
                    unique_frames.append(unique_frame)

                for frame in unique_frames:
                    frame_data = df[df['frame'] == frame]
                    prev_frame = frame - 1

                    for classid, individual_id in frame_data.groupby(['classid', 'id']):
                        duplicate_rows = individual_id[individual_id['duplicates'] == 1]
                        if len(duplicate_rows) > 1:
                            min_distance = float('inf')
                            min_distance_duplicate_index = None

                            for index, duplicate_row in duplicate_rows.iterrows():
                                id_to_compare = duplicate_row['id']
                                prev_frame_entry = df[(df['frame'] == prev_frame) & (df['id'] == id_to_compare)]

                                if not prev_frame_entry.empty:
                                    x = duplicate_row['x1']
                                    y = duplicate_row['y1']

                                    prev_x = prev_frame_entry['x1'].values[0]
                                    prev_y = prev_frame_entry['y1'].values[0]

                                    # Calculate the Euclidean distance between the coordinates
                                    distance = ((x - prev_x) ** 2 + (y - prev_y) ** 2) ** 0.5

                                    if distance < min_distance:
                                        min_distance = distance
                                        min_distance_duplicate_index = index

                            # Drop the duplicate rows that do not have the smallest distance
                            duplicate_rows_to_drop = duplicate_rows[duplicate_rows.index != min_distance_duplicate_index]
                            df.drop(duplicate_rows_to_drop.index, inplace=True)

            else:                       
                frames_with_duplicates = df.loc[df['duplicates'] == 1, 'frame'].unique()

                # Step 2: Create a list to store the corresponding unique entries of the frame column
                unique_frames = []

                # Iterate through frames_with_duplicates and append unique entries to unique_frames list
                for frame in frames_with_duplicates:
                    unique_frame = df.loc[(df['frame'] == frame) & (df['duplicates'] == 1), 'frame'].iloc[0]
                    unique_frames.append(unique_frame)

                # Step 3: Remove duplicates with smaller area for each unique frame and id combination
                for frame in unique_frames:
                    frame_data = df[df['frame'] == frame]
                    prev_frame = frame - 1

                    for classid, individual_id in frame_data.groupby(['classid', 'id']):
                        duplicate_rows = individual_id[individual_id['duplicates'] == 1]
                        if len(duplicate_rows) > 1:
                            min_area_difference = float('inf')
                            min_area_duplicate_index = None

                            for index, duplicate_row in duplicate_rows.iterrows():
                                id_to_compare = duplicate_row['id']
                                prev_frame_entry = df[(df['frame'] == prev_frame) & (df['id'] == id_to_compare)]

                                if not prev_frame_entry.empty:
                                    area_difference = abs(duplicate_row['area'] - prev_frame_entry['area'].values[0])
                                    if area_difference < min_area_difference:
                                        min_area_difference = area_difference
                                        min_area_duplicate_index = index

                            # Drop the duplicate row with the larger area
                            duplicate_rows_to_drop = duplicate_rows[duplicate_rows.index != min_area_duplicate_index]
                            df.drop(duplicate_rows_to_drop.index, inplace=True)

    return df    

    

def save_dataframe_to_csv(df, input_csv_file_path):
    
    """
    Saves the updated datafram as a csv 
    """
    columns_to_drop = ['duplicates', 'duplicate_frame', 'classid_error', 'classid_error_frame', 'area']
    # Drop the specified columns
    df = df.drop(columns=columns_to_drop)
    
    # Extract the directory path from the input CSV file's path
    directory_path = os.path.dirname(input_csv_file_path)
    
    # Extract the file name without extension from the input CSV file's path
    file_name_without_extension = os.path.splitext(os.path.basename(input_csv_file_path))[0]
    
    # Define the CSV file path for saving in the same directory as the input file
    csv_file_path = os.path.join(directory_path, f'Edited_{file_name_without_extension}.csv')
    
    # Save the DataFrame to CSV without headers
    df.to_csv(csv_file_path, index=False, header=False)

if __name__ == "__main__":
    main()


# In[ ]:




