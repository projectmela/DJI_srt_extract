#!/usr/bin/env python
# coding: utf-8

# In[4]:

#Author : Dipin
#Function : Autocorrect the duplicates and class error
import pandas as pd
import os
import math
import argparse

class DataProcessor:
    def __init__(self):
        self.file_path = None
        self.df = None

    def filepath(self):
        self.file_path = input("Enter the file path for the annotation file: ")

    def checkfile(self):
        """
        Check if the given file path is valid and the file exists.
        Returns:
            bool: True if the file exists and the path is valid, False otherwise.
        """
        return os.path.exists(self.file_path) and os.path.isfile(self.file_path)

    def processcsv(self):
        """
        Reads the CSV file and changes it into a DataFrame.
        Returns:
            df_in: DataFrame containing details of the annotations
        """
        # Define the column headers
        headers = ['frame', 'classid', 'id', 'x1', 'y1', 'width', 'height', 'a', 'b', 'c', 'd']

        # Read the CSV file into a DataFrame and assign the headers
        self.df = pd.read_csv(self.file_path, header=None, names=headers)

        self.df.sort_values(by='frame', inplace=True)

        # Calculate 'area' column
        self.df['area'] = self.df['width'] * self.df['height']
       
 
    def classerror(self):
        """
        Cleans the class error by comparing the IoU value of the error box with the one in the previous frame.
        """
        # Create a new column 'classid_error' and initialize it with 0
        self.df['classid_error'] = 0

        # Function to update 'classid_error' column based on 'classid' column
        def update_classid_error(row):
            if row['classid'] == -1:
                return 1
            else:
                return row['classid_error']

        # Apply the update_classid_error function to each row
        self.df['classid_error'] = self.df.apply(update_classid_error, axis=1)

        self.df['classid_error_frame'] = 0

        # Find frames with 'classid_error' entry of 1
        frames_with_classid_error = self.df[self.df['classid_error'] == 1]['frame'].unique()

        # Update 'classid_error_frame' column for the identified frames
        self.df.loc[self.df['frame'].isin(frames_with_classid_error), 'classid_error_frame'] = 1

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
        frames_with_classid_error = self.df[self.df['classid_error'] == 1]['frame'].unique()

        for frame in frames_with_classid_error:
            error_frame_rows = self.df[(self.df['frame'] == frame) & (self.df['classid_error'] == 1)]
            previous_frame = self.df[self.df['frame'] == frame - 1]

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
                self.df.at[index, 'classid'] = corrected_classid
                self.df.at[index, 'id'] = corrected_id
 
    
    

    def duplicates(self):
        # Create a new column 'duplicates' indicating if a row is a duplicate
        self.df['duplicates'] = self.df.groupby(['frame', 'classid'])['id'].transform(lambda x: x.duplicated(keep=False).astype(int))

        # Find unique frames with duplicates
        frames_with_duplicates = self.df.loc[self.df['duplicates'] == 1, 'frame'].unique()

        duplicate_statements = []

        for index, row in self.df[self.df['duplicates'] == 1].iterrows():
            statement = f"ID {row['id']} (Class {row['classid']}) has duplicates in frame {row['frame']}"

            duplicate_statements.append(statement)

        # Print frames with duplicates
        print("Frames with Duplicates:", frames_with_duplicates)

        # Step 2: Create a list to store the corresponding unique entries of the frame column
        unique_frames = []

        # Iterate through frames_with_duplicates and append unique entries to unique_frames list
        for frame in frames_with_duplicates:
            unique_frame = self.df.loc[(self.df['frame'] == frame) & (self.df['duplicates'] == 1), 'frame'].iloc[0]
            unique_frames.append(unique_frame)

        for frame in unique_frames:
            frame_data = self.df[self.df['frame'] == frame]

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
                    # Remove duplicates based on the distance
                    for index, duplicate_row in frame_data.iterrows():
                        if index != min_distance_duplicate_index:
                            self.df.drop(index, inplace=True)

                else:
                    # Step 3: Remove duplicates with smaller area for each unique frame and id combination
                    for classid, individual_id in frame_data.groupby(['classid', 'id']):
                        duplicate_rows = individual_id[individual_id['duplicates'] == 1]
                        if len(duplicate_rows) > 1:
                            min_area_difference = float('inf')
                            min_area_duplicate_index = None

                            for index, duplicate_row in duplicate_rows.iterrows():
                                id_to_compare = duplicate_row['id']
                                prev_frame_entry = self.df[(self.df['frame'] == frame - 1) & (self.df['id'] == id_to_compare)]

                                if not prev_frame_entry.empty:
                                    area_difference = abs(duplicate_row['area'] - prev_frame_entry['area'].values[0])
                                    if area_difference < min_area_difference:
                                        min_area_difference = area_difference
                                        min_area_duplicate_index = index

                            # Drop the duplicate row with the larger area difference
                            for index, duplicate_row in duplicate_rows.iterrows():
                                if index != min_area_duplicate_index:
                                    self.df.drop(index, inplace=True)

    

    def savetocsv(self):
        """
        Saves the updated DataFrame as a CSV file.
        """
        if self.df is not None:
            # Columns to retain
            columns_to_retain = ['frame', 'classid', 'id', 'x1', 'y1', 'width', 'height', 'a', 'b', 'c', 'd']

            # Retain the specified columns
            self.df = self.df[columns_to_retain]

            # Extract the directory path from the input CSV file's path
            directory_path = os.path.dirname(self.file_path)

            # Extract the file name without extension from the input CSV file's path
            file_name_without_extension = os.path.splitext(os.path.basename(self.file_path))[0]

            # Define the CSV file path for saving in the same directory as the input file
            csv_file_path = os.path.join(directory_path, f'Edited_{file_name_without_extension}.csv')

            # Save the DataFrame to CSV without headers
            self.df.to_csv(csv_file_path, index=False, header=False)
            


    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('file_path', nargs='?', help='File path for the annotation file')
        args = parser.parse_args()

        if args.file_path:
            self.file_path = args.file_path
        else:
            self.filepath()  # Prompt for the file path interactively if not provided as an argument

        if self.checkfile():
            self.processcsv()
            self.classerror()
            self.duplicates()
            self.savetocsv()
        else:
            print("Invalid file path or file does not exist.")

if __name__ == "__main__":
    data_processor = DataProcessor()
    data_processor.main()


# In[ ]:




