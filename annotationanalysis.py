#!/usr/bin/env python
# coding: utf-8

# In[9]:

#Author : Dipin
#Function : Analyze the annotated files
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import argparse

class DataProcessor:
    def __init__(self):
        self.file_path = None
        self.df = None
        self.unique_frames_count = None
        self.total_individuals = None
        self.unique_individuals = None
        self.csv_file_name = None
        self.input_directory = None

    def filepath(self):
        """
        Asks the user for the file location and saves it in file_path.
        """
        self.file_path = input("Enter the file path for the annotation file: ")

    def validfile(self):
        """
        Check if the given file path is valid and the file exists.
        Returns:
            bool: True if the file exists and the path is valid, False otherwise.
        """
        return os.path.exists(self.file_path) and os.path.isfile(self.file_path)
    
    def processcsv(self):
        """
        Reads the CSV file and changes it into a DataFrame.
        Initializes other attributes.
        Args:
        file path: Location of the csv
        
        Return:
        df: data frame containing details of the annotations
        unique_frames_count : total no of frames in the annotation
        total_individuals: total no of ids
        unique_individuals: excluding drones and classid errors
        csv_file_name: Name of the file
        input_directory: file path
        
        """
          # Define the column headers
        headers = ['frame', 'classid', 'id', 'x1', 'y1', 'width', 'height', 'a', 'b', 'c', 'd']
            # Read the CSV file into a DataFrame and assign the headers
        self.df = pd.read_csv(self.file_path, header=None, names=headers)
        
        self.input_directory = os.path.dirname(self.file_path)
        self.csv_file_name = os.path.splitext(os.path.basename(self.file_path))[0]
        
        # Find the number of unique frames in the 'frame' column
        self.unique_frames_count = self.df['frame'].nunique()

        # Get unique individuals
        unique_individuals = self.df[['classid', 'id']].drop_duplicates()

        # Remove entries with classid 2 or -1
        unique_individuals = unique_individuals[(unique_individuals['classid'] != 2) & (unique_individuals['classid'] != -1)]

        # Calculate the total number of unique individuals
        self.total_individuals = len(unique_individuals)

        # Calculate 'area' column
        self.df['area'] = self.df['width'] * self.df['height']
        
      #find classid  
    def classerror(self):
        """
        Identifies the classid error and updates the DataFrame.
        Args:
        df: data frame of the annotation
        
        Return:
        df: Updated data frame
        frames_with_classid_error: Frames that has class error
        box_classid_error: boundind boxes with class error
        classid_error_statements : Statement to be printed in the txt file
        
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
        self.frames_with_classid_error = frames_with_classid_error
        
        # Update 'classid_error_frame' column for the identified frames
        self.df.loc[self.df['frame'].isin(frames_with_classid_error), 'classid_error_frame'] = 1

        box_classid_error = self.df['classid_error'].sum()
        self.box_classid_error = box_classid_error
        
        # Print frames with classid errors
        print("Frames with classid errors:", frames_with_classid_error)
        
        classid_error_statements = []
        for frame in frames_with_classid_error:
            classid_error_entries = self.df[(self.df['frame'] == frame) & (self.df['classid_error'] == 1)]
            classid_id_pairs = set()  # Use a set to ensure unique pairs
            for index, row in classid_error_entries.iterrows():
                classid_id_pairs.add(f"({row['id']})")
            statement = f"Frame {frame} = {', '.join(classid_id_pairs)}"
            classid_error_statements.append(statement)
            
        self.classid_error_statements = classid_error_statements   

        
     
    #Finding duplicate frames
    def duplicates(self):
        """
        Identifies the duplicates in each frame
        
        Args:
        df: data frame of the annotation
        
        Return:
        df: Updated data frame
        frames_with_duplicates: Frames that has duplicates error
        duplicate_statements : Statement to be printed in the txt file
        
        """
        # Create a new column 'duplicates' indicating if a row is a duplicate
        self.df['duplicates'] = self.df.groupby(['frame', 'classid'])['id'].transform(lambda x: x.duplicated(keep=False).astype(int))
        duplicate_statements = []

        # Find unique frames with duplicates
        frames_with_duplicates = self.df.loc[self.df['duplicates'] == 1, 'frame'].unique()
        self.frames_with_duplicates = frames_with_duplicates

        # Create 'duplicate_frame' column and set values based on 'frame' and 'frames_with_duplicates'
        self.df['duplicate_frame'] = self.df['frame'].apply(lambda x: 1 if x in frames_with_duplicates else 0)

        for frame in frames_with_duplicates:
            duplicate_entries = self.df[(self.df['frame'] == frame) & (self.df['duplicates'] == 1)]
            classid_id_pairs = set()  # Use a set to ensure unique pairs
            for index, row in duplicate_entries.iterrows():
                classid_id_pairs.add(f"({row['classid']}:{row['id']})")
            statement = f"Frame {frame} = {', '.join(classid_id_pairs)}"
            duplicate_statements.append(statement)
            
        self.duplicate_statements = duplicate_statements    
        print("Frames with Duplicates:", self.frames_with_duplicates)

    
    def iou(self):
        """
        Calculates the iou value between each frame of an individual and finds the frame that has 0 IOU values

        Args:
        df_in: data frame of the annotation

        Return:
        df_in: Updated data frame
        row_with_iou_0: Frames that has IOU = 0 with the previous frame
        sentences_iou : Statement to be printed in the txt file 
    
        """
        
         #intersection over union
        # Sort the DataFrame by 'classid', 'id', and 'frame' in ascending order
        self.df.sort_values(by=['classid', 'id', 'frame'], inplace=True)
       
        # Create a new column to store the IoU values
        self.df['iou'] = 0.0 # Initialize with zeros

        last_bbox_dict = {}
          # Iterate through the DataFrame       
        for index, row in self.df.iterrows():
            classid = row['classid']
            individual_id = row['id']
            frame = row['frame']
            x1 = row['x1']
            y1 = row['y1']
            width = row['width']
            height = row['height']
             # Check if the individual within a class was present in the previous frame
            if (classid, individual_id) in last_bbox_dict:
                last_x1, last_y1, last_width, last_height = last_bbox_dict[(classid, individual_id)]
                # Calculate the coordinates of the current bounding box
                x2 = x1 + width
                y2 = y1 + height
                
               # Calculate the coordinates of the last known bounding box
                last_x2 = last_x1 + last_width
                last_y2 = last_y1 + last_height
                # Calculate the intersection coordinates
                intersection_x1 = max(x1, last_x1)
                intersection_y1 = max(y1, last_y1)
                intersection_x2 = min(x2, last_x2)
                intersection_y2 = min(y2, last_y2)
             # Calculate the areas of the current and last bounding boxes
                area_current = (x2 - x1) * (y2 - y1)
                area_last = (last_x2 - last_x1) * (last_y2 - last_y1)
                 # Calculate the area of intersection
                area_intersection = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
               # Calculate the IoU
                iou = area_intersection / (area_current + area_last - area_intersection)
                 # Update the DataFrame with the calculated IoU
                self.df.at[index, 'iou'] = iou
             # Update the last bounding box coordinates for the individual within the class
            last_bbox_dict[(classid, individual_id)] = (x1, y1, width, height)
         #Finding boxes with IOU=0   
        self.row_with_iou_0 = self.df[(self.df['iou'] == 0) & (self.df['frame'] != self.df.groupby(['classid', 'id'])['frame'].transform('min')) & (self.df['classid'] != -1)]
       # Initialize an empty list to store sentences
        self.sentences_iou = []
         # Iterate through the rows of row_with_iou_0 DataFrame
        for index, row in self.row_with_iou_0.iterrows():
            classid = int(row['classid'])
            id_value = int(row['id'])
            frame_start = int(row['frame'])
            frame_end = frame_start - 1   # Subtract 1 from frame_start
             # Create the sentence in the desired format
            sentence_iou = f'ID {id_value} (Class {classid}) has IOU value 0 between the frames {frame_end} and {frame_start}'
            # Append the sentence to the list
            self.sentences_iou.append(sentence_iou)
        # Print the sentences or save them as needed
        for sentence_iou in self.sentences_iou:
            print(sentence_iou)    
        
    def areaanalysis(self):
        """
        Filtering out boxes with an area above 4000
        Args:
        df: data frame of the annotation
       
         Return:
         total_individuals_area: number of individuals above 4000
         unique_individuals_area: dataframe of individuals above 4000
         frames_unique_entries_area: Frames with individuals with area above 4000
         statements_area: statement to be printed in the text

        """

    
         # Filter rows where 'area' is greater than 4000
        rows_with_high_area = self.df[self.df['area'] > 4000]
        self.statements_area = []
        #Removing classid error and drone(ID=2)
        frames_unique_high_area = rows_with_high_area[(rows_with_high_area['classid'] != -1) & (rows_with_high_area['classid'] != 2)]
        self.frames_unique_entries_area = frames_unique_high_area['frame'].nunique()

        
        # Print the number of unique entries
        print("Number of unique frames with area above 4000:", self.frames_unique_entries_area)
        
        # Get the unique combination of 'classid' and 'id' values for all individuals
        unique_individuals_area = rows_with_high_area[['classid', 'id']].drop_duplicates()
       #Removeing classid error and drone
        unique_individuals_area = unique_individuals_area[(unique_individuals_area['classid'] != -1) & (unique_individuals_area['classid'] != 2)]
        self.total_individuals_area = len(unique_individuals_area)
        self.unique_individuals_area = unique_individuals_area
   
        # Print the total number of unique individuals
        print("Total Number of Unique Individuals (Male and Female):", self.total_individuals_area)

         # Check if frames_unique_entries_area is greater than 0
        if self.frames_unique_entries_area > 0:
             # Create an empty DataFrame to store the results
            df_area = unique_individuals_area.copy()
            
            # Function to count frames where area > 4000 for an individual
            def count_frames_above_4000(classid, individual_id):
                filtered_df = self.df[(self.df['classid'] == classid) & (self.df['id'] == individual_id)]
                frames_above_4000 = len(filtered_df[filtered_df['area'] > 4000])
                return frames_above_4000
            
           # Apply the function to each row of df_area and store the result in a new column
            df_area['no of frames>4000'] = df_area.apply(lambda row: count_frames_above_4000(row['classid'], row['id']), axis=1)

        # Create an empty list to store the statements            
            statements_area = []

            # Function to count frames where area > 4000 for an individual and generate statements
            def generate_statements(classid, individual_id):
                filtered_df = self.df[(self.df['classid'] == classid) & (self.df['id'] == individual_id)]
                frames_above_4000 = len(filtered_df[filtered_df['area'] > 4000])
                # Generate the statement and append it to the list
                statement = f"ID {individual_id} (Class {classid}) = {frames_above_4000} frames."
                statements_area.append(statement)

            # Apply the function to each row of df_area
            df_area.apply(lambda row: generate_statements(row['classid'], row['id']), axis=1)

            self.statements_area = statements_area
             
           # Print the generated statements
            for statement in statements_area:
                print(statement)    
        else:
            print("No unique entries found above the threshold.")

    def disappearingboxes(self):
        """
        Finds the frames where an individual disappears and reappears
        
        Args:
        df_in: data frame of the annotation
        frames_with_classid_error: Frames that has class error
        frames_with_duplicates: Frames that has duplicates error
        
        
        Return:
        filtered_disappearance_statements : Statement to be printed in the txt file
    
        
        """

        # Sort the DataFrame by 'id' and 'frame' in ascending order
        self.df.sort_values(by=['id', 'frame'], inplace=True)

        # Create a dictionary to store the last frame for each individual
        last_frame_dict = {}
        disappearance_statements = []  # Initialize a list to store disappearance statements

        # Iterate through the DataFrame
        for index, row in self.df.iterrows():
            classid = int(row['classid'])
            individual_id = int(row['id'])
            frame = int(row['frame'])

            # Check if the individual was present in the previous frame
            if (classid, individual_id) in last_frame_dict:
                last_frame = last_frame_dict[(classid, individual_id)]

                if frame != last_frame + 1:
                    statement = f"ID {individual_id} (Class {classid}) disappeared in frame {last_frame} and reappeared in frame {frame}"
                    disappearance_statements.append(statement)  # Collect the statements

            # Update the last frame for the individual
            last_frame_dict[(classid, individual_id)] = frame

        # Create a new list without entries containing 'Class -1' and not having one less of the numbers stored in 'frames_with_classid_error'
        filtered_disappearance_statements = [
            statement for statement in disappearance_statements 
            if 'Class -1' not in statement 
            and all(f"frame {frame - 1}" not in statement for frame in self.frames_with_classid_error)
            and all(f"frame {frame}" not in statement for frame in self.frames_with_duplicates)
        ]
        self.filtered_disappearance_statements = filtered_disappearance_statements


    def plotsonarea(self):
        """
        Plot scatter plot with area on the x-axis and the number of bounding boxes on the y-axis
        

    
        Args:
        df: data frame of the annotation
        csv_file_name: Name of the file
        input_directory: file path
        
     
        
        """
     
        def calculate_percentile(df_in, column, percentile):
            df_in = df_in.sort_values(by=column)
            index = (len(df_in) - 1) * percentile
            floor_index = math.floor(index)
            ceil_index = math.ceil(index)
            if floor_index == ceil_index:
                return df_in.iloc[int(index)][column]
            else:
                floor_value = df_in.iloc[floor_index][column] 
                ceil_value = df_in.iloc[ceil_index][column]
                return floor_value + (ceil_value - floor_value) * (index - floor_index)  
            
        # Filter the DataFrame to exclude rows where 'classid' is 2
        df_area_filtered = self.df[self.df['classid'] != 2]

        # Calculate the frequency of each area value
        area_counts = df_area_filtered['area'].value_counts().sort_index()

        # Extract area values and their frequencies
        area_values = area_counts.index
        frequencies = area_counts.values

        # Calculate the mean area
        mean_area = df_area_filtered['area'].mean()

        # Calculate the standard deviation of the area values
        std_dev_area = round(df_area_filtered['area'].std(), 3)

        # Calculate the values for mean + standard deviation and mean - standard deviation
        mean_plus_std = mean_area + std_dev_area
        mean_minus_std = mean_area - std_dev_area

        # Calculate the values for mean + 2*standard deviation and mean - 2*standard deviation
        mean_plus_2std = mean_area + 2 * std_dev_area
        mean_minus_2std = mean_area - 2 * std_dev_area

        # Finding the 95th percentile
        percentile = calculate_percentile(self.df, 'area', 0.95)

        # Create a scatter plot with green color
        plt.figure(figsize=(15, 11))
        plt.scatter(area_values, frequencies, color="green", marker='o', s=12)
        plt.xlabel('Area')
        plt.ylabel('No of bounding boxes')
        plt.title('Scatter Plot of Bounding Box Areas')
        plt.grid()

        # Set the X-axis range to the minimum and maximum area values
        plt.xlim(min(area_values), max(area_values))


        # Annotate the plot with relevant statistics
        plt.annotate(f'Min Area: {min(area_values)}', xy=(0.75, 0.95), xycoords='axes fraction', color='black', fontsize=12)
        plt.annotate(f'Max Area: {max(area_values)}', xy=(0.75, 0.9), xycoords='axes fraction', color='black', fontsize=12)
        plt.annotate(f'μ = Mean Area: {mean_area:.2f}', xy=(0.75, 0.85), xycoords='axes fraction', color='black', fontsize=12)
        plt.annotate(f'σ = Standard Deviation: {std_dev_area}', xy=(0.75, 0.8), xycoords='axes fraction', color='black', fontsize=12)
        plt.annotate(f'P = 95th Percentile(red): {percentile}', xy=(0.75, 0.75), xycoords='axes fraction', color='black', fontsize=12)

        # Place an asterisk precisely on the x-axis at the mean_area value
        plt.annotate('|', xy=(mean_area, 0), xycoords='data', color='black', fontsize=10, ha='center', va='center')
        plt.annotate('μ', xy=(mean_area, 0), xycoords='data', color='black', fontsize=12, ha='center', va='top')

        # Add annotations for mean + standard deviation and mean - standard deviation on the x-axis below the axis
        plt.annotate('μ+σ', xy=(mean_plus_std, 0), xycoords='data', color='black', fontsize=12, ha='center', va='top')
        plt.annotate('μ-σ', xy=(mean_minus_std, 0), xycoords='data', color='black', fontsize=12, ha='center', va='top')
        plt.annotate('|', xy=(mean_plus_std, 0), xycoords='data', color='black', fontsize=10, ha='center', va='center')
        plt.annotate('|', xy=(mean_minus_std, 0), xycoords='data', color='black', fontsize=10, ha='center', va='center')

        # Add annotations for mean + 2*standard deviation and mean - 2*standard deviation on the x-axis below the axis
        plt.annotate('μ+2σ', xy=(mean_plus_2std, 0), xycoords='data', color='black', fontsize=12, ha='center', va='top')
        plt.annotate('μ-2σ', xy=(mean_minus_2std, 0), xycoords='data', color='black', fontsize=12, ha='center', va='top')
        plt.annotate('|', xy=(mean_plus_2std, 0), xycoords='data', color='black', fontsize=10, ha='center', va='center')
        plt.annotate('|', xy=(mean_minus_2std, 0), xycoords='data', color='black', fontsize=10, ha='center', va='center')

        plt.annotate('|', xy=(percentile, 0), xycoords='data', color='red', fontsize=15, ha='center', va='center')


        # Save the plot to a file
        image_filename_scatter = os.path.join(self.input_directory, f'{self.csv_file_name}_area_scatter_plot.png')
        plt.savefig(image_filename_scatter, bbox_inches='tight', format='png')

        if self.args.showplot:
            # Display the plot only if the user provides the --showplot option
            plt.show()

        
        
        #area vs aspect ratio
        """
        Plot graph with aspect ratio in the x axis and area in the y axis of each row.Each individual have different colour
        """
        df_inc=self.df.copy()
        # Create a custom color palette with 90 distinct colors
        custom_colors = ['#FF5733', '#33FF57', '#5733FF', '#33FFCC', '#FF33CC', '#CC33FF',
                        '#FF3366', '#33CCFF', '#66FF33', '#3366FF', '#66CCFF', '#CCFF33',
                        '#FFCC66', '#CC66FF', '#66FFCC', '#FF6666', '#6666FF', '#33CC66',
                        '#66CC33', '#3366CC', '#CC3366', '#66CCCC', '#CCCC66', '#666666',
                        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF',
                        '#FF9900', '#9900FF', '#0099FF', '#FF0099', '#99FF00', '#00FF99',
                        '#990000', '#009900', '#000099', '#999900', '#009999', '#990099',
                        '#660000', '#006600', '#000066', '#666600', '#006666', '#660066',
                        '#330000', '#003300', '#000033', '#333300', '#003333', '#330033',
                        '#110000', '#001100', '#000011', '#111100', '#001111', '#110011',
                        '#880000', '#008800', '#000088', '#888800', '#008888', '#880088',
                        '#440000', '#004400', '#000044', '#444400', '#004444', '#440044',
                        '#220000', '#002200', '#000022', '#222200', '#002222', '#220022']

        # Assuming you have a DataFrame 'df_inc' with 'classid' and 'id' columns

        # Exclude class IDs -1 and 2(drone) from the DataFrame
        df_filtered = df_inc[(df_inc['classid'] != -1) & (df_inc['classid'] != 2)]

        # Calculate the ratio of short side to long side (width/height)
        df_filtered['short_side'] = df_filtered[['width', 'height']].min(axis=1)
        df_filtered['long_side'] = df_filtered[['width', 'height']].max(axis=1)
        df_filtered['ratio'] = df_filtered['short_side'] / df_filtered['long_side']

        # Create a scatter plot with ratio on the x-axis and area on the y-axis
        plt.figure(figsize=(10, 6))

        # Loop through unique individual IDs and plot points with different colors
        unique_individual_ids = df_filtered['id'].unique()
        for i, individual_id in enumerate(unique_individual_ids):
            individual_data = df_filtered[df_filtered['id'] == individual_id]
            plt.scatter(
                individual_data['ratio'],
                individual_data['area'],
                s=10,
                alpha=0.7,
                label=f'Individual {individual_id}',
                c=custom_colors[i % len(custom_colors)]
            )

        # Add labels and title
        plt.xlabel('Short Side / Long Side (Ratio)')
        plt.ylabel('Area')
        plt.title('Scatter Plot of Short Side/Long Side vs. Area')
        plt.grid()
        # Set the X-axis range to the minimum and maximum area values
        area_values = df_filtered['area']
        plt.ylim(min(area_values), max(area_values))



        # Save the plot as an image
        image_filename_scatter = os.path.join(self.input_directory, f'{self.csv_file_name}_Aspect_ratio_vs_area.png')
        plt.savefig(image_filename_scatter, bbox_inches='tight', format='png')

        if self.args.showplot:
            # Display the plot only if the user provides the --showplot option
            plt.show()

 

    def individualplots(self):
        """
        Create plots of frame vs area for individuals with an area above 4000
        
        Args:
        df: data frame of the annotation
        unique_individuals_area: dataframe of individuals above 4000
        total_individuals_area: number of individuals above 4000
        csv_file_name: Name of the file
        input_directory: file path
    
        
        """
        if 50 > self.total_individuals_area > 0:
            # Create a single figure to contain all plots
            fig, axes = plt.subplots(len(self.unique_individuals_area), 1, figsize=(20, 8 * len(self.unique_individuals_area)))

            for i, (_, row) in enumerate(self.unique_individuals_area.iterrows()):
                classid, individual_id = row['classid'], row['id']

                # Select the current subplot
                ax = axes[i]

                # Clear the previous plot from the subplot
                ax.clear()

                filtered_df = self.df[(self.df['classid'] == classid) & (self.df['id'] == individual_id)]
                grouped_df = filtered_df.groupby('frame')['area'].mean()

                ax.set_title(f'Area vs. Frame for Class {classid}, Individual {individual_id}')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Area')

                # Plot the data points with a slight curve
                x = grouped_df.index
                y = grouped_df.values
                ax.plot(x, y, color='green', linestyle='-', linewidth=1)  # Adjust the linewidth for a thicker line

                # Set X axis ticks to be every 200 frames
                ax.set_xticks(range(0, max(x) + 1, 200))


                # Show gridlines
                ax.grid(True)

            # Save the figure containing all plots with the CSV file name as part of the filename
            image_filename_individual = os.path.join(self.input_directory, f'{self.csv_file_name}_individual_with_area_above_4000.png')
            plt.savefig(image_filename_individual, bbox_inches='tight', format='png')

            if self.args.showplot:
                # Display the plot only if the user provides the --showplot option
                plt.show()


    def dataframetocsv(self):
        """
        Saves the updated CSV file
        
        Args:
        df_in: data frame of the annotation
        file_path
        
        
        """
        # Extract the directory path from the input CSV file's path
        output_directory = os.path.dirname(self.file_path)
        self.df.sort_values(by='frame', inplace=True)

        # Extract the file name without extension from the input CSV file's path
        file_name_without_extension = os.path.splitext(os.path.basename(self.file_path))[0]

        # Define the CSV file path for saving
        output_file_path = os.path.join(output_directory, f'Analysed_{file_name_without_extension}.csv')

        # Save the DataFrame to CSV without headers
        self.df.to_csv(output_file_path, index=False)

        return output_file_path

    def textsave(self):
        """
        Saves the errors and information to a text file
        """
        # Define the text file name with the desired format
        text_file_name = os.path.join(self.input_directory, f'Analysed_{self.csv_file_name}.txt')

        # Check if there are frames with classid errors, duplicates, or other conditions
        with open(text_file_name, 'w') as file:
            if len(self.frames_with_classid_error) > 0:
                file.write("Frames with classid errors: \n")
                file.write(', '.join(map(str, self.frames_with_classid_error)) + '\n')
                file.write("Total no of boxes with classid error: {}\n".format(self.box_classid_error))
                file.write("Frame = (ID) \n")
                for statement in self.classid_error_statements:
                    file.write(statement + '\n')
                file.write("\n")  # Add a line gap

            if len(self.frames_with_duplicates) > 0:
                file.write("Frames with duplicates:\n")
                file.write(', '.join(map(str, self.frames_with_duplicates)) + '\n')
                file.write("Frame = (Class:ID) \n")
                for statement in self.duplicate_statements:
                    file.write(statement + '\n')
                file.write("\n")  # Add a line gap

            if self.frames_unique_entries_area > 0:
                file.write(f"\nOut of {self.unique_frames_count} frames, no of frames with bounding box area greater than 4000: {self.frames_unique_entries_area}\n")

            if self.total_individuals_area > 0:
                file.write(f"Out of {self.total_individuals} individuals, no of individuals with bounding box area greater than 4000: {self.total_individuals_area}\n")
                for statement in self.statements_area:
                    file.write(statement + '\n')
                file.write("\n")  # Add a line gap

            if len(self.filtered_disappearance_statements) > 0:
                file.write("\nDisappearance Statements:\n")
                for statement in self.filtered_disappearance_statements:
                    file.write(statement + '\n')
                file.write("\n")  # Add a line gap

            if len(self.sentences_iou) > 0:
                file.write("Intersection over union = 0:\n")
                for sentence_iou in self.sentences_iou:
                    file.write(sentence_iou + '\n')

        return text_file_name




    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('file_path', nargs='?', help='File path for the annotation file')
        parser.add_argument('--showplot', action='store_true', help='Flag to indicate whether to display the plot the flight')
        parser.add_argument('--noplots', action='store_true', help='Flag to skip plot generation')
        args = parser.parse_args()
        self.args = parser.parse_args()# Store args for later reference


        if args.file_path:
            self.file_path = args.file_path
        else:
            self.filepath()#collect the file location 
            
        if self.validfile():#check if file is present
            self.processcsv()#changes csv to dataframe
            self.classerror()#identify class
            self.duplicates()#identify duplicates
            self.iou()#Checks movement of boxes between frames
            self.areaanalysis() #checks for extreme areas
            self.disappearingboxes()#Identify disappearing boxe
            if not args.noplots:
                self.plotsonarea()  # Plot graph
                self.individualplots()#plot graph of individual above area 4000
            self.dataframetocsv()#saves the edited data frame to csv
            self.textsave()#print error statements to text file

        else:
            print("Invalid file path or the file does not exist.")

if __name__ == "__main__":
    data_processor = DataProcessor()
    data_processor.main()


# In[ ]:




