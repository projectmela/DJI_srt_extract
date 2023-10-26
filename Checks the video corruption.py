#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Author: Dipin
#Checks for correpted video files


import os
import cv2
import pandas as pd



def main():
    # Ask the user for the root folder location
    root_folder = input("Enter the root folder location: ")
    #root_folder = "D:/MELA/Test Dataset video/"

    # Check if the provided folder exists
    if not os.path.exists(root_folder):
        print("The specified folder does not exist. Please provide a valid folder path.")
        return

    result_dataframe = check_playable_videos(root_folder)
    print(result_dataframe)



def check_playable_videos(root_folder):
    video_data = []

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path):
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)

                if os.path.isdir(subfolder_path):
                    for video_file in os.listdir(subfolder_path):
                        if video_file.lower().endswith(".mp4"):  # Only process mp4 files
                            video_path = os.path.join(subfolder_path, video_file)

                            # Initialize 'corrupted' flag as 0, assuming the video is playable
                            corrupted = 0

                            # Check if the video is playable using OpenCV
                            try:
                                cap = cv2.VideoCapture(video_path)
                                if not cap.isOpened():
                                    corrupted = 1
                            except Exception as e:
                                print(f"Error while reading video '{video_file}': {e}")
                                corrupted = 1
                            finally:
                                cap.release()

                            # Append the video data to the list
                            video_data.append({
                                'Video_ID': video_file,
                                'corrupted file': corrupted,
                                'drone id': subfolder_name,
                                'section': folder_name
                            })

    # Create a DataFrame from the video data
    df = pd.DataFrame(video_data)

    return df


if __name__ == "__main__":
    main()





# In[ ]:




