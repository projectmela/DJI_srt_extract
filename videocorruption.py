#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Author: Dipin
#Checks for correpted video files

import os
import cv2
import pandas as pd
import argparse

class VideoChecker:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def check_playable_videos(self):
        video_data = []

        for folder_name in os.listdir(self.root_folder):
            folder_path = os.path.join(self.root_folder, folder_name)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_folder', nargs='?', help='Root folder location')
    args = parser.parse_args()

    # If root folder is not provided as an argument, ask the user for input
    if not args.root_folder:
        args.root_folder = input("Enter the root folder location: ")

    # Check if the provided folder exists
    if not os.path.exists(args.root_folder):
        print("The specified folder does not exist. Please provide a valid folder path.")
        return

    video_checker = VideoChecker(args.root_folder)
    result_dataframe = video_checker.check_playable_videos()
    print(result_dataframe)

if __name__ == "__main__":
    main()





# In[ ]:




