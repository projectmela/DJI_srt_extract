#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Author : Dipin
#Function : Renaming of srt and video files

import os

class FileRenamer:
    def __init__(self, root_folder):
        # Initialize the FileRenamer class with the root folder
        self.root_folder = root_folder
        self.files_found = False  # Flag to track if any files were found

    def rename_srt_files(self):
        srt_files_found = False  # Flag to track if any SRT files were found

        # Iterate through the directory structure to find SRT files
        for folder_name in os.listdir(self.root_folder):
            folder_path = os.path.join(self.root_folder, folder_name)

            if os.path.isdir(folder_path):
                for subfolder_name in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder_name)

                    if os.path.isdir(subfolder_path):
                        for sub_subfolder_name in os.listdir(subfolder_path):
                            sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder_name)

                            if os.path.isdir(sub_subfolder_path):
                                files = os.listdir(sub_subfolder_path)
                                for i, file in enumerate(files, start=1):
                                    file_name, file_ext = os.path.splitext(file)

                                    if file_ext.lower() == '.srt':
                                        srt_files_found = True
                                        self.files_found = True
                                        # Construct the new file name using f-string
                                        new_file_name = f"{folder_name}_{subfolder_name}_{sub_subfolder_name}_{file[-12:-4]}.srt"
                                        new_file_path = os.path.join(sub_subfolder_path, new_file_name)
                                        old_file_path = os.path.join(sub_subfolder_path, file)
                                        os.rename(old_file_path, new_file_path)

        if not srt_files_found:
            print("No SRT files found in the specified directory.")

    def rename_mp4_files(self):
        mp4_files_found = False  # Flag to track if any MP4 files were found

        # Iterate through the directory structure to find MP4 files
        for folder_name in os.listdir(self.root_folder):
            folder_path = os.path.join(self.root_folder, folder_name)

            if os.path.isdir(folder_path):
                for subfolder_name in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder_name)

                    if os.path.isdir(subfolder_path):
                        for sub_subfolder_name in os.listdir(subfolder_path):
                            sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder_name)

                            if os.path.isdir(sub_subfolder_path):
                                mp4_files = [f for f in os.listdir(sub_subfolder_path) if f.lower().endswith('.mp4')]
                                for i, mp4_file in enumerate(mp4_files, start=1):
                                    mp4_files_found = True
                                    self.files_found = True
                                    file_name, file_ext = os.path.splitext(mp4_file)
                                    new_file_name = f"{folder_name}_{subfolder_name}_{sub_subfolder_name}_{file_name}{file_ext}"
                                    new_file_path = os.path.join(sub_subfolder_path, new_file_name)
                                    old_file_path = os.path.join(sub_subfolder_path, mp4_file)
                                    os.rename(old_file_path, new_file_path)

        if not mp4_files_found:
            print("No MP4 files found in the specified directory.")

    def display_message(self, file_type):
        if not self.files_found:
            print(f"No {file_type} files found in the specified directory.")
        else:
            print(f"{file_type} files successfully renamed.")

def main():
    srt_root_folder = input("Enter the file location for SRT: ")

    # Check if the entered directory exists and is a valid directory
    if not os.path.exists(srt_root_folder) or not os.path.isdir(srt_root_folder):
        print(f"Warning: The specified directory '{srt_root_folder}' does not exist or is not a valid directory for SRT files.")
    else:
        srt_renamer = FileRenamer(srt_root_folder)
        srt_renamer.rename_srt_files()
        srt_renamer.display_message("SRT")

    video_root_folder = input("Enter the folder location of video files: ")

    # Check if the entered directory exists and is a valid directory
    if video_root_folder and (not os.path.exists(video_root_folder) or not os.path.isdir(video_root_folder)):
        print(f"Error: The specified directory '{video_root_folder}' does not exist or is not a valid directory for video files.")
        return

    mp4_renamer = FileRenamer(video_root_folder)
    mp4_renamer.rename_mp4_files()
    mp4_renamer.display_message("MP4")

if __name__ == "__main__":
    main()


# In[ ]:




