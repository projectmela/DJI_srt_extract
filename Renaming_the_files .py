#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Author : Dipin
#Function : Renaming of srt and video files


#Rename the srt files
import os

def main():
    srt_root_folder = input("Enter the file location for srt: ")
    #root_folder = "D:/MELA/AX/" 
    #AX > 20230308 
    rename_files(srt_root_folder)
    
    video_root_folder = input("Enter the folder location of video files: ")
    #root_folder = "D:/MELA/AX/" 
    # AX > 20230308
    rename_mp4_files(video_root_folder)

def rename_files(root_folder):
    if not os.path.exists(root_folder):
        print(f"Error: The specified directory '{root_folder}' does not exist.")
        return

    srt_files_found = False  # Flag to track if any SRT files were found

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

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
                                    # Construct the new file name using f-string
                                    new_file_name = f"{folder_name}_{subfolder_name}_{sub_subfolder_name}_{file[-12:-4]}.srt"
                                    new_file_path = os.path.join(sub_subfolder_path, new_file_name)
                                    old_file_path = os.path.join(sub_subfolder_path, file)
                                    os.rename(old_file_path, new_file_path)

    if not srt_files_found:
        print("No SRT files found in the specified directory.")

def rename_mp4_files(root_folder):
    if not os.path.exists(root_folder):
        print(f"Error: The specified directory '{root_folder}' does not exist.")
        return

    mp4_files_found = False  # Flag to track if any MP4 files were found

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

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
                                file_name, file_ext = os.path.splitext(mp4_file)
                                new_file_name = f"{folder_name}_{subfolder_name}_{sub_subfolder_name}_{file_name}{file_ext}"
                                new_file_path = os.path.join(sub_subfolder_path, new_file_name)
                                old_file_path = os.path.join(sub_subfolder_path, mp4_file)
                                os.rename(old_file_path, new_file_path)

    if not mp4_files_found:
        print("No MP4 files found in the specified directory.")
                                 
                                                            
                                    
if __name__ == "__main__":
    main()                                    

