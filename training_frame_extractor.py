import argparse
import os
import pandas as pd
import cv2
import random


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process .csv file, generate paths, and extract frames.")
    parser.add_argument("csv_file", default="test.csv", type=str, help="Path to the input .csv file")
    parser.add_argument("output_dir", default="D:\\mela\\dataDump\\test" ,type=str, help="Path to the output directory for storing frames")
    return parser.parse_args()


def read_csv_as_dataframe(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df


def generate_paths_with_extension(data_frame):
    paths = []
    for index, row in data_frame.iterrows():
        path_elements = [str(row[i]) for i in range(5)]
        path = "\\".join(path_elements) + ".MP4"
        paths.append(path)
    return paths


def check_file_existence(file_path):
    file_path = "Y:\\working\\rawdata\\Field_Recording_2023\\Original\\" + file_path
    return os.path.exists(file_path), file_path

def generate_well_distributed_frames(num_frames, total_frames):
    selected_frames = []

    # Select the first two frames within the range of 0 to 300
    selected_frames.extend(random.sample(range(0, min(301, total_frames)), 2))

    # Calculate the remaining frames with evenly distributed intervals
    remaining_frames = num_frames - len(selected_frames)
    interval = (total_frames - 301) // (remaining_frames - 1)
    for i in range(remaining_frames - 1):
        frame = min(300 + i * interval, total_frames - 1)
        selected_frames.append(frame)

    # Ensure the last frame is included
    selected_frames.append(total_frames - 1)

    return selected_frames


def extract_frames(video_path, num_frames, output_dir, file_name):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= num_frames:
        return None  # Not enough frames to extract

    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist.")
        return None

    # This logic will make sure we have two images
    selected_frames = generate_well_distributed_frames(num_frames, total_frames)

    print(len(selected_frames))
    for frame_number, idx in enumerate(selected_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_dir, file_name + f"frame_{idx}.jpg")
            print(frame_filename)
            cv2.imwrite(frame_filename, frame)
    cap.release()


if __name__ == "__main__":
   # args = parse_arguments()
   # csv_file_path = args.csv_file
   # output_dir = args.output_dir

    csv_file_path = "test.csv"
    output_dir = "D:\\mela\\dataDump"

    data_frame = read_csv_as_dataframe(csv_file_path)
    paths = generate_paths_with_extension(data_frame)

    print("Generated paths:")
    for idx, path in enumerate(paths):
        print(path)
        exists, videoPath = check_file_existence(path)
        file_name = path.replace("\\","_")
        filename_without_extension = os.path.splitext(file_name)[0]
        print(f"File exists at path: {exists}")
        if exists:
            num_frames = data_frame.iloc[idx, 5]  # Assuming column 6 contains num_frames values
            extract_frames(videoPath, num_frames, output_dir, filename_without_extension)
            print(f"Extracted and stored frames at {output_dir}.")
