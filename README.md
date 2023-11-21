
# 1)Annotation analyzer and error corrector

These code helps in checking the validity and errors of the annotations done through the darklabel annotation software. Along with the error checking the code also provide automated correction of certain errors.

# 2)Srt analyzer of drone flight
This code helps in anlysing you drone session using the srt files created
## Documentation

### Annotation ##

The codes are used to detect and correct the errors that have been identified in the CSV files that have been manually annotated. One code is for the detection of the errors. The end product of running this code will be a text file with details of the error, a CSV file with some errors marked and graphs. These are explained below. The second code is for the automatic correction of some of the errors created during the annotation (duplicates, class errors) 

### srt analyzer: Drone data ##
This code analyses the srt files of the flight session and gives you output in a csv form which contains data on things like miss clicks, frame drops etc


# **Annotation Analysis**
# Duplicates error
This error creates 2 bounding boxes of the same ID in a single frame.

### Detecting Error:

The code goes through each frame and checks if the same ID and class is repeated twice. When found it creates 2 new columns in the CSV file named duplicates and duplicates_frame. The individual ID with duplicates is marked with 1 in the duplicates column (initially 0) and the respective frame of that individual is marked with 1 in the duplicate_frame (initially 0).

The text file created will give information about the frame and ID that shows the duplicate error.

***Frames with duplicates: 127***

***Frame 127 = (1:7), (0:42)*** {Here(1:7) means class =1 and id = 7}

(In some instances, you may find -1 in the class position (-1,6). This is not a category and is caused by the class error explained below. It is automatically solved with the correction code)


# Class error

At certain frames, the class of an individual changes to -1 which is not an assigned category. This only happens for a single frame and the tracking is continued in the next frame. As -1 is not assigned to a particular category, the boxes are shown with numbers.

**Code for detecting error:** The code goes through the **“class id”** column of the CSV and finds rows that have been marked with -1. Then it creates 2 new columns named **“classid_error”** and **“classid_error_frame”**. The individual that has -1 in the class is marked as 1 in the classid_error column (initially 0). The respective frame that shows this error is flagged as 1 in the classid_error_frame column (initially 0).

The text file produced will give the frame number that is showing this error and the individual IDs that have this error in that particular frame.

***Frames with classid errors: 744*** 

***Total no of boxes with classid error: 18***

***Frame 744 = (10), (17), (11), (8), (13), (6), (4), (0), (7), (3), (15), (12), (14), (16), (5), (1), (2), (9)***


# Disappearing individuals

Here, an individual disappears from a frame and reappears in a subsequent frame. This creates a break in the tracking of that individual.

**Code for detecting error:** The code checks if each individual present in a particular frame is present in the previous frame or not. It gives a statement of the individuals that are missing from the text file created.

***Disappearance Statements:*** 

***ID 31 (Class 0) disappeared in frame 1601 and reappeared in frame 1603***

***ID 3 (Class 0) disappeared in frame 1740 and reappeared in frame 1742***



# Intersection over union of zero

IOU values are found by comparing the bounding boxes of one individual in frame X with frame X-1. It would assist us in understanding how fast bounding boxes move between frames. Value 1 means that there is complete overlap and value 0 means there is no overlap.

It was seen that some individuals showed an IOU value of 0 when compared with the previous frame. This pointes out frames where the bounding boxes had no individuals present in them. The text file created in the analysis will provide details regarding the individual and the frame where the error was detected.

***Intersection over union = 0:*** 

***ID 13 (Class 0) has IOU value 0 between the frames 2611 and 2612*** 

***ID 45 (Class 0) has IOU value 0 between the frames 1788 and 1789*** 

***ID 4 (Class 1) has IOU value 0 between the frames 1810 and 1811***



# Area

The area of the bounding box is calculated by multiplying the width and height of each bounding box. Some annotations have bounding boxes of an area above 10000. You can adjust the area measurement to analyze the annotation files by providing external commands. If not given, the area threshold will be kept at 4000. 
(Type --help to see all the options)

The text file produced will tell you how many frames have bounding boxes of area above 4000 and how many individuals have shown boxes above 4000.

***Out of 3101 frames, no of frames with bounding box area greater than 4000: 773***                                          

***Out of 67 individuals, no of individuals with bounding box area greater than 4000: 3
ID 1 (Class 0) = 744 frames.
ID 4 (Class 0) = 24 frames.
ID 7 (Class 0) = 5 frames.***



# Plots
*Area scatter plot*: A scatter plot is created with area along X axis and number of boxes along Y axis. This will help in analyzing the general trend of the distribution and helps in identifying the outliers. The graph also contains mean and the 95th percentile value for the respective annotated file. This graph is created along with the text and csv file during the analysis.

*Time scale of individuals over 4000:* This graph shows how the area of a particular individual varies as the frame changes. These graphs are only created for individuals whose area has gone above the 4000(can be changed) mark at any stage of the annotation. It can be used as a reference point while correcting the area(manually). 

*Aspect ratio vs Area graph*: This graph is plotted to see the trend of how the aspect ratio of each bounding box is changing. These are plotted in order to find outliers in the annotation.


# **Annotation Error Correction**
This code can automatically correct some of the errors identified by the above code.
# Duplicates error
**Code for solving the error**: First it detects the individuals that are showing the error with the code above. Then it checks the distance between the 2 duplicates.

If the distance is above 100, it calculates the distance between each duplicate and the coordinates of the same individual in the previous frame. The duplicate which has the largest distance is deleted.

If the distance is less than 100, the area of each duplicate (calculated by multiplying width and height) is compared with the area of the same individual from the previous frame. The one that shows the least difference in area from the previous frame is kept and the other is deleted.

# Class error
**Code for solving the error**: Here we use the intersection over union concept to solve the problem. First, the rows that are showing class errors are identified using the code above. The IOU of each bounding box with the error is compared with every individual present in the previous frame. The bounding box that has the largest IOU value from the previous frame is selected. The class and ID of this bounding box (previous frame) are copied and pasted onto the class and ID columns of the individual that is being corrected. This process is repeated for all the bounding boxes that are showing an error.

# **SRT analyzer** 
This code helps in identifies the errors and other unique data from each drone session. It creates a csv file combing all the data. 
The information in the csv file are:
Flight number, Global start/end time, max/min height, drift status, height variation, frame drop, miss clicks, start/end frame, relay video
## Installation

These codes can be directly run using python once you have downloaded and installed the required files.


dependencies:
  - python=3.10.9 
  - numpy=1.23.5
  - pandas=1.5.3
  - matplotlib=3.7.2
  - geopy=2.3.0 
  - opencv=4.6.0  
```
The codes can take more aruguments when called. Type --help to see more options.
srtanalyzer can take
the location of the session
drift threshold value
height threshold value

Annotationanalysis can take
the location of the annotated file
area threshold value
command to not create the graphs
command to display graphs while running the code
## Appendix

A test dataset is provided to help you in checking if the codes are running as intended. The test dataset contains both the files to be run and their expected outputs.
