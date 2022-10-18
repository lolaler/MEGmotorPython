"""
Created on Thu 25th of August

@author: lolaler

Easy to use script for plotting and reading & writing data from and to csv-files.

"""

from support_functions import *

# --------------------- säädä nämä mieleiseksi ---------------------

# choose the subject(s) whose data you want to work with. if many subjects, write it in the form subject = (1, 10)
subject = 7

# choose session, either one session or many, in which case  e.g. session = (1, 4)
session = 5

# choose data type, either "motor" or "evoked"
type = "motor" 

# do you want to plot the evoked data? 1 = yes, 0 = no
plot = 1

# do you want to read selected channels from a .csv file? 1 = yes, 0 = no
read_channels = 0

# define filename for the file from which we read (if none, write None). 
read_file  = "some csv-file"

# and the file into which we write
write_file = "output.csv"

# ------------------------ säädöt loppuu tähän -----------------------



# define column names (don't touch this)
column_names = ['subject', 'session', 'channel', 'evoked data array']

# don't touch this
main_function(subject, session, type, plot, read_channels, read_file, write_file, column_names)


