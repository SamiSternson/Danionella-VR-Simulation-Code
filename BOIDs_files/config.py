from unicodedata import name

import numpy as np
################################################################################################################################################
########################################## Main Program Variables ##########################################
# Once you identify the parameters you're interested in, I would reccomend moving these variables could be better organized in like a json file
################################################################################################################################################


Delay=0 # Habituation time (only relevant for real fish tracking experiments)
OffTime=1 # Time fish are hidden (s)
OnTime=300 # Time fish are shown (s)
NumTrials=3 # Number of trials to run
TestMode = True # If True, runs in test mode with different cam perspective etc. and no saving
SaveMode = False # If True, saves fish positions to a csv file
MonitorLength = 300 # Monitor length in mm (used for scaling if there is overhang)

# Panda3D parameters
CamDist=50 #distance of virtural camera to tank (game units) 
CamHeight=100 #height of virtual camera and tank (game units)  
FishScale=.5 #scale to apply to our firtual fish (note this is applied to all dimensions of the fish model, so it is not a linear scaling of length)
BackgroundColor=[0.94,1,1,1] #Background color of the game window (RGBA)
Fishalpha=1 #Transparency of the virtual fish (0=transparent, 1=opaque)


# Scale factors
fish_size = 12 #fish size in mm for model scaling
TankSize = 1000  # Tank size in mm for model scaling

# Schooling forces 
InteractionLimit = 1e10#50 # Distance threshold virtual fish engagement with eachother (mm)
InteractionLimitRealFish = 280 # Distance threshold virtual fish engagement with real fish (mm)
RealFishSchoolStrength = 0  # Relative strength of schooling forces to real fish versus virtual fish
ShowVirtualTankOutline = False
BURST_DIR = 0.01#0.1
GLIDE_DIR = 0.4#1-BURST_DIR
TARGET_VELOC = 5.5
FRICTION_COEFF = 0.5

# ArenaSize 
ArenaSize = [[35.75], [-3918.75], [572.2237324683415], [[129, 693]], [np.float64(117.0), np.float64(264.0)], [np.float64(125.0), np.float64(550.0)], [113, 121], [[113, 121], [686, 127], [672, 671], [129, 693]]]

# Pick side of stimulus presentation (left or right) randomly for each trial
stimulus_side = 'left'# random.choices(["left","right"],k=1)

if __name__ == "__main__":

    print("This is the config program. Launch from main.py to run the simulation.")
