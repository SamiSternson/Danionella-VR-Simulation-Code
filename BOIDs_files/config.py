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
TestMode = False# If True, runs in test mode with different cam perspective etc. and no saving
SaveMode = False # If True, saves fish positions to a csv file
MonitorLength = 300 # Monitor length in mm (used for scaling if there is overhang)

# Panda3D parameters
CamDist=50 #distance of virtural camera to tank (game units) 
CamHeight=0 #height of virtual camera and tank (game units)  
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
GLIDE_FREQUENCIES=[np.float64(0.0377634660421546), np.float64(0.1656908665105388), np.float64(0.14139344262295095), np.float64(0.10772833723653406), np.float64(0.12558548009367693), np.float64(0.10245901639344272), np.float64(0.07201405152224831), np.float64(0.0644028103044497), np.float64(0.04771662763466047), np.float64(0.030152224824356), np.float64(0.02459016393442625), np.float64(0.020199063231850133), np.float64(0.015222482435597201), np.float64(0.013466042154566754), np.float64(0.008196721311475414), np.float64(0.006440281030444967), np.float64(0.003512880562060889), np.float64(0.00468384074941452), np.float64(0.002049180327868852), np.float64(0.0014637002341920374), np.float64(0.00117096018735363), np.float64(0.0017564402810304447), np.float64(0.0008782201405152225), np.float64(0.0002927400468384075), np.float64(0.000585480093676815), np.float64(0.0), np.float64(0.0002927400468384075), np.float64(0.0), np.float64(0.0), np.float64(0.0002927400468384075)] 
GLIDE_BINS=[np.float64(0.04969753271803315), np.float64(0.08123029003130065), np.float64(0.11276304734456816), np.float64(0.14429580465783565), np.float64(0.17582856197110314), np.float64(0.20736131928437065), np.float64(0.23889407659763814), np.float64(0.2704268339109056), np.float64(0.30195959122417315), np.float64(0.3334923485374406), np.float64(0.36502510585070813), np.float64(0.3965578631639757), np.float64(0.4280906204772431), np.float64(0.45962337779051066), np.float64(0.4911561351037781), np.float64(0.5226888924170455), np.float64(0.5542216497303131), np.float64(0.5857544070435806), np.float64(0.6172871643568482), np.float64(0.6488199216701155), np.float64(0.680352678983383), np.float64(0.7118854362966506), np.float64(0.7434181936099181), np.float64(0.7749509509231857), np.float64(0.806483708236453), np.float64(0.8380164655497205), np.float64(0.8695492228629881), np.float64(0.9010819801762556), np.float64(0.932614737489523), np.float64(0.9641474948027904)]
BURST_BINS= [np.float64(0.010695111300471077), np.float64(0.021390222600942154), np.float64(0.03208533390141323), np.float64(0.04278044520188431), np.float64(0.05347555650235539), np.float64(0.06417066780282646), np.float64(0.07486577910329754), np.float64(0.08556089040376862), np.float64(0.09625600170423969), np.float64(0.10695111300471077), np.float64(0.11764622430518185), np.float64(0.12834133560565292), np.float64(0.139036446906124), np.float64(0.1497315582065951), np.float64(0.16042666950706616), np.float64(0.17112178080753723), np.float64(0.1818168921080083), np.float64(0.19251200340847938), np.float64(0.20320711470895048), np.float64(0.21390222600942155), np.float64(0.22459733730989262), np.float64(0.2352924486103637), np.float64(0.24598755991083476), np.float64(0.25668267121130584), np.float64(0.2673777825117769), np.float64(0.278072893812248), np.float64(0.2887680051127191), np.float64(0.2994631164131902), np.float64(0.31015822771366125), np.float64(0.3208533390141323)]
BURST_FREQUENCIES=[np.float64(0.0024775472282440383), np.float64(0.0003096934035305048), np.float64(0.0003096934035305048), np.float64(0.0018581604211830288), np.float64(0.013007122948281189), np.float64(0.043976463301331724), np.float64(0.06968101579436367), np.float64(0.0919789408485597), np.float64(0.0864044595850107), np.float64(0.08330752554970569), np.float64(0.07928151130380919), np.float64(0.06410653453081466), np.float64(0.05481573242489948), np.float64(0.0597708268813876), np.float64(0.052028491793124916), np.float64(0.04428615670486223), np.float64(0.04087952926602665), np.float64(0.03282750077423346), np.float64(0.03220811396717244), np.float64(0.024775472282440338), np.float64(0.018891297615360764), np.float64(0.020439764633013283), np.float64(0.01548467017652522), np.float64(0.01610405698358623), np.float64(0.009290802105915142), np.float64(0.00836172189532363), np.float64(0.00836172189532363), np.float64(0.012078042737689677), np.float64(0.00712294828120161), np.float64(0.005574481263549086)] 
TARGET_VELOC = 5.5
FRICTION_COEFF = 0.5

# ArenaSize 
ArenaSize = [[35.75], [-3918.75], [572.2237324683415], [[129, 693]], [np.float64(117.0), np.float64(264.0)], [np.float64(125.0), np.float64(550.0)], [113, 121], [[113, 121], [686, 127], [672, 671], [129, 693]]]

# Pick side of stimulus presentation (left or right) randomly for each trial
stimulus_side = 'left'# random.choices(["left","right"],k=1)

if __name__ == "__main__":

    print("This is the config program. Launch from main.py to run the simulation.")
