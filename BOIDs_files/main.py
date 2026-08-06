# Burst and glide model with boid-like interactions for schooling fish for Sami Sternson to get started with
## written originally by Geoff Meyerhof
##July 1, 2025
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
import random
import math
import os
from direct.actor.Actor import Actor  # Import Actor for animations
import queue
import multiprocessing
from collections import deque
import numpy as np
import time
import cv2
import datetime as dt


################################################################################################################################################
##########################################Program launcher##########################################
# I would keep this file as a launcher and for keeping track of various threads or processes you want to add (for example, you probably want a separate process for saving simulation data)
################################################################################################################################################



if __name__ == "__main__":


    #### MP events (These are leftover from when I was running the simulation in a separate thread from the main program, but they are still useful for controlling the flow of the simulation)
    fish_showing_event = multiprocessing.Event()  # Indicates whether fish are currently shown
    start_saving = multiprocessing.Event()  # Indicates whether to start saving fish positions
    start_timer = multiprocessing.Event()  # Indicates whether to start the timer
    stop_event = multiprocessing.Event()  # Indicates whether to stop the simulation
    
    # Store in dictionary to be passed to the simulation class
    events = {}
    events['fish_showing'] = fish_showing_event
    events['start_saving'] = start_saving
    events['start_timer'] = start_timer
    events['stop_event'] = stop_event

    ################################################################################################################################################
    ######Put threads and processes here if you want to add any (for example, a separate process for saving simulation data)#########
    ################################################################################################################################################




    ########################################################################
    # start Panda3D loop last to avoid conflicts (note, panda3d is blocking and will run in perpetuity unless you close the window, so make sure to have a way to exit the program)
    # (This sets size of game window and its position on your monitor)
    loadPrcFileData('', 'win-size 1920 600')
    loadPrcFileData("", "win-origin 0 100")  # Move window to (0, 0)
    loadPrcFileData('', 'show-frame-rate-meter 1')


    # Run the simulation
    from FishSchoolSimulation import *
    import config

    VT = VirtualFishTank(config,events)
    VT.run()
    ########################################################################