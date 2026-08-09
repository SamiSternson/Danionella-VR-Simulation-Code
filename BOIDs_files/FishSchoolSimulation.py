import multiprocessing

from panda3d.core import *
from direct.actor.Actor import Actor  # Import Actor for animations
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
import math 
import numpy as np
import random
import datetime as dt
import time

def inverse_power_distribution_index(n, p):
    probs = [(1 / i**p) for i in range(1, n + 1)]
    total = sum(probs)
    new_probs = [prob / total for prob in probs]
    return np.random.choice([i for i in range(n)], p=new_probs)


class VirtualFishTank(ShowBase):
    def __init__(self,config,events):
        super().__init__()
        

        # Unpack events 
        self.stop_event = events['stop_event']
        self.start_timer = events['start_timer']
        self.start_saving = events['start_saving']
        self.fish_showing = events['fish_showing']

        
        # Experiment parameters
        self.config = config
        self.trial_count = 0
        self.num_trials = self.config.NumTrials
        self.Delay = self.config.Delay+np.inf # avoid prematurely starting trial until fish are initialized
        self.OffTime = self.config.OffTime
        self.OnTime = self.config.OnTime
        self.fish_show_time=0
        self.OverhangRatio = self.config.MonitorLength/self.config.TankSize
        self.BURST_DIR = self.config.BURST_DIR
        self.GLIDE_DIR = self.config.GLIDE_DIR
        # Assign random location of virtual fish for each trial 
        self.tank_side_order = [self.config.stimulus_side] * self.num_trials
        # Accept the "escape" key to stop the game early
        #self.taskMgr.add(self.stop_game, "stopGameTask")
        self.accept("q", self.stop_game) # Quit the game
        self.accept("p",self.pause_game) # Pause the game
        self.set_background_color(self.config.BackgroundColor[0], self.config.BackgroundColor[1], self.config.BackgroundColor[2], self.config.BackgroundColor[3])
        self.disable_mouse()

        # Set up the side-scrolling camera
        self.camera.set_pos(0, -self.config.CamDist, self.config.CamHeight)
        self.camera.look_at(0, 0, self.config.CamHeight)

        # Add ambient light for uniform illumination
        ambient_light = AmbientLight("ambient_light")
        ambient_light.set_color((0.9, 0.9, 0.9, 1))  # Dimmer ambient light
        ambient_light_node = self.render.attach_new_node(ambient_light)
        self.render.set_light(ambient_light_node)

        directional_light = DirectionalLight("directional_light")
        directional_light.set_color((0.6, 0.6, 0.6, 1))  # Softer directional light
        directional_light_node = self.render.attach_new_node(directional_light)
        directional_light_node.set_hpr(45, -45, 0)  # Direction of light
        self.render.set_light(directional_light_node)
        
        # self.fish_model_paths = [r"fish_models\DC_blender_final_rigged.bam",
        #                          r"fish_models\DC_female_blender_final_rigged.bam"]

        self.fish_model_paths = [r"..\fish_models\DC_blender_final_rigged.bam"]                
        self.real_fish_path = r"..\fish_models\DC_blender_final_rigged.bam"

        #Terrain
        self.terrain = self.loader.load_model(r"..\terrain\uploads_files_2708212_terrain.fbx")
        self.terrain.reparent_to(self.render)
        self.terrain.set_scale(2, 0.7, 4)  # Adjust the scale as needed
        self.terrain.set_r(0)
        self.terrain.set_h(-270)
        self.terrain.set_p(90)

        self.terrain.set_pos(0, 0, -50)  # Position the terrain to act as the floor
        terrain_texture = self.loader.load_texture(r"..\terrain\aerial_grass_rock_diff_1k.jpg")
        self.terrain.set_texture(terrain_texture, 1)
        # Fish variables
        self.num_fish = 5
        self.fish_list = []
        self.fish_state = {}  # Dictionary to store the state of each fish
        # Pause variable
        self.paused = False
        ###Make all this shit a function 
        self.arena_rotation_rad = math.atan2(self.config.ArenaSize[7][1][1]-self.config.ArenaSize[7][0][1],self.config.ArenaSize[7][1][0]-self.config.ArenaSize[7][0][0])
        self.tank_rot_mat = [[math.cos(-self.arena_rotation_rad), -math.sin(-self.arena_rotation_rad)],
                            [math.sin(-self.arena_rotation_rad),  math.cos(-self.arena_rotation_rad)]]
        self.tank_origin=(int(self.config.ArenaSize[7][0][0]), int(self.config.ArenaSize[7][0][1]))
        

        print('Arena rotation (deg): ',np.rad2deg(self.arena_rotation_rad))
        # Calculate virtual tank coordinates
        self.get_tank_coordinates(sanity_check=self.config.ShowVirtualTankOutline,relative_depth=True,tank_height_percent=40,draw_normals=False)
        self.init_fish()
        # Task to update fish movements
        self.taskMgr.add(self.update_fish, "UpdateFishTask")
        # Task to update real fish movements 
        # self.taskMgr.add(self.update_real_fish,"UpdateRealFishTask") # Not needed for simulations 
        self.taskMgr.doMethodLater(0.1, self.check_time, "CheckTimeTask")  # Reduced frequency
        # Add saving tasks here 
        if self.config.SaveMode:
            self.fish_pos_path = self.config.ExperimentDirectory + r'/'+ self.config.ExperimentName + "_fish_positions.csv" 
            self.FirstSave = True
            self.task_mgr.add(self.save_metadata_once, "SaveMetadataOnceTask")
        if self.config.TestMode:
            self.camera_offset(FixedCameraAngle=0,cam_height_offset=0)
            print('Camera offset to fixed angle for testing')
            #self.taskMgr.add(self.camera_orbit, "SpinCameraTask")

    def get_tank_coordinates(self, sanity_check=False, relative_depth=False,tank_height_percent=30,draw_normals=False):
        """
        Compute and store tank boundaries and corner vertices relative to the camera's FOV.
        (breaks with camera roll/pitch/yaw.)
        Tank height percent is how tall the "virtual tank" will be on the window that's created 
        """
        tank_height_percent = 100 - tank_height_percent
        tank_height_prop = (tank_height_percent%100)/100
        #1.) Get camera fov
        fov_x, fov_y = self.camLens.get_fov()
        cam_pos = self.camera.get_pos()

        #2.) Get fish size 
        fish_path = self.fish_model_paths[0]
        fish = Actor(fish_path, {"swim": fish_path})
        fish.set_scale(self.config.FishScale)
        fish_depth, fish_width, fish_height = self.get_model_dimensions(fish)
        tank_depth = 8 * fish_depth * 10
        #3.) Get geom relative to camera 
        half_width_front  = math.tan(math.radians(fov_x / 2)) * abs(self.config.CamDist) 
        half_height_front = math.tan(math.radians(fov_y / 2)) * abs(self.config.CamDist) 

        if relative_depth:
            half_width_back  = math.tan(math.radians(fov_x / 2)) * abs(self.config.CamDist + tank_depth) 
            half_height_back = math.tan(math.radians(fov_y / 2)) * abs(self.config.CamDist + tank_depth)
        else:
            half_width_back  = half_width_front 
            half_height_back = half_height_front

        #4.) Get vertices for left tank 
        # 1.) Get offset related to overhang of tank (assumes top vertices of monitor aligned with tank)
        # Find overhang theta 
        opp=self.config.MonitorLength-self.config.TankSize
        CamDistWorldUnits = self.config.CamDist * 6.6 # 6.6 scalar empirically determined 
        adj=CamDistWorldUnits 
        overhang_theta = math.atan2(opp,adj) 
        y_front = 0
        y_back  = tank_depth
        z_bottom_front = cam_pos.z - half_height_front
        z_top_front    = cam_pos.z - half_height_front*tank_height_prop #cam_pos.z + half_height_front (uncomment to get full range of screen)
        z_bottom_back  = cam_pos.z - half_height_back
        z_top_back     = cam_pos.z - half_height_back*tank_height_prop # cam_pos.z + half_height_back (uncomment to get full range of screen)
        x_left_front   = cam_pos.x + math.sin(overhang_theta)*self.config.CamDist
        x_right_front  = cam_pos.x + half_width_front
        x_left_back    = cam_pos.x + math.sin(overhang_theta)*(self.config.CamDist+tank_depth)
        x_right_back   = cam_pos.x + half_width_back

        # Define the 8 corner vertices of the tank
        # Code order: LeftFrontBottom, LeftFrontTop, LeftBackBottom, LeftBackTop,...
        LFB = Point3(x_left_front, y_front, z_bottom_front)
        LFT = Point3(x_left_front, y_front, z_top_front)
        LBB = Point3(x_left_back,  y_back,  z_bottom_back)
        LBT = Point3(x_left_back,  y_back,  z_top_back)
        RFB = Point3(x_right_front, y_front, z_bottom_front)
        RFT = Point3(x_right_front, y_front, z_top_front)
        RBB = Point3(x_right_back,  y_back,  z_bottom_back)
        RBT = Point3(x_right_back,  y_back,  z_top_back)

        self.tank_vertices_left = [LFB, LFT, LBB, LBT, RFB, RFT, RBB, RBT]
        self.tank_bounding_box_left = BoundingBox()
        for vertex in self.tank_vertices_left:
            self.tank_bounding_box_left.extend_by(vertex)

        self.compute_tank_planes_left(draw_normals=draw_normals)
        
        # Sanity check visualization and prints screen coordinates 
        if sanity_check:
            self.show_bbox(vertices=self.tank_vertices_left)
            for vert in self.tank_vertices_left:
                smiley = self.loader.load_model("smiley")
                smiley.reparent_to(self.render)
                smiley.set_pos(vert)
                smiley.set_h(90)
                # print screen coordinates of each smiley
                #pixel_x,pixel_y = self.get_screen_coordinates(smiley)
                pixel_x,pixel_y = self.get_screen_coordinates_smiley(smiley)
                print(f'Pixel x {pixel_x}. Pixel y {pixel_y}')
                smiley.set_color(random.random(), random.random(), random.random(), 1)
        
        # Get scale from game units to mm based on tank size (assumes square tank base)
        self.tank_base_length = (RFB - LFB).length()
        self.tank_min_x = LFB.x
        self.tank_min_y = RFB.y
        self.tank_max_x = RBB.x
        self.tank_max_y = RBB.y
        tank_depth_length_y = abs(self.tank_max_y - self.tank_min_y)
        # Get fron versus back ratio for depth scaling
        self.tank_front_width = (RFB - LFB).length()
        self.tank_back_width = (RBB - LBB).length()
        self.tank_depth = tank_depth_length_y

        self.unit_to_mm_scale = self.config.TankSize / self.tank_base_length
        self.mm_to_unit_scale = self.tank_base_length / self.config.TankSize 
        real_tank_base_length = self.config.TankSize * self.mm_to_unit_scale

        TankSize_pixels = Point2(self.config.ArenaSize[7][1][0]-self.config.ArenaSize[7][0][0],self.config.ArenaSize[7][1][1]-self.config.ArenaSize[7][0][1]).length()
        self.pixel_to_unit_scale = self.tank_base_length / TankSize_pixels
        #Next, we're going to simulate a second tank next to our virtual tank. This is where the camera fish will live. And we can verify sizes 
        # RFT = "real fish tank" (tank to mimic real world dimensions)
        LFB_rft = Point3(x_left_front, y_front, z_bottom_front)
        LFT_rft = Point3(x_left_front, y_front, z_top_front)
        LBB_rft = Point3(x_left_back,  y_back-(real_tank_base_length+tank_depth_length_y),  z_bottom_back)
        LBT_rft = Point3(x_left_back,  y_back-(real_tank_base_length+tank_depth_length_y),  z_top_back)
        RFB_rft = Point3(x_right_front, y_front, z_bottom_front)
        RFT_rft = Point3(x_right_front, y_front, z_top_front)
        RBB_rft = Point3(x_right_front,  y_back-(real_tank_base_length+tank_depth_length_y),  z_bottom_back)
        RBT_rft = Point3(x_right_front,  y_back-(real_tank_base_length+tank_depth_length_y),  z_top_back)

        self.tank_vertices_rft = [LFB_rft, LFT_rft, LBB_rft, LBT_rft, RFB_rft, RFT_rft, RBB_rft, RBT_rft]
        if sanity_check and self.config.TestMode:      
            self.show_bbox(vertices=self.tank_vertices_rft)
        
        ## Lastly, let's repeat tank dimension calculations for right camera tank 
        opp=self.config.MonitorLength-self.config.TankSize
        CamDistWorldUnits = self.config.CamDist * 6.6 # 6.6 scalar empirically determined 
        adj=CamDistWorldUnits 
        overhang_theta = math.atan2(opp,adj) 
        y_front = 0
        y_back  = tank_depth
        z_bottom_front = cam_pos.z - half_height_front
        z_top_front    = cam_pos.z - half_height_front*tank_height_prop #cam_pos.z + half_height_front (uncomment to get full range of screen)
        z_bottom_back  = cam_pos.z - half_height_back
        z_top_back     = cam_pos.z - half_height_back*tank_height_prop # cam_pos.z + half_height_back (uncomment to get full range of screen)
        x_left_front   = cam_pos.x - math.sin(overhang_theta)*self.config.CamDist
        x_right_front  = cam_pos.x - half_width_front
        x_left_back    = cam_pos.x - math.sin(overhang_theta)*(self.config.CamDist+tank_depth)
        x_right_back   = cam_pos.x - half_width_back

        # Define the 8 corner vertices of the tank
        # Code order: LeftFrontBottom, LeftFrontTop, LeftBackBottom, LeftBackTop,...
        LFB = Point3(x_left_front, y_front, z_bottom_front)
        LFT = Point3(x_left_front, y_front, z_top_front)
        LBB = Point3(x_left_back,  y_back,  z_bottom_back)
        LBT = Point3(x_left_back,  y_back,  z_top_back)
        RFB = Point3(x_right_front, y_front, z_bottom_front)
        RFT = Point3(x_right_front, y_front, z_top_front)
        RBB = Point3(x_right_back,  y_back,  z_bottom_back)
        RBT = Point3(x_right_back,  y_back,  z_top_back)

        self.tank_vertices_right = [LFB, LFT, LBB, LBT, RFB, RFT, RBB, RBT]
        self.tank_bounding_box_right = BoundingBox()
        for vertex in self.tank_vertices_right:
            self.tank_bounding_box_right.extend_by(vertex)

        self.compute_tank_planes_right(draw_normals=draw_normals)
        
        # Sanity check visualization and prints screen coordinates 
        if sanity_check:
            self.show_bbox(vertices=self.tank_vertices_right)
            for vert in self.tank_vertices_right:
                smiley = self.loader.load_model("smiley")
                smiley.reparent_to(self.render)
                smiley.set_pos(vert)
                smiley.set_h(90)
                # print screen coordinates of each smiley
                #pixel_x,pixel_y = self.get_screen_coordinates(smiley)
                pixel_x,pixel_y = self.get_screen_coordinates_smiley(smiley)
                print(f'Pixel x {pixel_x}. Pixel y {pixel_y}')
                smiley.set_color(random.random(), random.random(), random.random(), 1)
        
    def compute_tank_planes_left(self,draw_normals=True):
        v = self.tank_vertices_left
        # unpack for clarity
        LFB, LFT, LBB, LBT, RFB, RFT, RBB, RBT = v
        # Normals are pointing outward 
        self.faces_left = {
            "left":   (LFB, LFT, LBB),
            "right":  (RFB, RBB, RFT),
            "front":  (LFB, RFB, LFT),
            "back":   (LBB, LBT, RBB),
            "bottom": (LFB, LBB, RFB),
            "top":    (LFT, RFT, LBT),
        }
        self.tank_planes_left = {}
        for name, (p1, p2, p3) in self.faces_left.items():
            normal = (p2 - p1).cross(p3 - p1)
            normal.normalize()
            d = -normal.dot(p1)
            print(f"{name} face, D = ",d," normal is ",normal)
            self.tank_planes_left[name] = (normal, d)   
        if draw_normals:
            counter = 0
            for name, (normal, d) in self.tank_planes_left.items(): 
                values = list(self.faces_left.values())  # Convert values view to a list
                first_value = (values[counter][0] + values[counter][1] + values[counter][2]) / 3 
                print('first value ',values[counter])

                # scale normal and translate 
                line_start = normal + first_value
                line_end = (normal * 10) + first_value
                
                # Create a LineSegs object
                ls = LineSegs()
                ls.setThickness(5.0)  # Set line thickness
                ls.setColor(VBase4(0, 1, 1, 1)) # Set line color (red)

                # Start a new line segment
                ls.moveTo(line_start)
                # Add points to the current line segment
                ls.drawTo(line_end)

                # Create a NodePath from the LineSegs geometry and attach it to render
                line_node = self.render.attachNewNode(ls.create())
                np = NodePath(line_node)
                np.reparentTo(self.render)
                counter+=1
    
    def compute_tank_planes_right(self,draw_normals=True):
        v = self.tank_vertices_right
        # unpack for clarity
        LFB, LFT, LBB, LBT, RFB, RFT, RBB, RBT = v
        # Normals are pointing outward 
        self.faces_right = {
            "left":   (LFB, LFT, LBB),
            "right":  (RFB, RBB, RFT),
            "front":  (LFB, RFB, LFT),
            "back":   (LBB, LBT, RBB),
            "bottom": (LFB, LBB, RFB),
            "top":    (LFT, RFT, LBT),
        }
        self.tank_planes_right = {}
        for name, (p1, p2, p3) in self.faces_right.items():
            normal = (p3 - p1).cross(p2 - p1)
            normal.normalize()
            d = -normal.dot(p1)
            print(f"{name} face, D = ",d," normal is ",normal)
            self.tank_planes_right[name] = (normal, d)   
        if draw_normals:
            counter = 0
            for name, (normal, d) in self.tank_planes_right.items(): 
                values = list(self.faces_right.values())  # Convert values view to a list
                first_value = (values[counter][0] + values[counter][1] + values[counter][2]) / 3 
                print('first value ',values[counter])

                # scale normal and translate 
                line_start = normal + first_value
                line_end = (normal * 10) + first_value
                
                # Create a LineSegs object
                ls = LineSegs()
                ls.setThickness(5.0)  # Set line thickness
                ls.setColor(VBase4(0, 1, 1, 1)) # Set line color (red)

                # Start a new line segment
                ls.moveTo(line_start)
                # Add points to the current line segment
                ls.drawTo(line_end)

                # Create a NodePath from the LineSegs geometry and attach it to render
                line_node = self.render.attachNewNode(ls.create())
                np = NodePath(line_node)
                np.reparentTo(self.render)
                counter+=1

    ###HORIZONTAL FISH#####################
    def init_fish(self):
        # Load the fish model and set its properties
        for i in range(self.num_fish):
            
            fish_path = random.choice(self.fish_model_paths)
            fish = Actor(fish_path, {"swim": fish_path})
            fish.reparent_to(self.render)
            fish.set_p(0)  
            fish.set_h(90)  # Rotate fish to be parallel with the top of the screen
            fish.set_attrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
            fish.set_scale(self.config.FishScale,self.config.FishScale,self.config.FishScale)  # Adjust the size of the fish
            # Set transparency and adjust alpha value
            fish.set_transparency(TransparencyAttrib.M_alpha)

            fish.set_color_scale(self.config.fish_size/12, self.config.fish_size/12, self.config.fish_size/12, self.config.Fishalpha)  # Set alpha to 0.5 for 50% transparency
            
            # Calculate the horizontal bounds based on the camera depth and generate random fish positions 
            fish_depth, fish_width, fish_height = self.get_model_dimensions(fish)
            print('fish width: ',fish_height,' interaction scale: ',fish_width * 4.16)
            # Place fish randomly in tank
            print(self.config.stimulus_side)
            point = self.random_point_in_tank(fish_width=fish_width,tank=self.config.stimulus_side,fish_height=fish_height)
            fish.set_pos(point)

            # Store the state of each fish in the dictionary
            self.fish_state[fish] = {
                # Create path vectors to keep fish from drifting in Y direction
                "fish_path_radius": fish_width, 
                "fish_target": LVector3(0,0,0), # To start, lets just pick the center of a wall as a target 
                "running_velocities": [],
                "fish_width":fish_width,
                "fish_height":fish_height,
                "velocity": LVector3(1, 0, 0),  # Random velocity for each fish
                "target_veloc": self.config.TARGET_VELOC,  # Target mean velocity in units/sec
                "direction": 1,  # Randomly choose direction: -1 for left, 1 for right
                "acceleration": LVector3(0, 0, 0),  # Acceleration for speed changes
                "force": LVector3(0, 0, 0),  # Force applied to the fish, either for bursts or friction
                "friction_coeff":self.config.FRICTION_COEFF,  # Coefficient for friction force
                "average_veloc": [],
                "Glide": False,
                "burst_timer":0,
                "burst_duration": self.config.BURST_DIR,#self.get_burst_duration(),# np.random.lognormal(mean=0.01, sigma=0.01), # np.abs(random.gauss(0,0.05)) <- Deep water , # np.random.lognormal(mean=0.12, sigma=0.5) # Palka model match
                "glide_timer":0,
                "glide_duration": self.config.GLIDE_DIR,#self.get_glide_duration(), # np.abs(random.gauss(.35,0.25)) <- Deep water   # np.abs(random.gauss(0.17, 0.45)) 
                "fish_paused":False,
                "pause_timer":0,
                "start_time":0,
                #Palka model match
                "mass":1,
                "direction":1,
                "burst_counter": 0 ,

                # Define tank side # 
                "tank_side": self.config.stimulus_side,

                # Burst force # 
                "max_burst_force": None,
                
                # Define relative contribution of forces *Note these will scale with burst force 
                "boundary_rel_s": 0.278,
                "boundary_rel_m": 1.86,
                "righting_s": 0.056,
                "righting_m": 0.189,
                "avoidance_s": 0.5,
                "avoidance_m": 1.19,
                "cohesion_s":  0.05,#0.006, # 0.006 #strong schooling 0.1
                "cohesion_m": 0.150,
                "alignment_s": 0.2, #  0.083 #strong schooling 0.4
                "alignment_m": 0.5, # 0.187 #strong schooling 0.5
                "random_index":0,#Random index for deciding which fish to be attracted to and align with
                # Boundary force #
                "boundary_radius":fish_width * 5,

                # Righting force # 
                "righting_threshold": 0.15,
                
                # Avoidance #
                "avoidance_radius": self.config.fish_size * self.mm_to_unit_scale*10,

                # Cohesion # 
                "cohesion_radius": self.config.InteractionLimit * self.mm_to_unit_scale*10,

                # Alignment # 
                "alignment_radius":self.config.InteractionLimit * self.mm_to_unit_scale*10,
            }
            self.fish_list.append(fish)
        # Calculate initial burst strength for each fish based on target velocity
        for fish in self.fish_list:
            state = self.fish_state[fish]
            Tb = state['burst_duration']
            Tg = state['glide_duration']
            Cd = state['friction_coeff']
            v_mean = state['target_veloc']
            peak_v = self.peak_velocity_for_mean(v_mean=v_mean,Cd=Cd,burst_duration=Tb,glide_duration=Tg)
            #print('V PEAK ', peak_v)
            # clamp peak velocity to reasonable max
            peak_v = min(peak_v, 17)

            # Calculate required burst force to reach peak velocity
            burst_strength = (peak_v) / Tb
            state['max_velocity'] = peak_v * 50  # Allow some margin above calculated peak velocity
            state['burst_strength'] = burst_strength
            #print('First burst strength ',burst_strength)
            state['max_burst_force'] = burst_strength * 100  # Allow some margin above calculated burst strength
            # Update forces based on burst strength 
            self.update_non_burst_forces(state=state)
            # ADD NOISE TO FIRST BURST GLIDE CYCLE SO FISH ARENT IN TOTAL SYNCHRONY
            state['burst_duration'] = np.random.uniform(0,state['burst_duration']*2)
            state['glide_duration'] = np.random.uniform(0,state['glide_duration']*2)
        # Initialize the "real fish." I.e., the fish being tracked by the camera
        self.real_fish = Actor(self.real_fish_path, {"swim": self.real_fish_path})
        if self.config.TestMode:
            self.real_fish.reparent_to(self.render)
        # Set color scale of real fish to turn it black for visibility
        self.real_fish.set_color_scale(0, 0, 0, 1)  # Black color
        # Set real fish scale 
        self.real_fish.set_scale(self.config.FishScale,self.config.FishScale,self.config.FishScale)  # Adjust the size of the fish
        # Position real fish in the center of the "real fish tank" area
        center_rft = (self.tank_vertices_rft[0] + self.tank_vertices_rft[6]) / 2
        self.real_fish.set_pos(center_rft)

        # Ensure real_fish_velocity always exists, even when live tracking (update_real_fish)
        # isn't running. Without this, alignment_force can crash with AttributeError the
        # moment it randomly picks the real fish as the alignment neighbor. Only set a
        # default if update_real_fish hasn't already populated a live value.
        if not hasattr(self, 'real_fish_velocity'):
            self.real_fish_velocity = LVector3(0, 0, 0)

        # Once done initing fish, start the timer 
        self.fish_show_time=globalClock.get_frame_time()
        if self.trial_count==0:
            self.Delay=self.config.Delay
            self.start_timer.set()
        # Interaction limit 
        #print('Interaction limit (units): ',self.config.InteractionLimit * self.mm_to_unit_scale)
    
    def update_non_burst_forces(self,state):
        burst_strength = state['burst_strength']
        # Define relative contribution of forces *Note these will scale with burst force 

        # Boundary force #
        state["max_boundary_force"] = max(250,state[ "boundary_rel_m"] * burst_strength)
        state["boundary_strength"] = max(41,state[ "boundary_rel_s"] * burst_strength)
        print('Max boundary force ',state["max_boundary_force"],' boundary strength ',state["boundary_strength"])
        # Righting force # 
        state["max_righting_force"] =state[ "righting_m"] * burst_strength
        state["righting_strength"] =state[ "righting_s"] * burst_strength
        
        # Avoidance #
        state["max_avoidance_force"] =state[ "avoidance_m"] * burst_strength
        state["avoidance_strength"] =state[ "avoidance_s"] * burst_strength

        # Cohesion # 
        state["max_cohesion_force"] = state[ "cohesion_m"] * burst_strength
        state["cohesion_strength"] =state[ "cohesion_s"] * burst_strength

        # Alignment # 
        state["max_alignment_force"] = state[ "alignment_m"] * burst_strength
        state["alignment_strength"]= state[ "alignment_s"] * burst_strength
        

    
    def boundary_force(self, fish, state, threshold=10.0, strength=1.0):
        """
        Smooth repulsion from tank walls.
        Fish are pushed away more strongly as they get closer to any wall.
        """
        eminent_collision = False
        pos = fish.get_pos()
        force = LVector3(0, 0, 0)
        tank_planes = self.tank_planes_left if state['tank_side'] == 'left' else self.tank_planes_right
        for name, (normal, d) in tank_planes.items():
            signed_dist = normal.dot(pos) + d  # positive = outside, negative = inside
            if abs(signed_dist)<state['fish_width']:
                eminent_collision = True
            if signed_dist > -threshold:  # near the wall (inside)
                # Linear increase when approaching wall
                force += -normal * strength * math.log((2**(signed_dist+threshold)))
                # Exponential increase when outside of wall 
            elif signed_dist > 0:  # emergency, outside the wall
                force += -normal * strength * (10**(signed_dist+threshold))
                eminent_collision=True
        # Threshold based on max force 
        if force.length()>state['max_boundary_force']:
            force=force.normalized()*state['max_boundary_force']
        
        return force,eminent_collision>0
    
    def avoidance_force(self, fish, neighbor_radius=100, strength=100):
        """
        Calculate a force to avoid nearby fish (separation), consistent with boid flocking.
        neighbor_radius: distance within which other fish influence this fish
        strength: scaling factor for the avoidance force
        """
        eminent_collision = False
        avoidance = LVector3(0, 0, 0)
        fish_pos = fish.get_pos()
        state=self.fish_state[fish]
        for other in self.fish_list:
            if other == fish:
                continue
            other_pos = other.get_pos()
            offset = fish_pos - other_pos
            distance = offset.length()
            if distance < neighbor_radius:
                # Repel stronger if closer
                avoidance += offset.normalized() * (strength / distance**2)
                if distance < state['fish_width']:
                    eminent_collision = True
        # Check real fish 
        other_pos = self.real_fish.get_pos()
        offset = fish_pos - other_pos
        distance = offset.length()
        if distance < neighbor_radius:
            # Repel stronger if closer
            avoidance += offset.normalized() * (strength / distance**2)
            if distance < state['fish_width']:
                eminent_collision = True
        # Threshold if needed 
        if avoidance.length()>state['max_avoidance_force']:
            avoidance=avoidance.normalized()*state['max_avoidance_force']
        return avoidance,eminent_collision
    
    def cohesion_force(self, fish, neighbor_radius=100, strength=5):
        """
        Calculate a cohesion force to move the fish toward the center of nearby fish.
        neighbor_radius: distance within which neighbors influence this fish
        strength: scaling factor for the cohesion force
        """
        cohesion = LVector3(0, 0, 0)
        fish_pos = fish.get_pos()
        state=self.fish_state[fish]
        
        neighbors=[(other, (other.get_pos()-fish_pos).length()) for other in self.fish_list if other != fish and (other.get_pos()-fish_pos).length() < neighbor_radius]
        neighbors.append((self.real_fish, (self.real_fish.get_pos()-fish_pos).length()))
        neighbors.sort(key=lambda x: x[1])  # Sort by distance
        random_index=inverse_power_distribution_index(len(neighbors), p=1.5)# Get a random index based on inverse power distribution
        state['random_index']=random_index
        if len(neighbors) > 0:
            target_neighbor, _ = neighbors[random_index]

            cohesion = (target_neighbor.get_pos() - fish_pos).normalized() * strength

        # Threshold if needed 
        if cohesion.length()>state['max_cohesion_force']:
            cohesion=cohesion.normalized()*state['max_cohesion_force']
        
        return cohesion
    
    def path_follow_force(self, fish, state, thrust, drag_force,deltaT):
        """This calculates a force to keep the fish on a path that should prevent drift out of frame"""
        path_following_force = LVector3(0,0,0)
        fish_pos = fish.get_pos()

        # Define the path endpoints
        # Swap path direction depending on which way fish is moving
        if state["direction"] > 0:  # moving right
            path_start = state["fish_path_vector_left"]
            path_end = state["fish_path_vector_right"]
        else:  # moving left
            path_start = state["fish_path_ vector_right"]
            path_end = state["fish_path_vector_left"]
        path_vec = path_end - path_start

        # Predict future position
        new_veloc = state["velocity"] + (thrust+drag_force) *deltaT
        future_pos = fish_pos + new_veloc * deltaT

        # Project the future position onto the path (dot product)
        ap = future_pos - path_start
        proj = ap.dot(path_vec.normalized())
        
        # Clamp the projection to stay within the path segment
        #proj = max(0, min(proj, path_vec.length()))
        normal_point = path_start + path_vec.normalized() * proj

        # Distance from path
        dist_from_path = (future_pos - normal_point).length()

        if dist_from_path > state["fish_path_radius"]:
            # Steer toward path ahead
            target_point = normal_point + path_vec.normalized() * max(10,state["velocity"].length())
            path_following_force = (target_point - fish_pos).normalized() * dist_from_path**3
        
        return path_following_force
    
    def alignment_force(self, fish, neighbor_radius=100, strength=1.0):
        """
        Calculate an alignment force to align the fish's velocity with nearby neighbors.
        neighbor_radius: distance within which neighbors influence this fish
        strength: scaling factor for alignment
        """
        alignment = LVector3(0, 0, 0)
        state = self.fish_state[fish]
        fish_pos = fish.get_pos()
        neighbors=[(other, (other.get_pos()-fish_pos).length()) for other in self.fish_list if other != fish and (other.get_pos()-fish_pos).length() < neighbor_radius]
        neighbors.append((self.real_fish, (self.real_fish.get_pos()-fish_pos).length()))
        neighbors.sort(key=lambda x: x[1])  # Sort by distance
        if len(neighbors) > 0:
            neighbor, _ = neighbors[state['random_index']]
            if neighbor != self.real_fish:
                neighbor_state = self.fish_state[neighbor]

                alignment = (neighbor_state["velocity"] - state["velocity"]).normalized() * strength
            else:
                alignment=(self.real_fish_velocity - state["velocity"]).normalized() * strength
            
            # Threshold if needed 
            if alignment.length()>state['max_alignment_force']:
                alignment=alignment.normalized()*state['max_alignment_force']
        return alignment
    def fish_pause(self, fish, neighbor_radius=100):
        fish_pos = fish.get_pos()
        neighbors=[(other, (other.get_pos()-fish_pos).length()) for other in self.fish_list if other != fish and (other.get_pos()-fish_pos).length() < neighbor_radius]
        neighbors.append((self.real_fish, (self.real_fish.get_pos()-fish_pos).length()))
        neighbors.sort(key=lambda x: x[1])  # Sort by distance
        dist_to_nearest_neighbor = neighbors[0][1] if neighbors else float('inf')
        state = self.fish_state[fish]
        if not state["fish_paused"]:
            paused_neighbors =0
            for neighbor, _ in neighbors:
                if neighbor != self.real_fish:
                    neighbor_state = self.fish_state[neighbor]
                    if neighbor_state["fish_paused"]:
                        paused_neighbors += 1
            rand=np.random.rand()
            if rand<0.1*(math.log(paused_neighbors+1+10**-10)+1) and dist_to_nearest_neighbor<neighbor_radius:
                state = self.fish_state[fish]
                state["fish_paused"] = True
                state["pause_timer"] = 0
                state["start_time"] = time.time()
        else:
            state["pause_timer"] = time.time()-state["start_time"]
            if state["pause_timer"] > random.random()*state["pause_timer"]*50 or state["pause_timer"] > 3 or dist_to_nearest_neighbor>neighbor_radius:
                print(state["pause_timer"])
                state["fish_paused"] = False
                state["pause_timer"] = 0
                state["start_time"] = 0
        
    def righting_force(self, fish, righting_threshold = 0.6, strength=5.0):
        """Apply righting force to keep fish level in the water column."""
        damping=0.1
        state = self.fish_state[fish]
        vel = state["velocity"]

        # Desired direction has zero vertical tilt
        target = LVector3(vel.x, vel.y, 0)
        if target.length() == 0:
            return LVector3(0, 0, 0)

        # Compute small tilt error
        tilt_error = target.normalized() - vel.normalized()
        if abs(tilt_error.z) < righting_threshold:
            return LVector3(0, 0, 0)
        # Apply smooth correction only in Z
        correction = LVector3(0, 0, -vel.z * strength) - tilt_error * damping

        # Clamp to max righting force
        if correction.length() > state['max_righting_force']:
            correction = correction.normalized() * state['max_righting_force']
        return correction

    def burst_force(self,fish,social_forces, deltaT):
        """This applies burst force with random noise based on reynolds wander model"""
        state = self.fish_state[fish]
        burst_mag =  max(0,state["burst_strength"]-social_forces.length())
        fish_pos = fish.get_pos()

        # Apply wander 
        fish_width = state['fish_width']
        future_point = fish_pos + (state['velocity'].normalized() * state['fish_width']*6) 
        
        theta = random.uniform(-math.pi,math.pi)
        phi = random.uniform(0,math.pi)
        
        # Get spherical coordinates 
        x = math.sin(phi) * math.cos(theta) * fish_width*2
        y = math.sin(phi) * math.sin(theta) * fish_width*2
        z = math.cos(phi) * fish_width*2
        point_on_sphere =  LVector3(x, y, z)

        # translate spherical coordinates to future point 
        future_point+=point_on_sphere

        # Get vector between future point and fish 
        new_vector = (future_point - fish.get_pos()).normalized()
        
        # Apply burst 
        burst_force = new_vector*burst_mag
        
        return burst_force

    def burst_and_glide(self, fish, state, deltaT):
        thrust = LVector3(0, 0, 0)
        burst_force = LVector3(0, 0, 0)
        target_force =  LVector3(0, 0, 0)
        cohesion_force = LVector3(0, 0, 0)
        alignment_force = LVector3(0, 0, 0)
        righting_force = LVector3(0, 0, 0)
        
        ## APPLY FORCES HERE 
        #Always apply boundary force 
        boundary_force,boundary_collision = self.boundary_force(fish, state, threshold = state["boundary_radius"],strength=state['boundary_strength'])
        avoidance_force,fish_collision = self.avoidance_force(fish, neighbor_radius=state['avoidance_radius'], 
        strength=state['avoidance_strength'])
        thrust+=boundary_force+avoidance_force
        
        eminent_collision=(boundary_collision+fish_collision)>0

        ## APPLY FORCES HERE 
        if not eminent_collision:
            target_force = LVector3(0, 0, 0) #self.path_follow_force(fish, state, burst_force, drag_force,deltaT)
            avoidance_force, _ = self.avoidance_force(fish, neighbor_radius=state['avoidance_radius'], strength=state['avoidance_strength'])
            cohesion_force = self.cohesion_force(fish, neighbor_radius= state['cohesion_radius'], strength = state['cohesion_strength'])
            alignment_force = self.alignment_force(fish, neighbor_radius= state['alignment_radius'], strength = state['alignment_strength'])
            boundary_force, _ = self.boundary_force(fish, state, threshold=state["boundary_radius"], strength=state['boundary_strength'])
            righting_force = self.righting_force(fish,righting_threshold= state['righting_threshold'], strength=state['righting_strength'])
            social_forces = target_force + cohesion_force + alignment_force + righting_force
            burst_force = self.burst_force(fish,social_forces,deltaT)
        if state["fish_paused"]:
            thrust = LVector3(0, 0, 0)
            fish.stop('swim')
            fish.pose("swim",0)
            self.fish_pause(fish, neighbor_radius=state['avoidance_radius'])
        # BURST phase
        elif not state["Glide"]:
            self.fish_pause(fish, neighbor_radius=state['avoidance_radius'])
            if state["fish_paused"]:
                thrust = LVector3(0, 0, 0)
                fish.stop('swim')
                fish.pose("swim",0)
            else:
                if state['alignment_strength']>0:
                    social_forces = target_force + cohesion_force + alignment_force 
                else:
                    social_forces = LVector3(0,0,0)
                burst_force = (social_forces.normalized()*0.2 + burst_force.normalized()*0.8).normalized() * state['burst_strength']
                thrust += burst_force + righting_force
                # print(f'Total thrust {np.round(thrust)}. TF {np.round(target_force)}. AF {np.round(avoidance_force)}. BF {np.round(boundary_force)}. RF {np.round(righting_force)}. CF {np.round(cohesion_force)}. AF {np.round(alignment_force)}')
                # Start burst timer
                if state["burst_timer"] == 0:
                    fish.play("swim")
                    fish.setPlayRate(min(10, thrust.length()), "swim")
                    state['StartBurstTime'] = globalClock.get_frame_time()
                    state["burst_timer"] = globalClock.get_frame_time()

                elapsed_time = globalClock.get_frame_time() - state['StartBurstTime']
                # Switch to glide when burst timer exceeded
                if elapsed_time >= state["burst_duration"]:
                    state["Glide"] = True
                    state["burst_timer"] = 0
                    state["glide_timer"] = 0
                    state["EndBurstTime"] = globalClock.get_frame_time()
                    state["TrueBurstDuration"] = elapsed_time
                    # if state['TrueBurstDuration']<0.9*self.BURST_DIR-deltaT or state['TrueBurstDuration']>1.1*self.BURST_DIR+deltaT:
                    #     print(f"True burst duration {state['TrueBurstDuration']} is outside the expected range {0.9*self.BURST_DIR} to {1.1*self.BURST_DIR}")

        # GLIDE phase
        else:
            if state["glide_timer"] == 0:
                state["glide_timer"] = globalClock.get_frame_time()  # Small value to indicate timer has started
                fish.stop('swim')
                fish.pose("swim",0)
                state["StartGlideTime"] = globalClock.get_frame_time()
            elapsed_glide_time = globalClock.get_frame_time() - state["StartGlideTime"]
            #social_force_scaling = 
            if state['alignment_strength']>0:
                social_forces = target_force + cohesion_force + alignment_force 
                social_forces = (social_forces.normalized()) * np.sqrt(state['velocity'].length())
            else:
                social_forces = LVector3(0,0,0)
            
            thrust += social_forces+righting_force  # No thrust during glide. (Maybe I want some small steering forces?)
            #thrust += righting_force
            # Stop gliding when glide timer exceeded
            if elapsed_glide_time >= state["glide_duration"]:
                state["Glide"] = False
                state['glide_duration'] = self.GLIDE_DIR#+np.random.uniform(-self.GLIDE_DIR*.01,self.GLIDE_DIR*.01)
                state['burst_duration'] = self.BURST_DIR#+np.random.uniform(-self.BURST_DIR*.01,self.BURST_DIR*.01)
                state["EndGlideTime"] = globalClock.get_frame_time()
                state["TrueGlideDuration"] = state["EndGlideTime"] - state["StartGlideTime"]
                
                # if state['TrueGlideDuration']<0.9*self.GLIDE_DIR-deltaT or state['TrueGlideDuration']>1.1*self.GLIDE_DIR+deltaT:
                #     print(f"True glide duration {state['TrueGlideDuration']} is outside the expected range {0.85*self.GLIDE_DIR} to {1.15*self.GLIDE_DIR}")
                state["glide_timer"] = 0
                state["burst_timer"] = 0
                # #print(f'Min velocity ',state['velocity'].length(),' New burst strength ',state['burst_strength'])
                #print(f'Average velocity/cycle ',np.round(np.mean(state['running_velocities']),3),' True burst duration ',np.round(state["TrueBurstDuration"],3),' True glide duration ',np.round(state["TrueGlideDuration"],3), ' Burst strength ',np.round(state['burst_strength'],3))
                cycle_mean_speed = np.mean(state['running_velocities'])
                speed_error = state['target_veloc'] - cycle_mean_speed
                # if np.abs(speed_error)>0.05*state['target_veloc']:
                #     print(f'Speed error {np.round(speed_error,3)} for target veloc {state["target_veloc"]} and cycle mean speed {np.round(cycle_mean_speed,3)}')
                # Smoothing to reduce noise
                if 'speed_error_ema' not in state:
                    state['speed_error_ema'] = speed_error
                    state['burst_counter'] = 0 
                    state['kf_e'] = speed_error          # initial estimate
                    state['kf_P'] = 13.0                  # initial uncertainty
                    state['kf_Q'] = 0.1                # process noise (slow drift)
                    state['kf_R'] = 0.5                  # measurement noise
                    state['burst_counter'] = 0
                #state['speed_error_ema'] = ((1 - alpha) * state['speed_error_ema'] + alpha * speed_error)
                #new_burst_strength = min(max(0,state['burst_strength']+state['speed_error_ema']),state['max_burst_force'])
                #vibe coded 1d kalman filter (need to double check)
                P_pred = state['kf_P'] + state['kf_Q']
                x_pred = state['kf_e']

                # --- Update ---
                K = P_pred / (P_pred + state['kf_R'])    # Kalman gain
                #print('Kalman gain ',K)
                x_new = x_pred + K * (speed_error - x_pred)
                P_new = (1 - K) * P_pred

                state['kf_e'] = x_new
                state['kf_P'] = P_new
                new_burst_strength = np.clip(state['burst_strength'] + state['kf_e'],0,state['max_burst_force'])
                state['burst_strength'] = new_burst_strength
                #state['burst_strength'] =  new_burst_strength
                state['running_velocities'].clear()
                state['average_veloc'] = cycle_mean_speed
                state['burst_counter']+=1

        return thrust
    
    def peak_velocity_for_mean(self,v_mean=5.5,Cd=0.4,burst_duration=0.1,glide_duration=0.8):
        x_grid = np.linspace(0.01, 50.0, 2000)
        lhs = v_mean * (burst_duration + glide_duration)
        rhs = (
            (burst_duration / (2 * Cd * glide_duration)) * x_grid * (1 + 1/(1 + x_grid)) + (1 / Cd) * np.log(1 + x_grid))
        x = np.interp(lhs, rhs, x_grid)
        return x / (Cd * glide_duration)

    def update_fish(self, task):
        # This code updates the fish positions. Functions included below calculate forces etc. which ultimately move the fish
        if  globalClock.get_frame_time() - self.fish_show_time >1 and not self.paused:
            deltaT = globalClock.get_dt()
            for idx,fish in enumerate(self.fish_list):
                
                state = self.fish_state[fish]
                v = state["velocity"]
                v[2]=max(-self.config.vz_lim, min(self.config.vz_lim,v[2])) # Limit vertical velocity to prevent fish from going out of bounds
                m = state["mass"]
                Cd = state["friction_coeff"]
                speed = v.length()
                state['running_velocities'].append(speed)
                if speed > 0:
                    heading_rad = math.atan2(v.y, v.x)
                    fish.set_h(math.degrees(heading_rad)+90)
                    horizontal_speed = math.sqrt(v.x**2 + v.y**2)
                    pitch_rad = math.atan2(-v.z, horizontal_speed)
                    fish.set_p(math.degrees(pitch_rad))
                
                # Handle burst and glide mechanics
                thrust = self.burst_and_glide(fish,state,deltaT)
                drag_dir = -v.normalized()
                drag_force = drag_dir * Cd * (speed ** 2) if speed > 0 else LVector3(0,0,0) # and state['Glide'] 

                # Update motion
                state["acceleration"] = (thrust + drag_force)/m
                v += state["acceleration"] * deltaT
                # Enforce max velocity 
                v_mag = min(v.length(),state["max_velocity"])
                v = v.normalized() * v_mag
                new_pos = fish.get_pos() + v * deltaT 
                #set new pos 
                fish.set_pos(new_pos)
                # Save updated velocity
                state["velocity"] = v

        return Task.cont
    
    def update_real_fish(self,task):
        # This code would update the real fish position based on tracking centroid data stored in q_centroid
        if q_centroid:
            # 1.) Get centroid data from tracking system 
            centroid = q_centroid.popleft()
            if centroid:
                
                # 2.) Center centroid arround origin 
                XY_array = np.array([centroid[0]-self.tank_origin[0],centroid[1]-self.tank_origin[1]])
                
                # 3.) Rotate centroid to match arena rotation
                rotated_XY = np.dot(self.tank_rot_mat,XY_array)
                
                #3a.) Flip Y axis and X axis to match game coordinates
                rotated_XY[0] = -rotated_XY[0]
                rotated_XY[1] = -rotated_XY[1]+ArenaSize[2][0]
                
                # 4.) Scale to game units
                scaled_XY = rotated_XY * self.pixel_to_unit_scale

                # 5.) Get Z position based on some logic (for now, just center of tank height)
                min_point = self.tank_bounding_box_right.get_min()
                max_point = self.tank_bounding_box_right.get_max()
                tank_height = max_point.z - min_point.z
                Z_pos = min_point.z + tank_height/2
                
                # 7.) Set new position
                self.real_fish.set_pos(scaled_XY[1],scaled_XY[0],Z_pos)
                #print(f'Setting real fish pos to {np.round(scaled_XY[1])},{np.round(scaled_XY[0])}')
                #.)8 Adjust heading based on movement direction
                deltaT = globalClock.get_dt()
                if hasattr(self, 'last_real_fish_pos'):
                    movement_vector = self.real_fish.get_pos() - self.last_real_fish_pos
                    if movement_vector.length() > 0:
                        # Compute target heading
                        target_heading = math.degrees(math.atan2(movement_vector.y, movement_vector.x))

                        # Initialize if first frame
                        if not hasattr(self, 'smoothed_heading'):
                            self.smoothed_heading = target_heading
                        else:
                            smooth_rate = 0.0016/deltaT  # lower = smoother 

                            # Compute smallest angular difference (wrap-around safe)
                            delta = (target_heading - self.smoothed_heading + 180) % 360 - 180

                            # Apply smoothing toward target
                            self.smoothed_heading += smooth_rate * delta
                            # Convert smoothed heading to unit vector velocity 
                            self.real_fish_velocity = LVector3(
                                math.cos(math.radians(self.smoothed_heading)),
                                math.sin(math.radians(self.smoothed_heading)),
                                0
                            ) * (movement_vector.length()/deltaT)
                        # Apply to model (+90 to align model's facing direction)
                        self.real_fish.set_h(self.smoothed_heading + 90)

                self.last_real_fish_pos = self.real_fish.get_pos()
        return Task.cont
    

    def stop_game(self, task=None):
        self.userExit()  # This safely exits the game
        return Task.done  # Stops the task after execution
    def pause_game(self, task=None):
        self.paused = not self.paused
        if self.paused:
            print("Game paused.")
        else:
            print("Game resumed.")
        return Task.cont
    
    def check_time(self, task):
        if self.start_timer.is_set():
            current_time = globalClock.get_frame_time() - self.fish_show_time
            #print('Current time: ',current_time)
            if current_time >= self.Delay and self.trial_count == 0 and self.start_saving.is_set() is False:
                self.start_saving.set()
                if self.config.SaveMode:
                    print('Starting to save fish positions at ', dt.datetime.now())
                    self.taskMgr.doMethodLater(0.03, self.save_fish_positions, "SaveTask")  # Reduced frequency
                
            #
            if current_time < self.Delay + self.OffTime:
                # OFF PHASE
                self.hide_fish()
                if self.fish_showing.is_set():  # only if currently shown
                    self.fish_showing.clear()
                    print('Hiding fish at time ',dt.datetime.now())
            elif current_time < self.Delay + self.OffTime + self.OnTime:
                # ON phase: show fish
                if not self.fish_showing.is_set():
                    self.fish_showing.set()
                    print('Showing fish at time ', dt.datetime.now())
                    self.show_fish()
            else:
                # End of ON phase → start next trial
                if self.fish_showing.is_set():
                    self.fish_showing.clear()
                    self.hide_fish()
                    # Clear fish list and state 
                    print('NUM OBJECTS FULL ',self.render.getNumChildren())
                    for fish in self.fish_list:
                        fish.cleanup()
                        fish.removeNode()
                    self.fish_list.clear()
                    self.fish_state.clear()
                    print('NUM OBJECTS AFTER CLEAR ',self.render.getNumChildren())
                    # Re-init fish 
                    self.init_fish()
                if self.trial_count==0: #zero out habituation delay for subsequent trials
                    self.Delay=0
                self.trial_count += 1
                print(f'Trial count: {self.trial_count}')
                self.fish_show_time = globalClock.get_frame_time()

                # Check for end of all trials
                if self.trial_count >= self.num_trials:
                    print('All trials complete. Stopping data saving.')
                    self.stop_event.set()
                    return Task.done
            return Task.again  # Use again instead of cont for timed tasks
    
    def get_model_dimensions(self, model):
        min_point, max_point = model.getTightBounds()
        size = max_point - min_point
        return size.x, size.y, size.z  # Depth,Width, Height
    
    def get_FOV_coordinates(self, fish):
        """
        Returns camera FOV bounds in world coordinates at a given depth.
        depth: distance along the camera's forward axis (Y in Panda3D).
        Outputs: LeftBound, RightBound, TopBound, BottomBound
        """
        # Camera position and orientation
        cam_pos = self.camera.get_pos(render)
        fish_pos = fish.get_pos()

        # Half-width and half-height at depth
        fov_x, fov_y = self.camLens.get_fov()
        half_width  = math.tan(math.radians(fov_x / 2)) * (abs(cam_pos.y)+fish_pos.y)
        half_height = math.tan(math.radians(fov_y / 2)) * (abs(cam_pos.y)+fish_pos.y)

        # Approximate horizontal bounds in world X
        LeftBound  = 0 #cam_pos.x - half_width
        RightBound = cam_pos.x + half_width

        # Approximate vertical bounds in world Z
        BottomBound = cam_pos.z - half_height
        TopBound    = cam_pos.z + half_height

        return LeftBound, RightBound, TopBound, BottomBound
    def get_screen_coordinates_smiley(self,obj):
        # Get the 3D position relative to camera 
        cam_space_pos = self.camera.get_relative_point(self.render, obj.get_pos(self.render))
        screen_pos = Point2()
        if self.camLens.project(cam_space_pos, screen_pos):
            win_width = base.win.get_x_size()
            win_height = base.win.get_y_size()
            pixel_x = (screen_pos.x + 1) / 2 * win_width
            pixel_y = (1 - screen_pos.y) / 2 * win_height
            return pixel_x, pixel_y
        else:
            return None, None
        
    def get_screen_coordinates(self, fish):
        """
        Convert a fish's 3D position to 2D screen coordinates in pixels,
        aligned to the top-down video.
        """
        state = self.fish_state[fish]
        # 1. Get 3D world position of fish
        try:
            fish_pos = fish.get_pos(self.render)
        except:
            return None,None

        # 2. Convert to camera-relative coordinates
        cam_pos = self.camera.get_relative_point(self.render, fish_pos)

        # 3. Project to 2D normalized device coordinates [-1,1]
        screen_pos = Point3()
        if not self.camLens.project(cam_pos, screen_pos):
            return None, None  # behind camera or outside view
        
        # 4. Determine arena pixel size and offsets
        arena_width  = self.config.ArenaSize[2][0] #* self.OverhangRatio # pixels along X (horizontal in top-down view), Multiplied by 1.07 due to overhanging edge in video
        arena_height = arena_width                  # Square arena in top-down view
        x_offset = self.config.ArenaSize[7][0][0] if state['tank_side']=='left' else self.config.ArenaSize[7][1][0] # left edge pixel X versus right edge pixel X
        y_offset = self.config.ArenaSize[7][0][1] if state['tank_side']=='left' else self.config.ArenaSize[7][1][1]# left edge pixel Y versus right edge pixel Y
        
        # Interpolate left/right world X at the fish’s Y position
        vertices = self.tank_vertices_left if  state['tank_side']=='left' else self.tank_vertices_right
        LFB, LFT, LBB, LBT, RFB, RFT, RBB, RBT = vertices

        # Normalize Y position (0 = front, 1 = back)
        norm_y = (fish_pos.y - self.tank_min_y) / self.tank_depth
        norm_y = max(0.0, min(1.0, norm_y))

        # Left and right walls move in X as Y increases (front → back)
        left_x_at_y  = LFB.x + (LBB.x - LFB.x) * norm_y
        right_x_at_y = RFB.x + (RBB.x - RFB.x) * norm_y

        # Compute normalized X position between these edges
        den = (right_x_at_y - left_x_at_y)
        if abs(den) < 1e-6:
            norm_x = 0.5
        else:
            norm_x = (fish_pos.x - left_x_at_y) / den

        # Clamp to [0,1]
        norm_x = max(0.0, min(1.0, norm_x))
        if  state['tank_side']=='right':
            norm_x = 1-norm_x
        B = x_offset + arena_width if state['tank_side']=='left' else  x_offset - arena_width
        screen_x = (x_offset - B) * norm_x + B   # Invert X axis for top-down view

        # Use camera projection for Y
        screen_y = y_offset - ((screen_pos.y) / 2) * arena_height
        
        # Rotate coordinates to account for tank rotation 
        XY_array = np.array([screen_x-x_offset,screen_y-y_offset])
                
        rotated_XY = np.dot(self.tank_rot_mat,XY_array)

        # Add back in offset 
        screen_x,screen_y = rotated_XY[0]+x_offset,rotated_XY[1]+y_offset
        
        return screen_x, screen_y
    def hide_fish(self):
        for fish in self.fish_list:
            fish.hide()
            
    def show_fish(self):
        for fish in self.fish_list:
            fish.show()
    
    def show_bbox(self,vertices):
        # 8 corners of box
        [LFB, LFT, LBB, LBT, RFB, RFT, RBB, RBT] = vertices#[LFB,LFT,LBB,LBT,RFB,RFT,RBB,RBT]

        lines = LineSegs()
        lines.set_thickness(2.0)
        lines.set_color(1,0,0,1)  # red

        # Connect edges of the box
        def connect(a,b):
            lines.move_to(a)
            lines.draw_to(b)

        # front face
        connect(LFB,LFT)
        connect(LFT,RFT)
        connect(RFT,RFB)
        connect(RFB,LFB)

        # back face
        connect(LBB,LBT)
        connect(LBT,RBT)
        connect(RBT,RBB)
        connect(RBB,LBB)

        # sides
        connect(LFB,LBB)
        connect(LFT,LBT)
        connect(RFB,RBB)
        connect(RFT,RBT)

        node = lines.create()
        np = NodePath(node)
        np.reparent_to(self.render)
        
        return np
    def random_point_in_tank(self,fish_width,fish_height,tank,max_tries=5000):
        """
        Returns a random Point3 inside the convex tank defined by self.tank_planes.
        Uses rejection sampling inside the axis-aligned bounding box.
        """
        if tank=='left':
            min_pt = self.tank_bounding_box_left.get_min()
            max_pt = self.tank_bounding_box_left.get_max()
        elif tank=='right':
            min_pt = self.tank_bounding_box_right.get_min()
            max_pt = self.tank_bounding_box_right.get_max()
        else:
            print('ERROR! ENTER LEFT OR RIGHT!!') 
            
        for _ in range(max_tries):
            # Sample random point inside the axis-aligned bounding box
            p = Point3(
                random.uniform(min_pt.x+fish_width, max_pt.x-fish_width),
                random.uniform(min_pt.y, max_pt.y),
                random.uniform(min_pt.z+fish_height, max_pt.z-fish_height)
            )
            # Check if point is inside all planes
            inside = True
            tank_planes = self.tank_planes_left if tank=='left' else self.tank_planes_right
            for normal, d in tank_planes.values():
                if normal.dot(p) + d > 0:  # outside
                    inside = False
                    break

            if inside:
                return p
        raise RuntimeError("Failed to find a point inside the tank after many tries")

    def camera_orbit(self,task):
        if not hasattr(self, "Distance"):
            first_cam_pos = self.camera.get_pos() 
            # Look at center of tank 
            min_point = self.tank_bounding_box_left.get_min()
            max_point = self.tank_bounding_box_left.get_max()
            self.center = (min_point + max_point) * 0.5
            self.Distance = (first_cam_pos - self.center).length()
        else:
            angleDegrees = task.time * 2.0
            angleRadians = angleDegrees * (math.pi / 180.0)
            Radius = self.Distance
            self.camera.setPos(Radius*math.sin(angleRadians), -Radius * math.cos(angleRadians), self.config.CamHeight)
            self.camera.look_at(self.center)
            
        return Task.cont
    def camera_offset(self,FixedCameraAngle=45,cam_height_offset=0):
        first_cam_pos = self.camera.get_pos() 
        # Look at the center of whichever tank side actually has the virtual fish in it
        # (previously this always used tank_bounding_box_left, regardless of stimulus_side)
        if self.config.stimulus_side == 'left':
            bounding_box = self.tank_bounding_box_left
        else:
            bounding_box = self.tank_bounding_box_right
        min_point = bounding_box.get_min()
        max_point = bounding_box.get_max()
        self.center = (min_point + max_point) * 0.5
        self.Distance = (first_cam_pos - self.center).length()
        angleDegrees = FixedCameraAngle
        angleRadians = angleDegrees * (math.pi / 180.0)
        Radius = self.Distance
        # Anchor camera height to the tank's own height (plus a small optional offset)
        # rather than CamHeight+cam_height_offset, which previously let a large
        # cam_height_offset (100) put the camera almost directly above the tank.
        self.camera.setPos(Radius*math.sin(angleRadians), -Radius * math.cos(angleRadians), self.center.z+cam_height_offset)
        self.camera.look_at(self.center)
    
    ### Saving functions ###
    # def save_metadata_once(self,task):
    #     self.metadataParams = {
    #         "experiment_name": experiment_name,
    #         "numTrials": self.num_trials,
    #         "Delay time": self.config.Delay,
    #         "Off time": self.config.OffTime,
    #         "On time": self.config.OnTime,
    #         "Number of fish": self.num_fish,
    #         # Enter scale factors
    #         "Pixel to unit scale": self.pixel_to_unit_scale,
    #         "MM to unit scale": self.mm_to_unit_scale,
    #         "Arena size (pixels)": self.config.ArenaSize[7],
    #         "Fish size (mm)": self.config.fish_size,
    #         "Real fish school strength": self.config.RealFishSchoolStrength,
    #         "Interaction limit virtual fish (mm)": self.config.InteractionLimit,
    #         "Interaction limit real fish (mm)": self.config.InteractionLimitRealFish,
    #         "Tank size": self.config.TankSize,
    #         # Save out  fish state parameters for just one fish as reference
    #         "Fish state parameters": self.fish_state[self.fish_list[0]],
    #         "Fish model path": self.fish_model_paths,
    #         "Fish scale": self.config.FishScale,
    #         "Fish alpha": self.config.Fishalpha,
    #         "Background color": BackgroundColor,
    #         "Camera distance": CamDist,
    #         "Camera height": CamHeight,
    #         "Camera position": self.camera.get_pos(),
    #         "Camera orientation": self.camera.get_hpr(),
    #         "Terrain scale": self.terrain.get_scale(),
    #         "Target velocity": TARGET_VELOC,
    #         "Metadata_filepath": experiment_directory + r'/'+ experiment_name + "_metadata.csv"
    #     }
    #     self.save_metadata(self.metadataParams)
    #     self.taskMgr.remove("SaveMetadataTask")
    #     return Task.done  # Stop the task after saving metadata

    # def save_metadata(self, params):
    #     """Saves metadata parameters to a CSV file."""
    #     file_path = params["Metadata_filepath"]

    #     # Write metadata to CSV
    #     with open(file_path, mode="w", newline="") as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["Parameter", "Value"])
    #         for key, value in params.items():
    #             writer.writerow([key, value])

    #     print(f"Metadata saved to: {file_path}")
    # def save_fish_positions(self, task):
    #     if start_saving.is_set():
    #         current_time = globalClock.getRealTime() 
    #         with open(self.fish_pos_path, mode='a', newline='') as csvfile:
    #             writer = csv.writer(csvfile)
    #             if self.FirstSave:
    #                 header = ['Timestamp']
    #                 for idx, fish in enumerate(self.fish_list):
    #                     # Pixel coordinates
    #                     header.append(f'Fish_{idx}_Y_pixel')
    #                     header.append(f'Fish_{idx}_X_pixel')
    #                     # Game world coordinates here
    #                     header.append(f'Fish_{idx}_X_world')
    #                     header.append(f'Fish_{idx}_Y_world')
    #                     header.append(f'Fish_{idx}_Z_world')
    #                     header.append(f'Fish_{idx}_Burst_counter')
    #                     header.append(f'Fish_{idx}_Avg_veloc')
    #                 writer.writerow(header)
    #                 self.FirstSave = False
    #             # Write a new row
    #             row = [current_time]
    #             positions = []
    #             for fish in self.fish_list:
    #                 state = self.fish_state[fish]
    #                 # Get fish position and flipper state
    #                 y_pixel,x_pixel = self.get_screen_coordinates(fish)
    #                 # Store for calibration/debug/visualize
    #                 positions.append((x_pixel, y_pixel))
    #                 fish_pos = fish.get_pos()
    #                 if y_pixel is not None:
    #                     row.extend([int(y_pixel), int(x_pixel),fish_pos.x, fish_pos.y, fish_pos.z,state['burst_counter'],state['average_veloc']])
    #                 else:
    #                     row.extend([None, None, None, None, None])
    #             writer.writerow(row)
    #             # Try to put the latest positions in the queue
    #             try:
    #                 q_school.put_nowait(positions)
    #             except queue.Full:
    #                 # If full, remove the old positions and put the new ones
    #                 try:
    #                     q_school.get_nowait()
    #                 except queue.Empty:
    #                     pass
    #                 q_school.put_nowait(positions)
    #     return Task.again   


if __name__ == "__main__":

    print("BOID fish school simulation classes and helper funcs. \nTo run the simulation, execute the main script (main.py) which imports this module.")