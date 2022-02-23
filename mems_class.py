# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import os, sys, glob

try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import carla
from carla import ColorConverter as cc

import numpy as np
from time import sleep
import copy
from scipy.spatial.transform import Rotation as R

def r_0(t, T):
    """
    See: https://docs.blickfeld.com/cube/v1.0.1/scan_pattern.html
    Simple Ramp Function, 
    with only uo-ramping phase.
    
    Input:
        t - time
        T - frame duration
    Returns:
        ramp value at time t
    """
    return t / T
    
def h_mirror(t, t_h_max = 80, f = 150):
    """
    See: https://docs.blickfeld.com/cube/v1.0.1/scan_pattern.html
    Horizontal Mirrow
    Input:
        t - time
        t_h_max - horizontal field of view
        f - eigenfrequency
    
    Returns:
        Angle of the horizontal mirrow.
    """
    return t_h_max / 2 * np.cos(2 * np.pi * f * t)
    
def v_mirror(t, t_v_max = 30, f = 150, T = None, ramp_function = r_0):
    """
    See: https://docs.blickfeld.com/cube/v1.0.1/scan_pattern.html
    Vertical Mirrow
    Input:
        t - time
        t_v_max - vertical field of view
        f - eigenfrequency
        T - frame duration
        ramp_function - Ramp Function
    
    Returns:
        Angle of the vertical mirrow.
    """
    return ramp_function(t, T = T) * t_v_max / 2 * np.sin(2 * np.pi * f * t)

def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])

class MEMS_Sensor(object):
    def __init__(self, parent_actor, carla_transform, image_width = 800, v_fov = 30, h_fov_total = 90, 
                 h_fov_pc = 70, n_scan_lines = 100, n_points_pattern = 100000, 
                 max_range = 100, out_root = "out/MEMS", add_noise = True):
        """
        Class for MEMS Sensor
        """
        print("Spawn MEMS Sensor")
        
        self.out_root = out_root
        self.sensor = None
        self.frame = 0
        self.add_noise = add_noise
        self.parent = parent_actor
        self.max_range = max_range
        self.carla_transform = carla_transform
        
        world = self.parent.get_world()
        
        # Calculates the image height from the field of view and the image width
        image_height = np.tan(v_fov * np.pi / 180 / 2) / np.tan(h_fov_total * np.pi / 180 / 2) * image_width
        
        # Intrinsic Matrix of the Camera
        self.K = intrinsic_from_fov(image_height, image_width, h_fov_total)
        self.K_inv = np.linalg.inv(self.K)
        
        # Scan Pattern of the MEMS Sensor
        self.pixel_coords_scan_pattern = MEMS_Sensor.scan_pattern(n_scan_lines = n_scan_lines, 
                                            n_points_pattern = n_points_pattern,
                                            width = image_width, height = image_height,
                                            h_fov_total = h_fov_total, h_fov_pc = h_fov_pc, v_fov = v_fov)
        
        # Spawn of a depth map sensor
        sensor = world.get_blueprint_library().find('sensor.camera.depth')
        sensor.set_attribute('image_size_x',str(image_width))
        sensor.set_attribute('image_size_y',str(image_height))
        sensor.set_attribute('fov',str(h_fov_total))
        self.sensor = world.spawn_actor(sensor, self.carla_transform, attach_to = parent_actor)
        
        # Starts the recording
        self.sensor.listen(lambda dm: self.save_data(dm))

    def save_data(self, raw_depth_map):
        depth_map = np.float64(MEMS_Sensor.get_depth_map(raw_depth_map))
        depth_map = MEMS_Sensor.in_meters(depth_map)
        depth_map[depth_map > self.max_range] = np.nan
        
        point_cloud = self.depth_map_to_point_cloud(depth_map)
        print(os.path.join(self.out_root, "%06d" % self.frame))
        np.save(os.path.join(self.out_root, "%06d" % self.frame), point_cloud)
        self.frame += 1
    
    def depth_map_to_point_cloud(self, depth_map):
        """
            Calculates a 3D point cloud according to
            the scan pattern and depth map.
        """
        cam_coords = self.K_inv[:3, :3] @ (self.pixel_coords_scan_pattern)
        cam_coords = cam_coords * depth_map[np.int32(self.pixel_coords_scan_pattern[1]),
                                 np.int32(self.pixel_coords_scan_pattern[0])].flatten()
        self.lidar_pc = cam_coords[np.logical_not(np.isnan(cam_coords))].reshape(3,-1)
        
        self.lidar_pc = self.lidar_pc[[2,0,1]]
        self.lidar_pc[2] *= -1
        self.rot_transl_pc()
        
        if self.add_noise:
            self.noise()
            
        return self.lidar_pc

    def destroy(self):
        if self.sensor:
            print("MEMS Sensor destroyed")
            self.sensor.destroy()
            
    def rot_transl_pc(self):
        rot_mat = R.from_euler("xyz",[self.parent.get_transform().rotation.roll - self.sensor.get_transform().rotation.roll, # roll
                                      self.parent.get_transform().rotation.pitch - self.sensor.get_transform().rotation.pitch, # pitch
                                      - self.parent.get_transform().rotation.yaw + self.sensor.get_transform().rotation.yaw], # yaw
                               degrees=True).as_matrix()
        self.lidar_pc = np.dot(rot_mat, self.lidar_pc)
        self.lidar_pc = (self.lidar_pc.T + [self.carla_transform.location.x, self.carla_transform.location.y, self.carla_transform.location.z]).T
    
    def noise(self):
        """
        Add some noise on the data.
        """
        # Randomly dropping Points
        self.lidar_pc = self.lidar_pc[:,np.random.choice(self.lidar_pc.shape[1], np.int32(self.lidar_pc.shape[1]*0.95), replace=False)]
        
        # Disturb each point along the vector of its raycast
        self.lidar_pc += self.lidar_pc * np.random.normal(0, 0.005, self.lidar_pc.shape[1])
        
    @staticmethod
    def in_meters(depth_map):
        """
        Transforms the depth map image from RGB encoding to
        values in meter.
        See: https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
        """
        R_channel = depth_map[:,:,0].copy()
        G_channel = depth_map[:,:,1].copy()
        B_channel = depth_map[:,:,2].copy()
        depth_map = 1000 * (R_channel + G_channel * 256 + B_channel * 256 * 256) / (256 * 256 * 256 - 1)
        return depth_map

    @staticmethod
    def get_depth_map(raw_depth_map):
        """
        Fetchs the corresponding depth map image from the CARLA server.
        """
        raw_depth_map.convert(cc.Raw)
        
        depth_map = np.frombuffer(raw_depth_map.raw_data, dtype=np.dtype("uint8"))
        depth_map = np.reshape(depth_map, (raw_depth_map.height, raw_depth_map.width, 4))
        depth_map = depth_map[:, :, :3]
        depth_map = depth_map[:, :, ::-1]
        return depth_map

    @staticmethod
    def scan_pattern(n_scan_lines, n_points_pattern, width, height, h_fov_total = 80, h_fov_pc = 70, v_fov = 30, ramp_function = r_0, f = 150):
        """
        See: https://docs.blickfeld.com/cube/v1.0.1/scan_pattern.html
        Generates the scan pattern.
        Input:
            n_scan_lines - Number of scan lines
            n_points_pattern - Number of points on the total scan pattern
            width - depth map width (influences the accuracy)
            height - depth map height (influences the accuracy)
            h_fov_total - horizontal field of view of the scan patter in total
            h_fov_pc - horizontal field of view of the point cloud
            v_fov - vertical field of view
            ramp_function - Ramp Function
            f - eigenfrequency
        
        Returns:
            Coordinates of the points on the depth image.
        """
        T = n_scan_lines / 2 / f
        t_list = np.arange(0, T, T / n_points_pattern)
        
        h_mirror_list = np.array([h_mirror(t, t_h_max = h_fov_total, f = f) for t in t_list])
        v_mirror_list = np.array([v_mirror(t, t_v_max = v_fov, f = f, T = T, ramp_function = ramp_function) for t in t_list])
        x = (width / (h_fov_total) * (h_mirror_list + h_fov_total / 2) - 1)[(h_mirror_list > - h_fov_pc / 2) & (h_mirror_list < h_fov_pc / 2)]
        y = (height / (v_fov) * (v_mirror_list + v_fov / 2) - 1)[(h_mirror_list > - h_fov_pc / 2) & (h_mirror_list < h_fov_pc / 2)]
        
        return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

if __name__ == "__main__":
    pass
