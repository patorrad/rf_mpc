#!/usr/bin/env python

import rospy
import numpy as np
import math
import sys
from PIL import Image

import utils
from utils import get_angular_distance, get_angle_2_vectors

from sensor_msgs.msg import LaserScan, JointState
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion

import tf
import tf.transformations

np.set_printoptions(threshold=sys.maxsize)

SCAN_TOPIC = '/car/scan' # The topic to subscribe to for laser scans
#CMD_TOPIC = '/vesc/high_level/ackermann_cmd_mux/input/nav_0' # The topic to publish controls to
CMD_TOPIC = '/car/mux/ackermann_cmd_mux/input/navigation'
#POSE_TOPIC = '/sim_car_pose/pose' # The topic to subscribe to for current pose of the car
POSE_TOPIC = '/car/car_pose'
#POSE_TOPIC = '/gps_new'
VIZ_TOPIC = '/laser_wanderer/rollouts' # The topic to publish to for vizualizing
                                       # the computed rollouts. Publish a PoseArray.

MAX_PENALTY = 10000 # The penalty to apply when a configuration in a rollout
                    # goes beyond the corresponding laser scan

'''
Wanders around using minimum (steering angle) control effort while avoiding crashing
based off of laser scans. 
'''
class LaserWanderer:

  '''
  Initializes the LaserWanderer
    rollouts: An NxTx3 numpy array that contains N rolled out trajectories, each
              containing T poses. For each trajectory, the t-th element represents
              the [x,y,theta] pose of the car at time t+1
    deltas: An N dimensional array containing the possible steering angles. The n-th
            element of this array is the steering angle that would result in the 
            n-th trajectory in rollouts
    speed: The speed at which the car should travel
    compute_time: The amount of time (in seconds) we can spend computing the cost
    laser_offset: How much to shorten the laser measurements
  '''
  def __init__(self, rollouts, deltas, speed, compute_time, laser_spread, laser_offset):
    self.rollouts = rollouts
    self.deltas = deltas
    self.speed = speed
    self.compute_time = compute_time
    self.laser_spread = laser_spread
    self.laser_offset = laser_offset
    self.goal_pose = np.array([10, 5, 0, 0, 0, 0, 1])
    self.current_pose = np.array([0, 0, 0, 0, 0, 0, 1])
    self.rssi_rollouts = np.zeros(len(deltas))
    self.counter = 0
    self.cmd_pub = rospy.Publisher(CMD_TOPIC, AckermannDriveStamped, queue_size=1)
    self.laser_sub = rospy.Subscriber(SCAN_TOPIC, LaserScan, self.wander_cb, queue_size=2)
    self.viz_pub = rospy.Publisher(VIZ_TOPIC, PoseArray, queue_size=1)
    self.viz_sub = rospy.Subscriber(POSE_TOPIC, PoseStamped, self.viz_sub_cb, queue_size=1)
    self.goal_pub = rospy.Publisher('/goal_PT', PoseStamped, queue_size=1)
    self.rssi_sub = rospy.Subscriber('/occupancy_grid', OccupancyGrid, self.rssi_map_cb, queue_size=1)
    self.viz_rssi_sub = rospy.Subscriber(POSE_TOPIC, PoseStamped, self.rssi_pose_cb, queue_size=1)
    self.viz_rssi_pub = rospy.Publisher("/rssi_markers", MarkerArray, queue_size = 2)
    self.cost_pub = rospy.Publisher('/mpc_cost', JointState, queue_size = 1)

  def rssi_map_cb(self, msg):
    rssi_array = np.array(msg.data).reshape((msg.info.height, msg.info.width)).astype(float)
    #self.rssi_map = rssi_array/np.linalg.norm(rssi_array) # Normalize data
    

    if self.counter == 0:
      array = rssi_array.astype(np.uint8)
      image = Image.fromarray(array)

      image.save('/home/robot/Downloads/output_image.png')
      self.counter += 1
    rssi_array = rssi_array.astype(float)
    self.rssi_map = (rssi_array-np.min(rssi_array))/(np.max(rssi_array)-np.min(rssi_array)) 
  '''
  Get value of rssi at certain pose
  '''
  def rssi_pose_cb(self, msg):
    ma = MarkerArray()
    
    trans = np.array([msg.pose.position.x, msg.pose.position.y]).reshape((2,1))
    car_yaw = utils.quaternion_to_angle(msg.pose.orientation)
    rot_mat = utils.rotation_matrix(car_yaw)
    for i in xrange(self.rollouts.shape[0]):
        robot_config = self.rollouts[i,-1,0:2].reshape(2,1)
        map_config = rot_mat*robot_config+trans
        map_config.flatten()
        
        marker = Marker()

        marker.header.frame_id = "/map"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 1
        marker.id = i
        
        width = int(round(map_config[0] / 0.3)) # 0.3 is the map scale
        if width >= 75: width = 74
        height = int(round(map_config[1] / 0.3))
        if height >= 112: height = 111
        rssi_value = self.rssi_map[height, width]
        self.rssi_rollouts[i] = rssi_value

        # Set the scale of the marker
        marker.scale.x = rssi_value
        marker.scale.y = rssi_value
        marker.scale.z = rssi_value

        # Set the color
        marker.color.r = rssi_value
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = rssi_value

        # Set the pose of the marker
        marker.pose.position.x = map_config[0]
        marker.pose.position.y = map_config[1]
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        ma.markers.append(marker) 
    self.viz_rssi_pub.publish(ma)
  '''
  Vizualize the rollouts. Transforms the rollouts to be in the frame of the world.
  Only display the last pose of each rollout to prevent lagginess
    msg: A PoseStamped representing the current pose of the car
  '''      
  def viz_sub_cb(self, msg):
    pa = PoseArray()
    pa.header.frame_id = '/map'
    pa.header.stamp = rospy.Time.now()
    
    self.current_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    trans = np.array([msg.pose.position.x, msg.pose.position.y]).reshape((2,1))
    car_yaw = utils.quaternion_to_angle(msg.pose.orientation)
    rot_mat = utils.rotation_matrix(car_yaw)
    for i in xrange(self.rollouts.shape[0]):
        robot_config = self.rollouts[i,-1,0:2].reshape(2,1)
        map_config = rot_mat*robot_config+trans
        map_config.flatten()
        pose = Pose()
        pose.position.x = map_config[0]
        pose.position.y = map_config[1]
        pose.position.z = 0.0
        pose.orientation = utils.angle_to_quaternion(self.rollouts[i,-1,2]+car_yaw)
        pa.poses.append(pose)
    self.viz_pub.publish(pa)

  '''
  Compute the cost of one step in the trajectory. It should penalize the magnitude
  of the steering angle. It should also heavily penalize crashing into an object
  (as determined by the laser scans)
    delta: The steering angle that corresponds to this trajectory
    rollout_pose: The pose in the trajectory 
    laser_msg: The most recent laser scan
  '''      
  def compute_cost(self, delta, rollout_pose, current_local_goal, rssi, laser_msg, i, car_yaw, goal_yaw):
    #cost = np.abs(delta)
    '''
    # Compute the angle between this rollout and robot
    raycast_angle = np.arctan2(rollout_pose[1], rollout_pose[0])
    raycast_length = np.sqrt(rollout_pose[0]**2 + rollout_pose[1]**2)
    
    
    scan_idx = int((raycast_angle - laser_msg.angle_min)/laser_msg.angle_increment)
    
    if scan_idx < 0 or scan_idx >= len(laser_msg.ranges):
        return cost
    
    min_scan_idx = max(0, scan_idx-int(0.5*self.laser_spread/laser_msg.angle_increment)-1)
    max_scan_idx = min(len(laser_msg.ranges)-1, scan_idx+int(0.5*self.laser_spread/laser_msg.angle_increment)+1)
    laser_scans = np.array(laser_msg.ranges[min_scan_idx:max_scan_idx+1], dtype=np.float)
    laser_scans[np.isnan(laser_scans)] = 100.0 
    laser_scans[laser_scans[:] == 0] = 100.0
    scan_idx = min_scan_idx + np.argmin(laser_scans)
    
    if (not math.isnan(laser_msg.ranges[scan_idx])) and (raycast_length > laser_msg.ranges[scan_idx] - np.abs(self.laser_offset)):
        cost += MAX_PENALTY
    '''
    # Steering cost
    delta = 0.05 * np.abs(delta)
    # Distance cost
    dist = np.linalg.norm(rollout_pose[:2] - current_local_goal[:2])
    #print dist
    #import pdb; pdb.set_trace()
    # Angle deviation cost
    angle = 0
    #angle = get_angle_2_vectors(current_local_goal[:2], rollout_pose[:2])
    trans = rollout_pose[:2] - current_local_goal
    rot_mat = utils.rotation_matrix(rollout_pose[2]).T
    #current_local_goal = np.array(np.dot(rot_mat, current_local_goal[:2])).squeeze()
    if True:
      trans = self.current_pose[:2].reshape((2,1))
      car_yaw = car_yaw
      rot_mat = utils.rotation_matrix(car_yaw)
      robot_config = rollout_pose[:2].reshape(2,1)
      map_config = rot_mat*robot_config+trans
      map_config.flatten()
      map_config = np.array(map_config).squeeze()
      direction_vector = np.array([self.goal_pose[0] - map_config[0], self.goal_pose[1] - map_config[1]])
      theta_direction = np.arctan2(direction_vector[1], direction_vector[0])
      diff = (rollout_pose[2] + car_yaw - theta_direction + np.pi) % (2 * np.pi) - np.pi
      angle = np.abs(diff)
 
    # Communication cost
    rssi = 1/np.square(rssi)

    # Total cost
    cost = angle #+ rssi

    return cost    
  

  '''
  Controls the steering angle in response to the received laser scan. Uses approximately
  self.compute_time amount of time to compute the control
    msg: A LaserScan
  '''    
  def wander_cb(self, msg):
    start = rospy.Time.now().to_sec()
    
    current_local_goal = self.goal_pose[:2] - self.current_pose[:2]
    car_yaw = utils.quaternion_to_angle(Quaternion(self.current_pose[3], self.current_pose[4], self.current_pose[5], self.current_pose[6]))
    goal_yaw = get_angle_2_vectors(np.array([1,0]), current_local_goal)
    goal_yaw = utils.quaternion_to_angle(Quaternion(self.goal_pose[3], self.goal_pose[4], self.goal_pose[5], self.goal_pose[6])) + np.pi*4
    rot_mat = utils.rotation_matrix(car_yaw)
    # Apply the inverse rotation
    current_local_goal = np.array(np.dot(rot_mat, current_local_goal[:2])).squeeze() 

    delta_costs = np.zeros(self.deltas.shape[0], dtype=np.float)
    traj_depth = 0
    while (rospy.Time.now().to_sec() - start < self.compute_time and 
           traj_depth < self.rollouts.shape[1]):
        for i in xrange(self.deltas.shape[0]):
            delta_costs[i] += self.compute_cost(self.deltas[i],
                                                self.rollouts[i,traj_depth,:],
                                                current_local_goal,
                                                self.rssi_rollouts[i],
                                                msg, traj_depth, car_yaw, goal_yaw)
        traj_depth += 1

    delta_idx = np.argmin(delta_costs)
    print delta_costs.astype(int), delta_idx
    #print np.array(delta_costs).astype(int)
    costs = JointState()
    costs.header.stamp = rospy.Time.now()
    costs.position = delta_costs
    self.cost_pub.publish(costs)
    #print str(self.deltas[delta_idx]) + ' ' + str(traj_depth) + ' ' + str(delta_costs[delta_idx])
    
    ads = AckermannDriveStamped()
    ads.header.frame_id = '/map'
    ads.header.stamp = rospy.Time.now()
    ads.drive.steering_angle = self.deltas[delta_idx]
    ads.drive.speed = self.speed

    pose = PoseStamped()
    pose.header.frame_id = '/map'
    pose.pose.position.x = self.goal_pose[0]
    pose.pose.position.y = self.goal_pose[1]
    pose.pose.position.z = 0.0
    self.goal_pub.publish(pose)
    #print np.linalg.norm(self.current_pose[:2] - self.goal_pose[:2]), current_local_quaternion_goal, 1/np.square(self.rssi_rollouts)
    #print current_local_goal
    if np.linalg.norm(self.current_pose[:2] - self.goal_pose[:2]) < 0.5:
    	ads.drive.speed = 0
    #ads.drive.speed = 0
    self.cmd_pub.publish(ads)

    
'''
Apply the kinematic model to the passed pose and control
  pose: The current state of the robot [x, y, theta]
  control: The controls to be applied [v, delta, dt]
  car_length: The length of the car
Returns the resulting pose of the robot
'''
def kinematic_model_step(pose, control, car_length):
  x,y,theta = pose
  v,delta,dt = control
  
  if np.abs(delta) < 1e-2:
    dx = v*np.cos(theta)*dt
    dy = v*np.sin(theta)*dt
    dtheta = 0.0
  else:
    beta = np.arctan(0.5*np.tan(delta))
    sin2beta = np.sin(2*beta)
    dtheta = ((v/car_length) * sin2beta) * dt
    dx = (car_length/sin2beta)*(np.sin(theta+dtheta)-np.sin(theta))
    dy = (car_length/sin2beta)*(-1*np.cos(theta+dtheta)+np.cos(theta))
    
  while theta + dtheta > 2*np.pi:
    dtheta -= 2*np.pi
    
  while theta + dtheta < 2*np.pi:
    dtheta += 2*np.pi
    
  return np.array([x+dx, y+dy, theta+dtheta], dtype=np.float)
  
'''
Repeatedly apply the kinematic model to produce a trajectory for the car
  init_pose: The initial pose of the robot [x,y,theta]
  controls: A Tx3 numpy matrix where each row is of the form [v,delta,dt]
  car_length: The length of the car
Returns a Tx3 matrix where the t-th row corresponds to the robot's pose at time t+1
'''
def generate_rollout(init_pose, controls, car_length):
  T = controls.shape[0]
  rollout = np.zeros((T, 3), dtype=np.float)
  
  cur_pose = init_pose[:]

  for t in xrange(T):
    control = controls[t, :]  
    rollout[t,:] = kinematic_model_step(cur_pose, control, car_length)
    cur_pose = rollout[t,:]   
    
  return rollout
   
'''
Helper function to generate a number of kinematic car rollouts
    speed: The speed at which the car should travel
    min_delta: The minimum allowed steering angle (radians)
    max_delta: The maximum allowed steering angle (radians)
    delta_incr: The difference (in radians) between subsequent possible steering angles
    dt: The amount of time to apply a control for
    T: The number of time steps to rollout for
    car_length: The length of the car
Returns a NxTx3 numpy array that contains N rolled out trajectories, each
containing T poses. For each trajectory, the t-th element represents the [x,y,theta]
pose of the car at time t+1
'''   
def generate_mpc_rollouts(speed, min_delta, max_delta, delta_incr, dt, T, car_length):

  deltas = np.arange(min_delta, max_delta, delta_incr)
  N = deltas.shape[0]
  
  init_pose = np.array([0.0,0.0,0.0], dtype=np.float)
  
  rollouts = np.zeros((N,T,3), dtype=np.float)
  for i in xrange(N):
    controls = np.zeros((T,3), dtype=np.float)
    controls[:,0] = speed
    controls[:,1] = deltas[i]
    controls[:,2] = dt
    rollouts[i,:,:] = generate_rollout(init_pose, controls, car_length)
    
  return rollouts, deltas

    

def main():

  rospy.init_node('laser_wanderer', anonymous=True)
  speed = rospy.get_param("~speed", 1.0)
  min_delta = rospy.get_param("~min_delta", -0.34)
  max_delta = rospy.get_param("~max_delta", 0.341)
  delta_incr = rospy.get_param("~delta_incr", 0.34/3)
  dt = rospy.get_param("~dt", 0.01)
  T = rospy.get_param("~T", 300)
  car_length = rospy.get_param("car_kinematics/car_length", 0.33)
  compute_time = rospy.get_param("~compute_time", 0.09)
  laser_spread = rospy.get_param("~laser_spread", 0.314)
  laser_offset = rospy.get_param("~laser_offset", 1.0)
  
  rollouts, deltas = generate_mpc_rollouts(speed, min_delta, max_delta,
                                           delta_incr, dt, T, car_length)
                                           
  lw = LaserWanderer(rollouts, deltas, speed, compute_time, laser_spread, laser_offset)
  rospy.spin()
  

if __name__ == '__main__':
  main()
