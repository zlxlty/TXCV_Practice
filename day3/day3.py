import numpy as np
import matplotlib.pyplot as plt
import math

# convert points from euclidian to homogeneous
def to_homog(points):
    points_homog = np.vstack((points, np.ones(points.shape[1])))
    return points_homog

# convert points from homogeneous to euclidian
def from_homog(points_homog):
    for i in range(points_homog.shape[1] - 1):
        points_homog[i] /= points_homog[-1]

    # print(points_homog)        
    # points_homog
    points = points_homog[:-1]
    return points

# project 3D euclidian points to 2D euclidian
def project_points(P_int, P_ext, pts):

    pts_homog = to_homog(pts)
    pts_2d_homog = P_int @ P_ext @ pts_homog
    pts_2d = from_homog(pts_2d_homog)
    return pts_2d

def camera1():
    P_int_proj = np.array([[-1,0,0,0],
                           [0,-1,0,0],
                           [0,0,1,0]])
    P_ext = np.eye(4, 4)
    return P_int_proj, P_ext

def camera2():
    P_int_proj = np.array([[-1,0,0,0],
                           [0,-1,0,0],
                           [0,0,1,0]])    
    P_ext = np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,1,1],
                      [0,0,0,1]])
    return P_int_proj, P_ext

def camera3():
    P_int_proj = np.array([[-1,0,0,0],
                           [0,-1,0,0],
                           [0,0,1,0]])    
    z_angle = math.pi/6
    y_angle = math.pi/3

    P_ext = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]]) @ np.array([[math.cos(z_angle), -math.sin(z_angle), 0, 0],[math.sin(z_angle), math.cos(z_angle), 0, 0],[0,0,1,0],[0,0,0,1]]) @ \
        np.array([[math.cos(y_angle), 0, math.sin(y_angle), 0],[0,1,0,0],[-math.sin(y_angle),0,math.cos(y_angle),0],[0,0,0,1]])

    return P_int_proj, P_ext

def camera4():
    """
    replace with your code
    """
    P_int_proj = np.array([[-5,0,0,0],
                           [0,-5,0,0],
                           [0,0,1,0]])
    z_angle = math.pi/6
    y_angle = math.pi/3
    P_ext = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,13],[0,0,0,1]]) @ np.array([[math.cos(z_angle), -math.sin(z_angle), 0, 0],[math.sin(z_angle), math.cos(z_angle), 0, 0],[0,0,1,0],[0,0,0,1]]) @ \
        np.array([[math.cos(y_angle), 0, math.sin(y_angle), 0],[0,1,0,0],[-math.sin(y_angle),0,math.cos(y_angle),0],[0,0,0,1]])
    return P_int_proj, P_ext


# test code. Do not modify

def plot_points (points, title='', style='.-r', axis=[]):
    inds = list(range(points.shape[1])) + [0]
    plt.plot(points[0, inds], points[1, inds], style)
    if title:
        plt.title(title)
    if axis:
        plt.axis('scaled')
        # plt.axis(axis)

def main():
    point1 = np.array([[-1, -0.5, 2]]).T
    point2 = np.array([[1, -0.5, 2]]).T
    point3 = np.array([[1, 0.5, 2]]).T
    point4 = np.array([[-1, 0.5, 2]]).T
    points = np.hstack((point1, point2, point3, point4))
    
    for i, camera in enumerate([camera1, camera2, camera3, camera4]):
        P_int_proj, P_ext = camera()
        plt.subplot(2, 2, i+1)
        plot_points(project_points(P_int_proj, P_ext, points), title='Camera %d Projective' %(i + 1), axis=[-0.6, 0.6, -0.6, 0.6])
    plt.show()
   
main()