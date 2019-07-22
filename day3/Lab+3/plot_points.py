import numpy as np
import matplotlib.pyplot as plt

# convert points from euclidian to homogeneous
def to_homog(points):
    """
    your code here
    """
    return points_homog

# convert points from homogeneous to euclidian
def from_homog(points_homog):
    """
    your code here
    """
    return points

# project 3D euclidian points to 2D euclidian
def project_points(P_int, P_ext, pts):
    """
    your code here
    """
    # return the 2d euclidiean points
    pts_2d = np.zeros([2, 1])
    return pts_2d

def camera1():
    """
    replace with your code
    """
    P_int_proj = np.eye(3, 4)
    P_ext = np.eye(4, 4)
    return P_int_proj, P_ext

def camera2():
    """
    replace with your code
    """
    P_int_proj = np.eye(3, 4)
    P_ext = np.eye(4, 4)
    return P_int_proj, P_ext

def camera3():
    """
    replace with your code
    """
    P_int_proj = np.eye(3, 4)
    P_ext = np.eye(4, 4)
    return P_int_proj, P_ext

def camera4():
    """
    replace with your code
    """
    P_int_proj = np.eye(3, 4)
    P_ext = np.eye(4, 4)
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
        plt.subplot(1, 2, 1)
        plot_points(project_points(P_int_proj, P_ext, points), title='Camera %d Projective' %(i + 1), axis=[-0.6, 0.6, -0.6, 0.6])
        plt.show()
        
main()
