import numpy as np
from random import sample

"""
Compute H.

Inputs:
p1, p2: N × 2 matrices specifying corresponding point locations by two cameras

Output:
H: 3 × 3 homography matrix that best matches the linear equations
"""
def compute_H(p1, p2):
    N = len(p1)
    A = np.array([[0,0,0,0,0,0,0,0,0]])
    
    for i in range(N):
        row1 = np.array([p1[i][0], p1[i][1], 1, 0, 0, 0, -p1[i][0]*p2[i][0], -p1[i][1]*p2[i][0], -p2[i][0]]) 
        row2 = np.array([0, 0, 0, p1[i][0], p1[i][1], 1, -p1[i][0]*p2[i][1], -p1[i][1]*p2[i][1], -p2[i][1]]) 
        rows = np.vstack((row1, row2))
        A = np.vstack((A, rows))

    A = np.delete(A, 0, axis=0)

    c1, c2 = np.linalg.eig(A.T @ A)

    h = c2[np.argmin(c1)]

    H = h.reshape(3,3)

    ##### YOUR CODE HERE #####

    ##########################

    return H

def train_model(v1, v2):
    return compute_H(v1, v2)

"""
Compute homographies automatically between two images using RANSAC (Random Sample Convention) algorithm.

Function Inputs:
locs1, locs2: N × 2 matrices specifying point locations in each of the images
matches: N × 2 matrix specifying matches between locs1 and locs2

Algorithm Inputs:
n_iter: the number of iterations to run RANSAC for
tol: the tolerance value for considering a point to be an inlier

Output:
bestH: the homography model with the most inliers found during RANSAC

PARTIAL CREDITS:
- Find model for randomly selected points (10 pts)
- Extend model to all inliers of model (15 pts)
- Iterate correctly to get best-fitting H (15 pts)

Hint:
1. You can use "float('inf')" for a really big number.
2. You can use the given "match" function to get p1 and p2.
3. The model here is the compute_H you implement above.
"""
def ransac_H(matches, locs1, locs2, n_iter, tol):
    N = len(matches) # length of locs1, locs2, and matches
    iter = 0
    fit_threshold = 3 # must be between 0 and N. Feel free to adjust.
    bestFit = None
    bestErr = 999999999999999999

    for iter in range(n_iter):
        thisErr = 0

        indexes = sample(range(0, N), 4)
        pairs = matches[indexes]
        maybeInliers = match(locs1, locs2, pairs)

        maybeOutlierMatches = np.delete(matches, indexes, axis=0)
        maybeOutliers = match(locs1, locs2, maybeOutlierMatches)
        maybeModel = train_model(maybeInliers[0], maybeInliers[1])
        alsoInliers = []

        for points in maybeOutliers:
            maybeErr = compute_error(points[0], points[1], maybeModel)
            if maybeErr < tol:
                thisErr += maybeErr
                alsoInliers.append(points)
        
        if alsoInliers[0].shape[0] > fit_threshold:
            betterModel = maybeModel

            for maybeInlier in maybeInliers:
                thisErr += compute_error(maybeInlier[0], maybeInlier[1], betterModel)

            if thisErr < bestErr:
                bestFit = betterModel
                bestErr = thisErr

    return bestFit

def match(locs1, locs2, pairs):
    N = len(pairs)
    p1 = p2 = np.zeros((N, 2))
    for i in range(len(pairs)):
        p1[i] = locs1[pairs[i][0]]
        p2[i] = locs2[pairs[i][1]]
    return (p1, p2)

def compute_error(loc1, loc2, H):
    return np.linalg.norm(np.matmul(H, np.append(loc2, 1)) - np.append(loc1, 1)) # TODO: Find this way?