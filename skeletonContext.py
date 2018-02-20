from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment as lsa


def skeletonTangentEstimate(edgeList, landMarkPoints=4):
    # edgeLength = [len(x) for x in edgeList]
    # mainSkeletonList = edgeList[np.argmax(edgeLength)]
    estimatedTangent = np.array([])
    skeletonPointsList = []
    for branchPoints in edgeList:
        numPoints = len(branchPoints)
        skeletonPointsList.extend(branchPoints)
        estimatedTangentBranch = np.zeros(numPoints)
        for i, point in enumerate(branchPoints):
            if i <= landMarkPoints:
                startPoint = np.array(branchPoints[0])
            else:
                startPoint = np.array(branchPoints[i - landMarkPoints])
            if i >= numPoints - landMarkPoints:
                endPoint = np.array(branchPoints[-1])
            else:
                endPoint = np.array(branchPoints[i + landMarkPoints])
            vector = endPoint - startPoint
            tan = vector[0] / vector[1] if vector[1] != 0 else np.inf
            estimatedTangentBranch[i] = np.arctan(tan)
        estimatedTangent = np.concatenate((estimatedTangent, estimatedTangentBranch))
    return estimatedTangent, skeletonPointsList


def mat2gray(img, minRange=0, maxRange=1):
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
        # Convert matrix to grayscale with the defined range
    minImg = np.min(img)
    maxImg = np.max(img)
    return (img - minImg) * (maxRange - minRange) / (maxImg - minImg) + minRange


def bdry_extract(skeleton, edgeList):
    t = np.zeros(len(edgeList))
    G2, G1 = np.gradient(skeleton)
    for i, point in enumerate(edgeList):
        t[i] = np.arctan2(G2[point[0], point[1]], G1[point[0], point[1]]) + np.pi / 2
    return t


def dist2(x, c):
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    if dimx != dimc:
        raise ValueError('Dimensions mismatch!')
    d2 = (np.dot(np.ones((ncenters, 1)), np.sum(np.square(x).T, 0, keepdims=True))).T + np.dot(np.ones((ndata, 1)),
                                                                                               np.sum(np.square(c).T, 0,
                                                                                                      keepdims=True)) - 2 * np.dot(
        x, c.T)
    return d2


def get_samples(edgeList, t, tangent, nsamp, k=3):
    '''Using Jitendras sampling method'''
    edgeListi = np.array(edgeList)
    N = len(edgeList)
    sortInd = np.arange(N)
    Nstart = min(k * nsamp, N)

    ind0 = np.random.permutation(N)
    ind0 = ind0[0:Nstart]

    edgeListi = edgeListi[ind0, :]
    ti = t[ind0]
    tangenti = tangent[ind0]
    sortIndi = sortInd[ind0]

    d2 = dist2(edgeListi, edgeListi)
    diag = np.zeros((Nstart, Nstart))
    np.fill_diagonal(diag, np.inf)
    d2 += diag

    s = 1

    while s:
        # Find Closest pair
        cp = np.argwhere(d2 == np.min(d2))
        cp = cp[0, :]
        # Remove one of the points
        edgeListi = np.delete(edgeListi, cp[1], 0)
        ti = np.delete(ti, cp[1], 0)
        tagenti = np.delete(tangenti, cp[1], 0)
        sortIndi = np.delete(sortIndi, cp[1], 0)
        d2 = np.delete(d2, cp[1], 0)
        d2 = np.delete(d2, cp[1], 1)
        if d2.shape[0] == nsamp:
            s = 0
    order = np.argsort(sortIndi)
    edgeListi = edgeListi[order]
    ti = ti[order]
    tangenti = tangenti[order]
    return edgeListi, ti, tangenti


def skeletonContext(Bsamp, Tsamp, nbins_theta, nbins_r, r_inner, r_outer, outVec, meanDistance=None):
    nsamp = Bsamp.shape[1]
    inVec = outVec == 0

    # Compute r and theta arrays
    rArray = np.sqrt(dist2(Bsamp.T, Bsamp.T))

    thetaArrayAbs = np.arctan2(
        np.tile(Bsamp[1, :][:, np.newaxis], nsamp) - np.tile(Bsamp[1, :][np.newaxis, :], (nsamp, 1)),
        np.tile(Bsamp[0, :][:, np.newaxis], nsamp) - np.tile(Bsamp[0, :][np.newaxis, :], (nsamp, 1)))
    thetaArray = thetaArrayAbs - np.tile(Tsamp[:, np.newaxis], nsamp)

    # Compute mean distance for normalization
    if not meanDistance:
        tmp = rArray[inVec, :]
        tmp = tmp[:, inVec]
        meanDistance = np.mean(tmp)
    rArrayNorm = rArray / meanDistance

    # Create LogSpace
    rBinEdges = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r)
    rArrayBin = np.zeros((nsamp, nsamp))

    for rbin in rBinEdges:
        rArrayBin += (rArrayNorm < rbin).astype('int')
    # Indicate points inside outer boundry
    insdidePoints = rArrayBin > 0

    thetaArray = thetaArray % (2 * np.pi)
    thetaArrayBin = 1 + np.floor(thetaArray / (2 * np.pi / nbins_theta))

    nbins = nbins_r * nbins_theta

    pointHistogram = np.zeros((nsamp, nbins))
    for i in range(nsamp):
        insdidePointsi = insdidePoints[i, :] & inVec
        selectedPointsRBin = rArrayBin[i, insdidePointsi] - 1
        selectedPointsThetaBin = thetaArrayBin[i, insdidePointsi] - 1
        data = np.ones(selectedPointsRBin.shape)
        sparseMat = csr_matrix((data, (selectedPointsThetaBin, selectedPointsRBin)),
                               shape=(nbins_theta, nbins_r)).toarray()
        pointHistogram[i, :] = sparseMat.T.reshape(-1)
    return pointHistogram, meanDistance


def HistCost(SC1, SC2):
    nsamp1, nbins = SC1.shape
    nsamp2, _ = SC2.shape
    eps = np.finfo(float).eps
    SC1n = SC1 / (np.tile(np.sum(SC1, 1) + eps, (nbins, 1)).T)
    SC2n = SC2 / (np.tile(np.sum(SC2, 1) + eps, (nbins, 1)).T)

    SC1Temp = np.tile(SC1n.reshape(nsamp1, 1, nbins), [1, nsamp2, 1])
    SC2Temp = np.tile(SC2n.reshape(1, nsamp2, nbins), [nsamp1, 1, 1])

    HistCost = 0.5 * np.sum(pow((SC1Temp - SC2Temp), 2) / (SC1Temp + SC2Temp + eps), 2)
    return HistCost


def hungarian(A):
    B = A.T
    rows, cols = lsa(B)
    return cols, sum(B[rows, cols])


def bookstien(X, Y, beta_k=None):
    '''Bookstien PAMI 89'''

    N = X.shape[0]

    if N != Y.shape[0]:
        raise ValueError(' Number of points must be equal')
    rX = dist2(X, X)

    # add identity matrix to rX to make zero on diagonal
    K = rX * np.log(rX + np.eye(N))
    P = np.array(np.bmat([np.ones((N, 1)), X]))
    L = np.array(np.bmat([[K, P], [P.T, np.zeros((3, 3))]]))
    V = np.array(np.bmat([Y.T, np.zeros((2, 3))]))

    # Check if regularization parameter provided
    if beta_k:
        L[0:N, 0:N] = L[0:N, 0:N] + beta_k * np.eye(N)

    invL = np.linalg.inv(L)

    c = np.dot(invL, V.T)
    cx = c[:, 0]
    cy = c[:, 1]

    Q = np.dot(np.dot(c[0:N, :].T, K), c[0:N, :])
    E = np.mean(np.diag(Q))
    return cx, cy, E, L


def SC_plot(SC, nbins_theta, nbins_r, r_inner, r_outer, N=20):
    '''Plotting polar histogram of skeleton context for each point'''
    import matplotlib.pyplot as plt
    if len(SC) != nbins_theta * nbins_r:
        raise ValueError('dimension mismatch, check the number of bins provided')
    SC_mat = SC.reshape(nbins_theta, nbins_r)
    rbins = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r)
    thetabins = np.linspace(0, 2 * np.pi, nbins_theta, endpoint=False)
    ranges2plot = np.argwhere(SC_mat)
    rplot = np.array([])
    thetaplot = np.array([])
    colors = np.array([])
    for rg in ranges2plot:
        rstart = rbins[rg[1] - 1] if rg[1] > 0 else 0
        rend = rbins[rg[1]]
        thetastart = thetabins[rg[0]]
        thetaend = thetabins[rg[0] + 1] if rg[0] < nbins_theta - 1 else 2 * np.pi
        r = np.linspace(rstart + 0.05, rend - 0.05, N)
        theta = np.linspace(thetastart + 0.05, thetaend - 0.05, N)
        rv, thetav = np.meshgrid(r, theta)
        rv = np.reshape(rv.T, -1)
        thetav = np.reshape(thetav.T, -1)
        c = SC_mat[rg[0], rg[1]] * np.ones(len(rv))
        rplot = np.concatenate((rplot, rv))
        thetaplot = np.concatenate((thetaplot, thetav))
        colors = np.concatenate((colors, c))

    area = 2
    ax = plt.subplot(111, projection='polar')
    ax.scatter(thetaplot, rplot, c=colors, cmap='hot_r')
    ax.set_yticks(rbins)
    ax.set_xticks(thetabins)
    plt.show()
    return
