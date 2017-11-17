import numpy as np
from scipy import ndimage



'''Functions needed'''


def mat2gray(img, minRange=0, maxRange=1):
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
        # Convert matrix to grayscale with the defined range
    minImg = np.min(img)
    maxImg = np.max(img)
    return (img - minImg) * (maxRange - minRange) / (maxImg - minImg) + minRange


def removeEmpty(l):
    """Remove empty lists in a nested list"""
    return list(
        filter(lambda x: not isinstance(x, list) or x, (removeEmpty(x) if isinstance(x, list) else x for x in l)))


def gkern(kernLen, sigma=5):
    """Returns a 2D Gaussian kernel array."""

    xv, yv = np.meshgrid(range(-kernLen, kernLen + 1), range(-kernLen, kernLen + 1), sparse=False, indexing='xy')
    return np.exp(-(xv * xv + yv * yv) / (2 * pow(sigma, 2)))


def graphDrawing(skeleton, edgeList, eps):
    M, N = skeleton.shape
    edgeLen = len(edgeList)
    graphImg = np.ones((M, N, 3))
    colorMat = np.hstack((np.random.uniform(0, 1, size=(edgeLen, 1)), np.random.uniform(0, 1, size=(edgeLen, 1)),
                          np.random.uniform(0, 1, size=(edgeLen, 1))))
    for i, edge in enumerate(edgeList):
        if i > 0:
            while (np.linalg.norm(colorMat[i, :] - colorMat[i - 1, :]) < eps):
                colorMat[i, :] = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)])
        c = colorMat[i,:]
        for point in edge:
            graphImg[point[0]-1:point[0]+2, point[1]-1:point[1]+2, :] = np.stack((c[0] * np.ones((3, 3)), c[1] * np.ones((3, 3)), c[2] * np.ones((3, 3)))).T
    return graphImg



def findBranchPoints(skeleton, return_image=False):
    pixelPoints = np.argwhere(skeleton)
    neighbFilter4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    neighbFilter8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    branchPoints = np.zeros((1, 2))
    branchImg = np.zeros(skeleton.shape)
    endPoints = np.zeros((1, 2))
    endImg = np.zeros(skeleton.shape)
    skeletonTemp = np.copy(skeleton)
    for selectedPoint in pixelPoints:
        pointMatrix = np.array(
            skeleton[selectedPoint[0] - 1:selectedPoint[0] + 2, selectedPoint[1] - 1:selectedPoint[1] + 2], copy=True)
        pointMatrix[1, 1] = 0
        verticeNumber = np.count_nonzero(pointMatrix)
        edgeMap = pointMatrix * ndimage.convolve(pointMatrix, neighbFilter4, mode='constant', cval=0.0)
        edgeNumber = np.sum(edgeMap) / 2
        euilerNumber = verticeNumber - edgeNumber
        if (euilerNumber > 2):
            branchPoints = np.vstack((branchPoints, selectedPoint))
            branchImg[selectedPoint[0], selectedPoint[1]] = 1
            skeletonTemp[selectedPoint[0], selectedPoint[1]] = np.nan
        elif ((euilerNumber == 1) & (verticeNumber < 5)):
            endPoints = np.vstack((endPoints, selectedPoint))
            endImg[selectedPoint[0], selectedPoint[1]] = 1
            skeletonTemp[selectedPoint[0], selectedPoint[1]] = -np.inf
        elif ((euilerNumber == 2) & (verticeNumber >= 4)):
            connectedTrees4, connectedTrees4Num = ndimage.label(pointMatrix, neighbFilter4)
            connectedTrees8, connectedTrees8Num = ndimage.label(pointMatrix, neighbFilter8)
            label4, verticesNumberTrees = np.unique(connectedTrees4[connectedTrees4 > 0], return_counts=True)
            cornerCondition = (np.sum(pointMatrix[0:2, 0:2]) == 3) | (np.sum(pointMatrix[1:3, 1:3]) == 3) | (
            np.sum(pointMatrix[1:3, 0:2]) == 3) | (np.sum(pointMatrix[0:2, 1:3]) == 3)
            if ((abs(verticesNumberTrees[0] - verticesNumberTrees[1]) >= 2) & cornerCondition & (
                connectedTrees8Num > 1)):
                branchPoints = np.vstack((branchPoints, selectedPoint))
                branchImg[selectedPoint[0], selectedPoint[1]] = 1
                skeletonTemp[selectedPoint[0], selectedPoint[1]] = np.nan
    branchPoints = branchPoints[1:, :].astype('int64')
    endPoints = endPoints[1:, :].astype('int64')
    if return_image:
        skeletonGraphPointsImg = np.tile(skeleton, (3, 1, 1))
        skeletonGraphPointsImg = skeletonGraphPointsImg + np.stack(
            (np.zeros(skeleton.shape), -branchImg, -branchImg)) + np.stack((-endImg, np.zeros(skeleton.shape), -endImg))
        skeletonGraphPointsImg = np.moveaxis(skeletonGraphPointsImg, 0, -1)
        return branchPoints, endPoints, skeletonTemp, skeletonGraphPointsImg
    else:
        return branchPoints, endPoints, skeletonTemp


def branchMask(searchMatrixPositive, branchsNearby):
    branchMask = ~ branchsNearby
    branchMat = np.argwhere(branchsNearby)
    # find special cases of branches
    branchMatRed = branchMat[np.where(
        np.all(branchMat == [1, 0], axis=1) | np.all(branchMat == [0, 1], axis=1) | np.all(branchMat == [2, 1],
                                                                                           axis=1) | np.all(
            branchMat == [1, 2], axis=1))]

    if (~ branchMatRed).all():
        for branch in branchMatRed:
            if branch[0] == 1:
                branchMask[[[0, 2], [branch[1], branch[1]]]] = 0
            else:
                branchMask[[[branch[0], branch[0]], [0, 2]]] = 0
    searchMatrixNew = searchMatrixPositive * branchMask
    return searchMatrixNew, branchMat

def branching(searchBranchPoint,skeletonTemp,edgeList,edgeNumber):
    branchNeighborMatrix = np.array(skeletonTemp[searchBranchPoint[0]-1:searchBranchPoint[0]+2,searchBranchPoint[1]-1:searchBranchPoint[1]+2],copy = True)
    branchNeighborMatrix[1,1] = 0
    branchNearbyBranch = np.isnan(branchNeighborMatrix)
    endNearbyBranch = np.isinf(branchNeighborMatrix)
    skeletonTemp[searchBranchPoint[0],searchBranchPoint[1]] = - edgeNumber
    with np.errstate( invalid='ignore'):
        branchNeighborMatrixPositive = branchNeighborMatrix > 0
    if (branchNearbyBranch.any()):
        branchNeighborMatrixNew, searchBranchMat2 = branchMask(branchNeighborMatrixPositive, branchNearbyBranch)
        searchBranchPoint2 = searchBranchMat2 + searchBranchPoint - [1,1]
        branchConnected4,branchConnected4Num = ndimage.label(branchNeighborMatrixNew, np.array([[0,1,0],[1,0,1],[0,1,0]]))
        if branchConnected4Num:
            labels = np.unique(branchConnected4[branchConnected4>0])
            for l in labels:
                newBranchPoints = np.argwhere(branchConnected4 == l) + searchBranchPoint - [1,1]
                edgeList.append(np.concatenate((np.array([searchBranchPoint]),newBranchPoints)).tolist())
                edgeNumberNew3 = len(edgeList)
                skeletonTemp[newBranchPoints.T.tolist()] = -edgeNumberNew3

        for branch2 in searchBranchPoint2:
            edgeList.append(np.stack((searchBranchPoint,branch2)).tolist())
            edgeNumberNew = len(edgeList)
            skeletonTemp, edgeList = branching(branch2, skeletonTemp, edgeList, edgeNumberNew)
    elif (endNearbyBranch.any()):
        endNearbyBranchPoints = np.argwhere(endNearbyBranch) + searchBranchPoint - [1,1]
        for branch2 in endNearbyBranchPoints:
            edgeList.append(np.stack((searchBranchPoint,branch2)).tolist())
            edgeNumberNew2 = len(edgeList)
            skeletonTemp[branch2.tolist()] = -edgeNumberNew2

        branchNeighborMatrixNew = np.array(branchNeighborMatrixPositive, copy=True)
        branchConnected4,branchConnected4Num = ndimage.label(branchNeighborMatrixNew, np.array([[0,1,0],[1,0,1],[0,1,0]]))
        if branchConnected4Num:
            labels = np.unique(branchConnected4[branchConnected4>0])
            for l in labels:
                newBranchPoints = np.argwhere(branchConnected4 == l) + searchBranchPoint - [1,1]
                edgeList.append(np.concatenate((np.array([searchBranchPoint]),newBranchPoints)).tolist())
                edgeNumberNew3 = len(edgeList)
                skeletonTemp[newBranchPoints.T.tolist()] = -edgeNumberNew3
    else:
        branchNeighborMatrixNew = np.array(branchNeighborMatrixPositive, copy=True)
        branchConnected4,branchConnected4Num = ndimage.label(branchNeighborMatrixNew, np.array([[0,1,0],[1,0,1],[0,1,0]]))
        if branchConnected4Num:
            labels = np.unique(branchConnected4[branchConnected4>0])
            for l in labels:
                newBranchPoints = np.argwhere(branchConnected4 == l) + searchBranchPoint - [1,1]
                edgeList.append(np.concatenate((np.array([searchBranchPoint]),newBranchPoints)).tolist())
                edgeNumberNew3 = len(edgeList)
                skeletonTemp[newBranchPoints.T.tolist()] = -edgeNumberNew3


    return skeletonTemp,edgeList


def mirrorBW(BW , t = 1):
    M,N = BW.shape
    mirrorImg = np.zeros([M+2*t,N+2*t])
    mirrorImg[t:M+t,t:N+t] = BW

    mirrorImg[0:t,t:N+t]           = np.flip(mirrorImg[t:2*t,t:N+t],0)
    mirrorImg[M+t:M+2*t,t:N+t]     = np.flip(mirrorImg[M:M+t,t:N+t],0)
    mirrorImg[t:M+t,0:t]           = np.flip(mirrorImg[t:M+t,t:2*t],1)
    mirrorImg[t:M+t,N+t:N+2*t]     = np.flip(mirrorImg[t:M+t,N:N+t],1)

    mirrorImg[0:t,0:t]             = np.flip(np.flip(mirrorImg[t:2*t,t:2*t],0),1)
    mirrorImg[M+t:M+2*t,N+t:N+2*t] = np.flip(np.flip(mirrorImg[M:M+t,N:N+t],0),1)
    mirrorImg[0:t,N+t:N+2*t]       = np.flip(np.flip(mirrorImg[t:2*t,N:N+t],0),1)
    mirrorImg[M+t:M+2*t,0:t]       = np.flip(np.flip(mirrorImg[M:M+t,t:2*t],0),1)
    return mirrorImg

def flux(delD_xn, delD_yn):
    Nx = -1/np.sqrt(2) * np.array([[-1, 0, 1],[-np.sqrt(2), 0, np.sqrt(2)],[-1, 0, 1]])
    Ny = -1/np.sqrt(2) * np.array([[-1, -np.sqrt(2), -1],[0, 0, 0],[1, np.sqrt(2), 1]])
    flux = np.zeros(delD_xn.shape)
    flux.fill(np.nan)
    nonNanPix = np.argwhere(np.invert(np.isnan(delD_xn) | np.isnan(delD_yn)))
    for pix in nonNanPix:
        flux_x = Nx * delD_xn[pix[0]-1:pix[0]+2,pix[1]-1:pix[1]+2]
        flux_y = Ny * delD_yn[pix[0]-1:pix[0]+2,pix[1]-1:pix[1]+2]
        flux_x[1,1] = np.nan
        flux_y[1,1] = np.nan
        flux_temp = flux_x + flux_y
        flux[pix[0]-1:pix[0]+2,pix[1]-1:pix[1]+2] = np.nansum(flux_temp)/np.count_nonzero(~np.isnan(flux_temp))
    return flux

'''Computing the graph of skeleton'''

def skeleton2Graph(skeleton, fluxMap, sigma = 5):
    branchPoints, endPoints, skeletonTemp = findBranchPoints(skeleton)
    skeletonTemp1 = np.copy(skeletonTemp)
    vertices = np.concatenate((endPoints, branchPoints))
    edgeList = [[]]

    # Initialization
    edgeList[0].append([endPoints[0, 0], endPoints[0, 1]])
    skeletonTemp[endPoints[0, 0], endPoints[0, 1]] = -1
    edgeNumber = 1
    pointNumber = 0
    adjacencyMatrix = np.zeros((len(vertices), len(vertices)))
    verticesProperties = [[] for _ in range(len(vertices))]
    verticesProperties2 = [[] for _ in range(len(vertices))]

    while (edgeNumber <= len(edgeList)):

        if (pointNumber > len(edgeList[edgeNumber - 1]) - 1):
            if ((not edgeList[edgeNumber - 1]) & (pointNumber == 1)):
                edgeNumber += 1
                continue
            searchPointValue = skeletonTemp[searchPoint[0], searchPoint[1]]
            newEdgeInd = np.argwhere((searchMatrix != -1) & (searchMatrix < 0) & (searchMatrix > -np.inf))
            if len(newEdgeInd) == 1:
                edgeNumber2 = -searchMatrix[newEdgeInd]
                edgePoints2 = list(np.flipud(edgeList[edgeNumber2 - 1]))
                edgeList[edgeNumber - 1].extend(edgePoints2)
                edgeList[edgeNumber2 - 1] = []
            edgeNumber += 1
            pointNumber = 1
            continue

        searchPoint = edgeList[edgeNumber - 1][pointNumber]
        if ((pointNumber == 1) & ((np.isnan(skeletonTemp1[searchPoint[0], searchPoint[1]])) | (
        np.isinf(skeletonTemp1[searchPoint[0], searchPoint[1]])))):
            edgeNumber += 1
            continue

        searchMatrix = np.array(
            skeletonTemp[searchPoint[0] - 1:searchPoint[0] + 2, searchPoint[1] - 1:searchPoint[1] + 2], copy=True)
        searchMatrix[1, 1] = 0
        vec2Branch = np.array(searchPoint) - np.array(edgeList[edgeNumber - 1][0])
        if (np.linalg.norm(vec2Branch) < 1.5):
            branchOldInd = [1, 1] - vec2Branch
            searchMatrix[branchOldInd[0], branchOldInd[1]] = 0

        with np.errstate(invalid='ignore'):
            searchMatrixPositive = searchMatrix > 0

        if (np.count_nonzero(searchMatrix)):
            branchsNearby = np.isnan(searchMatrix)
            endsNearby = np.isinf(searchMatrix)
            branchsEmpty = not np.count_nonzero(branchsNearby)
            endsEmpty = not np.count_nonzero(endsNearby)

            if (branchsEmpty & endsEmpty):
                edgePoints = np.argwhere(searchMatrixPositive) + searchPoint - [1, 1]
                edgeList[edgeNumber - 1].extend(edgePoints.tolist())
                skeletonTemp[edgePoints.T.tolist()] = - edgeNumber

                # New Assignment
                pointNumber += 1

            elif (not branchsEmpty):
                searchMatrixNew, branchMat = branchMask(searchMatrixPositive, branchsNearby)
                branchMat += np.array(searchPoint) - [1, 1]

                # Adding points to EdgeList while ommiting other branches points
                edgePoints = np.argwhere(searchMatrixNew) + searchPoint - [1, 1]
                edgeList[edgeNumber - 1].extend(edgePoints.tolist())
                skeletonTemp[edgePoints.T.tolist()] = - edgeNumber
                edgeList[edgeNumber - 1].append(list(branchMat[0, :]))

                for branch in branchMat:
                    if (not np.isnan(skeletonTemp[branch[0], branch[1]])):
                        continue
                    else:
                        skeletonTemp, edgeList = branching(branch, skeletonTemp, edgeList, edgeNumber)
                edgeNumber += 1
                pointNumber = 1

            elif (branchsEmpty & (not endsEmpty)):
                endPoint = np.argwhere(endsNearby) + searchPoint - [1, 1]
                edgePoints = np.argwhere(searchMatrixPositive) + searchPoint - [1, 1]
                edgeList[edgeNumber - 1].extend(edgePoints.tolist())
                edgeList[edgeNumber - 1].extend(endPoint.tolist())
                skeletonTemp[edgePoints.T.tolist()] = - edgeNumber

                edgeNumber += 1
                pointNumber = 1

        else:
            edgeNumber += 1
            pointNumber = 1

    edgeList = removeEmpty(edgeList)
    edgeLength = [len(edge) for edge in edgeList]
    maxEdgeLength = max(edgeLength)
    gaussianKernelMatrix = gkern(maxEdgeLength, sigma)
    edgeProperties = np.zeros((3, len(edgeList)))
    edgeProperties2 = np.zeros((3, len(edgeList)))

    for i, edgePoints in enumerate(edgeList):
        startInd = np.argwhere(np.all(vertices == edgePoints[0], axis=1))[0][0]
        endInd = np.argwhere(np.all(vertices == edgePoints[-1], axis=1))
        edgeProperties[2, i] = len(edgePoints)
        if (endInd):
            endInd = endInd[0][0]
            adjacencyMatrix[startInd, endInd] = i + 1
            adjacencyMatrix[endInd, startInd] = -(i + 1)
            vector2EndGaussian = np.array(edgePoints) - edgePoints[-1] + [maxEdgeLength, maxEdgeLength]
            endGaussianValue = gaussianKernelMatrix[vector2EndGaussian.T.tolist()]
            endFluxValue = fluxMap[np.array(edgePoints).T.tolist()] * endGaussianValue
            edgeProperties[1, i] = np.sum(endFluxValue[0:-1]) / (len(edgePoints) - 1)
            verticesProperties[endInd].append([edgeProperties[1, i], i])
        vector2StartGaussian = np.array(edgePoints) - edgePoints[0] + [maxEdgeLength, maxEdgeLength]
        startGaussianValue = gaussianKernelMatrix[vector2StartGaussian.T.tolist()]
        startFluxValue = fluxMap[np.array(edgePoints).T.tolist()] * startGaussianValue
        edgeProperties[0, i] = np.sum(startFluxValue[1:]) / (len(edgePoints) - 1)
        verticesProperties[startInd].append([edgeProperties[0, i], i])

    adjacencyMatrix = adjacencyMatrix.astype('int64')

    for v, vertex in enumerate(vertices):
        edgeLinkedNumber = adjacencyMatrix[v, np.where(adjacencyMatrix[v, :] != 0)[0]]
        if (~ edgeLinkedNumber.any()):
            continue
        edgeLinkedLength = np.array(edgeLength)[map(int, list(np.abs(edgeLinkedNumber) - 1))].tolist()
        searchDepth = min(edgeLinkedLength)
        for el in edgeLinkedNumber:
            if el > 0:
                edgeLinkedPoints = edgeList[int(el) - 1][1:searchDepth]
            else:
                edgeLinkedPoints = edgeList[-int(el) - 1][-searchDepth:-1]
            vector2VertexGaussian = np.array(edgeLinkedPoints) - vertex + [maxEdgeLength, maxEdgeLength]
            vertexGaussianValue = gaussianKernelMatrix[vector2VertexGaussian.T.tolist()]
            vertexFluxValue = fluxMap[np.array(edgeLinkedPoints).T.tolist()] * vertexGaussianValue
            verticesProperties2[v].append([np.sum(vertexFluxValue) / (searchDepth - 1), np.abs(el) - 1])

    for edgeInd in range(len(edgeList)):
        verticesOnEdge = np.argwhere(adjacencyMatrix == edgeInd + 1)
        if (verticesOnEdge.any()):
            edgeProperties2[0, edgeInd] = verticesProperties[verticesOnEdge[0, 0]][
                np.argwhere(np.array(verticesProperties2[verticesOnEdge[0, 0]])[:, 1] == edgeInd)[0, 0]][0]
            edgeProperties2[1, edgeInd] = verticesProperties[verticesOnEdge[0, 1]][
                np.argwhere(np.array(verticesProperties2[verticesOnEdge[0, 1]])[:, 1] == edgeInd)[0, 0]][0]
        edgeProperties2[2, edgeInd] = edgeLength[edgeInd]

    return adjacencyMatrix, edgeList,edgeProperties,edgeProperties2, verticesProperties, verticesProperties2, endPoints, branchPoints