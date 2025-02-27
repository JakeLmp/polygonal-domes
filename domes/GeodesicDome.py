import numpy as np

class GeodesicDome:
    def __init__(self, numSides, numLayers, sideLength=None, radius=None, height=None, verticalRadius=None, equator='ridge'):
        if (sideLength is None and radius is None) or (sideLength is not None and radius is not None):
            raise ValueError("Provide either sideLength or radius, but not both or neither.")
        
        self.numSides = numSides
        self.numLayers = numLayers
        
        if sideLength is not None:
            self.sideLength = sideLength
            self.radius = sideLength / (2 * np.sin(np.pi / numSides))
        else:
            self.radius = radius
            self.sideLength = 2 * self.radius * np.sin(np.pi / numSides)
        
        self.height = height if height is not None else self.radius
        self.verticalRadius = verticalRadius if verticalRadius is not None else self.radius
        self.equator = equator
        
        self.validateParameters()
        self.calculateGeometry()
    
    def validateParameters(self):
        minVertRad = 0.5 * np.sqrt(self.radius**2 + self.height**2)
        if self.verticalRadius < minVertRad:
            raise ValueError(f"verticalRadius must be larger than {minVertRad}")
    
    def calculateGeometry(self):
        self.constructivePoints()
        self.vertices = self.calculateVertices()
        self.faces = self.calculateFaces()
    
    def constructivePoints(self):
        self.top = np.array([0, 0, self.height])
        self.side = np.array([self.radius, 0, 0])
        self.centerBase = np.array([0, 0, 0])
        self.centerVertical = self.calculateVerticalCenter()
    
    def calculateVerticalCenter(self):
        x = -((self.verticalRadius - np.sqrt(self.radius**2 + self.height**2))**2) / (2 * self.radius)
        return np.array([x, 0, 0])
    
    def calculateVertices(self):
        verts = []
        angleStep = 2 * np.pi / self.numSides
        
        for i in range(self.numLayers + 1):
            theta = np.linspace(self.calculateStartTheta(), self.calculateEndTheta(), self.numLayers + 1)[i]
            layerRadius = self.verticalRadius * np.cos(theta) + self.centerVertical[0]
            layerHeight = self.verticalRadius * np.sin(theta) + self.centerVertical[2]
            
            for j in range(self.numSides):
                angle = j * angleStep
                x = layerRadius * np.cos(angle)
                y = layerRadius * np.sin(angle)
                z = layerHeight
                verts.append(np.array([x, y, z]))
        
        return verts
    
    def calculateStartTheta(self):
        return np.arcsin((self.side[2] - self.centerVertical[2]) / self.verticalRadius)
    
    def calculateEndTheta(self):
        return np.arccos((self.top[0] - self.centerVertical[0]) / self.verticalRadius)
    
    def calculateFaces(self):
        faces = []
        for i in range(self.numLayers):
            for j in range(self.numSides):
                nextJ = (j + 1) % self.numSides
                v1 = self.index(i, j)
                v2 = self.index(i + 1, j)
                v3 = self.index(i + 1, nextJ)
                v4 = self.index(i, nextJ)
                faces.append([v1, v2, v3, v4])
        return faces
    
    def index(self, layer, side):
        return layer * self.numSides + side
