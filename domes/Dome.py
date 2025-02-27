import numpy as np
import warnings

def rotationMatrixX(theta):
    return np.array([
        [1, 0,              0            ],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

def rotationMatrixY(theta):
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotationMatrixZ(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def rotationMatrix(thetaX, thetaY, thetaZ):
    return rotationMatrixZ(thetaZ) @ rotationMatrixY(thetaY) @ rotationMatrixX(thetaX)

class Dome:
    def __init__(self,
                 numSides: int,
                 numLayers: int,
                 sideLength: float = None,
                 radius: float = None,
                 height: float = None,
                 verticalRadius: float = None,
                 equator: str = 'ridge'
                 ):
        """Create a dome-like object consisting of trapezoid faces.
        Dimensions are defined in terms of the (partial) regular polygons formed by the object in the x-y and x-z planes.

        Args:
            nr_sides (int): number of sides of base polygon (top-view)
            nr_layers (int): number of stacked polygons (side-view)
            side_length (float, optional): length of sides of base polygon. 
                                           User should provide value for either side_length or radius, not neither or both. 
                                           Defaults to None.
            radius (float, optional): circumscribed radius of base polygon. 
                                      User should provide value for either side_length or radius, not neither or both. 
                                      Defaults to None.
            height (float, optional): height of the structure, as measured from centre of base polygon to the top. 
                                      If not given, value is set equal to radius.
                                      Defaults to None.
            vertical_radius (float, optional): circumscribed radius of vertical polygon. 
                                               If not given, value is set equal to radius.
                                               Defaults to None.
            equator (str, optional): put the 'equator', i.e. the base plane, on a ridge or halfway on the bottom row of faces.
                                     Options: ['ridge', 'face'].
                                     Defaults to 'ridge'.

        Raises:
            Exception: either side_length or radius are required, not neither or both.
            ValueError: vertical_radius must be larger than 0.5*sqrt(radius^2 + height^2)
            ValueError: equator argument value not recognised
        """

        self.numSides = numSides
        self.numLayers = numLayers

        if sideLength is not None and radius is None:
            self.sideLength = sideLength
            self.radius = self.sideLength / (2 * np.sin(np.pi / self.numSides))
        elif radius is not None and sideLength is None:
            self.radius = radius
            self.sideLength = 2 * self.radius * np.sin(np.pi / self.numSides)
        else:
            raise Exception(f"Either sideLength or radius is required, not neither or both (given: {sideLength=} {radius=})")

        self.height = height if height is not None else self.radius
        
        self.centerBase = (0, 0)
        
        height, radius, centerBase = self.height, self.radius, self.centerBase
        self.top = (centerBase[0], centerBase[1] + height)
        self.side = (centerBase[0] + radius, centerBase[1])
        self.s = ((self.top[0] + self.side[0]) / 2, (self.top[1] + self.side[1]) / 2)

        self.verticalRadius = verticalRadius if verticalRadius else None

        if self.verticalRadius is not None:
            minVertRad = 0.5 * np.sqrt(self.radius**2 + self.height**2)
            if self.verticalRadius < minVertRad:
                raise ValueError(f"verticalRadius must be larger than 0.5*sqrt(radius^2 + height^2) = {minVertRad}")

        self.slope, self.intercept = self._calcLocus(verticalRadius is not None)

        if verticalRadius is None:
            self.verticalRadius = self.radius + self.intercept / self.slope
            self.centerCurve = (-self.intercept / self.slope, 0)
        else:
            self.centerCurve, _ = self._candidateCenters(rp=self.verticalRadius)
        
        match equator:
            case 'ridge':
                self.t1 = np.asin((self.side[1] - self.centerCurve[1]) / self.verticalRadius)
                self.t2 = np.acos((self.top[0] - self.centerCurve[0]) / self.verticalRadius)
                self.tDelta = (self.t2 - self.t1) / self.numLayers
            case 'face':
                if verticalRadius is not None:
                    warnings.warn("Setting equator to 'face' when also setting verticalRadius explicitly might result in unexpected behavior.")
                altNumLayers = self.numLayers * 2 - 1
                q1 = np.asin((-self.top[1] - self.centerCurve[1]) / self.verticalRadius)
                q2 = np.acos((self.top[0] - self.centerCurve[0]) / self.verticalRadius)
                q = (q2 - q1) / altNumLayers
                self.t1 = np.asin((self.side[1] - self.centerCurve[1]) / self.verticalRadius) - 0.5 * q
                self.t2 = np.acos((self.top[0] - self.centerCurve[0]) / self.verticalRadius)
                self.tDelta = (self.t2 - self.t1) / self.numLayers
            case _:
                raise ValueError(f"Equator argument value not recognized: '{equator}'.")

        self._vertices = self._calcVertices()
        self._faces = self._calcFaces()

    def _calcVertices(self, angleOffset: float = 0):
        alphas = np.linspace(angleOffset, 2 * np.pi + angleOffset, self.numSides, endpoint=False)
        betas = np.linspace(self.t1, self.t2, self.numLayers + 1, endpoint=True)
        L = np.stack((self.verticalRadius * np.cos(betas) + self.centerCurve[0],
                      np.zeros_like(betas),
                      self.verticalRadius * np.sin(betas) + self.centerCurve[1]),
                     axis=1)
        xyz = np.empty(shape=(0, 3))
        for a in alphas:
            rotated = np.matmul(rotationMatrixZ(a), L.T).T
            xyz = np.vstack((xyz, rotated))
        return xyz

    @property
    def vertices(self):
        return self._vertices

    def _calcFaces(self):
        faces = np.zeros((self.numSides * self.numLayers, 5, 3))
        k = 0
        for i in range(self.numLayers):
            for j in range(self.numSides):
                faces[k, 0, :] = self._vertices[self._idx(i, j), :]
                faces[k, 1, :] = self._vertices[self._idx(i + 1, j), :]
                faces[k, 2, :] = self._vertices[self._idx(i + 1, j + 1), :]
                faces[k, 3, :] = self._vertices[self._idx(i, j + 1), :]
                faces[k, 4, :] = faces[k, 0, :]
                k += 1
        return faces

    @property
    def faces(self):
        return self._faces

    def rotateByAngle(self, angle):
        self._vertices = self._calcVertices(angle)
        self._faces = self._calcFaces()

if __name__ == '__main__':
    pass
