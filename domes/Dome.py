import numpy as np

def rotation_matrix_x(theta):
    return np.array([
        [1, 0,              0            ],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

def rotation_matrix_y(theta):
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_matrix_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def rotation_matrix(theta_x, theta_y, theta_z):
    return rotation_matrix_z(theta_z) @ rotation_matrix_y(theta_y) @ rotation_matrix_x(theta_x)

class Dome:
    def __init__(self,
                 nr_sides : int,
                 nr_layers : int,
                 side_length : float = None,
                 radius : float = None,
                 height : float = None,
                 vertical_radius : float = None,
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

        Raises:
            Exception: either side_length or radius are required, not neither or both.
            ValueError: vertical_radius must be larger than sqrt(radius^2/4 + height^2)
        """
        
        # ----- MATH ATTRIBUTES ------
        self.NR_SIDES = nr_sides
        self.NR_LAYERS = nr_layers

        if side_length is not None and radius is None:
            self.SIDE_LEN = side_length
            self.R = self.SIDE_LEN/(2*np.sin(np.pi/self.NR_SIDES))
        elif radius is not None and side_length is None:
            self.R = radius
            self.SIDE_LEN = 2*self.R*np.sin(np.pi/self.NR_SIDES)
        else:
            raise Exception(f"Either side_length or radius is required, not neither or both (given: {side_length=} {radius=})")

        # if not given, assume spherical (H=R=R_PRIME)
        self.H = height if height is not None else self.R
                
        # centre of base
        self.M1 = (0, 0)
        
        H, R, M1 = self.H, self.R, self.M1
        self.TOP = (M1[0], 
                    M1[1] + H)
        self.SIDE = (M1[0] + R, 
                     M1[1])
        self.S = ((self.TOP[0] + self.SIDE[0])/2, 
                  (self.TOP[1] + self.SIDE[1])/2)

        # if not given, assign value later. For now we just need it to exist
        self.R_PRIME = vertical_radius if vertical_radius else None

        if self.R_PRIME is not None:
            min_vert_rad = np.sqrt(self.R**2/4 + self.H**2/4)
            if self.R_PRIME < min_vert_rad:
                raise ValueError(f"vertical_radius must be larger than sqrt(radius^2/4 + height^2) = {min_vert_rad}")

        # calculate locus for centre positions
        self.m, self.c = self._calc_locus(vertical_radius is not None)

        # if not given, assume centered on base plane
        if vertical_radius is None:
            self.R_PRIME = self.R + self.c/self.m
            self.M2 = (-self.c/self.m, 0)
        else:
            self.M2, _ = self._candidate_centres(RP=self.R_PRIME)
        
        # arc parameters
        self.T1 = np.asin((self.SIDE[1] - self.M2[1])/self.R_PRIME)
        self.T2 = np.acos((self.TOP[0] - self.M2[0])/self.R_PRIME)
        self.TD = (self.T2 - self.T1)/self.NR_LAYERS

        # ----- RESULTING ATTRIBUTES -----
        self._verts = self._calc_vertices()
        self._polys = self._calc_polys()

    def __repr__(self,):
        return f"""
Dome(nr_sides = { self.NR_SIDES },
     nr_layers = { self.NR_LAYERS},
     side_length = { self.SIDE_LEN},
     height = {self.H},
     vertical_radius = {self.R_PRIME})
            """
    
    def __str__(self,):
        return f"""
Dome 
    Nr of sides : \t{self.NR_SIDES}
    Nr of layers : \t{self.NR_LAYERS}
    Side length : \t{self.SIDE_LEN}
    Vert. side length : {2*self.R_PRIME*np.sin(0.5*self.TD)}
    Height : \t\t{self.H}
    Equator radius : \t{self.R}
    Vertical radius : \t{self.R_PRIME}
        """

    @property
    def constructive_points(self,) -> tuple:
        """
        Points used during construction of object:
            1. Top
            2. Point on equator
            3. Centre point
            4. Centre of circle defining vertical curvature
        """
        return (self.TOP, self.SIDE, self.M1, self.M2)

    def _candidate_centres(self, 
                           A : tuple[float, float] = None, 
                           B : tuple[float, float] = None,
                           RP : float = None,
                           ) -> tuple[tuple[float, float], 
                                      tuple[float, float]]:
        if A is None:
            A = self.TOP
        if B is None:
            B = self.SIDE
        
        d = np.sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2)
        l = d/2
        h = np.sqrt(d**2 - l**2) if RP is None \
            else np.sqrt(RP**2 - l**2)

        a = (l/d)*(B[0] - A[0]) + (h/d)*(B[1] - A[1]) + A[0]
        b = (l/d)*(B[1] - A[1]) - (h/d)*(B[0] - A[0]) + A[1]

        MA = (a, b)

        a = (l/d)*(B[0] - A[0]) - (h/d)*(B[1] - A[1]) + A[0]
        b = (l/d)*(B[1] - A[1]) + (h/d)*(B[0] - A[0]) + A[1]

        MB = (a, b)

        return MA, MB

    def _calc_locus(self, 
                    RP_given : bool = True,
                    ) -> tuple[float, float]:
        MA, MB = self._candidate_centres(RP = self.R_PRIME if RP_given else None)

        # locus is a line, return coefficients
        m = (MB[1] - MA[1])/(MB[0] - MA[0])
        c = MA[1] - m*MA[0]

        return m, c

    def _idx(self, i: int, j: int) -> int:
        """Index mapping used in creation of faces"""
        return i%(self.NR_LAYERS+1) + j*(self.NR_LAYERS+1)%((self.NR_LAYERS+1)*self.NR_SIDES)
    
    def _calc_vertices(self, angle_offset : float = 0) -> np.array:
        """Calculate vertices of the object"""
        alphas = np.linspace(angle_offset, 2*np.pi+angle_offset, self.NR_SIDES, endpoint=False)
        betas = np.linspace(self.T1, self.T2, (self.NR_LAYERS+1), endpoint=True)
        X, Y = np.meshgrid(alphas, betas)
        angles = np.array([X.flatten(), Y.flatten()]).T
        
        # single rib of vertical vertices
        L = np.stack((self.R_PRIME*np.cos(betas) + self.M2[0],     # x
                      np.zeros_like(betas),                        # y
                      self.R_PRIME*np.sin(betas) + self.M2[1]),    # z
                     axis = 1)
        
        # array to store vertices in
        xyz = np.empty(shape=(0,3))
                
        # rotate rib for every angle in base polygon
        for i, a in enumerate(alphas):
            rotated = np.matvec(rotation_matrix_z(a), L)
            xyz = np.concat((xyz, rotated),
                            axis=0)

        return xyz

    @property
    def vertices(self,):
        return self._verts
    
    def _calc_polys(self,):
        polys = np.zeros((self.NR_SIDES*self.NR_LAYERS, 5, 3))

        k = 0
        for i in range(self.NR_LAYERS):
            for j in range(self.NR_SIDES):
                polys[k, 0, :] = self._verts[self._idx(i, j), :]
                polys[k, 1, :] = self._verts[self._idx(i+1, j), :]
                polys[k, 2, :] = self._verts[self._idx(i+1, j+1), :]
                polys[k, 3, :] = self._verts[self._idx(i, j+1), :]
                polys[k, 4, :] = polys[k, 0, :] # add first vertex again to close the loop
                
                k += 1
        
        return polys
    
    @property
    def faces(self,):
        return self._polys
    
    def rotate_by_angle(self, angle):
        self._verts = self._calc_vertices(angle)
        self._polys = self._calc_polys()

    # TODO: fix for vertical_radius != None
    def pieces_template_2D(self,):
        S = 2*self.R_PRIME*np.sin(0.5*self.TD)  # vertical polygon side length

        polys = np.zeros((self.NR_LAYERS, 5, 2))
        for j in range(self.NR_LAYERS):
            # working radii
            rad_1 = self.R - self.R_PRIME*(1 - np.cos(j*self.TD))
            rad_2 = self.R - self.R_PRIME*(1 - np.cos((j+1)*self.TD))

            # working side lenghts
            side_len_1 = self.SIDE_LEN*rad_1/self.R
            side_len_2 = self.SIDE_LEN*rad_2/self.R

            # working apothema
            apo_1 = j*S
            apo_2 = (j+1)*S
            
            # piece
            polys[j,:,:] = np.array([[apo_1, -0.5*side_len_1],
                                     [apo_2, -0.5*side_len_2],
                                     [apo_2,  0.5*side_len_2],
                                     [apo_1,  0.5*side_len_1],
                                     [apo_1, -0.5*side_len_1]])
        return polys


        
    
if __name__ == '__main__':
    pass