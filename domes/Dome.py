import numpy as np

class Dome:
    def __init__(self,
                 nr_sides : int,
                 nr_layers : int,
                 side_length : float = None,
                 radius : float = None,
                 height : float = None,
                 vertical_radius : float = None,
                 ):
        
        # ----- MATH VARIABLES ------
        self.NR_SIDES = nr_sides
        self.NR_LAYERS = nr_layers

        if side_length is not None and radius is None:
            self.SIDE_LEN = side_length
            self.R = self.SIDE_LEN/(2*np.sin(np.pi/self.NR_SIDES))
        elif radius is not None and side_length is None:
            self.R = radius
            self.SIDE_LEN = self.R/(2*np.sin(np.pi/self.NR_SIDES))
        else:
            raise Exception(f"Either side_length or radius is required, not neither or both (given: {side_length=} {radius=})")

        # if not given, assume spherical (H=R=R_PRIME)
        self.H = height if height else self.R
                
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

        # calculate locus for centre positions
        self.m, self.c = self._calc_locus(vertical_radius is not None)

        # if not given, assume centered on base plane
        if vertical_radius is None:
            self.R_PRIME = self.R + self.c/self.m
            self.M2 = (-self.c/self.m, 0)
        else:
            self.M2, _ = self._candidate_centres()
        
        # arc parameters
        self.T1 = np.asin((self.SIDE[1] - self.M2[1])/self.R_PRIME)
        self.T2 = np.acos((self.TOP[0] - self.M2[0])/self.R_PRIME)
        self.TD = (self.T2 - self.T1)/self.NR_LAYERS

        # -----
        self._verts = self._calc_vertices()
        self._polys = self._calc_polys()

    def __repr__(self):
        return f"""
Dome(nr_sides = { self.NR_SIDES },
     nr_layers = { self.NR_LAYERS},
     side_length = { self.SIDE_LEN},
     height = {self.H},
     vertical_radius = {self.R_PRIME}
            """

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
    
    def _idx(self, i, j)-> int:
        """Index mapping between angles and cartesian points"""
        return i%self.NR_SIDES + self.NR_SIDES*j%(self.NR_SIDES*(self.NR_LAYERS+1))

    def _calc_vertices(self, angle_offset : float = 0) -> np.array:
        """Calculate vertices of the object"""
        alphas = np.linspace(angle_offset, 2*np.pi+angle_offset, self.NR_SIDES, endpoint=False)
        betas = np.linspace(self.T1, self.T2, (self.NR_LAYERS+1), endpoint=True)
        X, Y = np.meshgrid(alphas, betas)
        angles = np.array([X.flatten(), Y.flatten()]).T
        
        # L = np.array([(R_prime*np.cos(i*td + t1) + M_vert[0], 
        #                R_prime*np.sin(i*td + t1) + M_vert[1]) for i in range(NR_LAYERS+1)])

        xyz = np.zeros((angles.shape[0], 3))
        xyz[:, 0] = (self.R - self.R_PRIME*(1 - np.cos(angles[:,1])))*np.cos(angles[:,0])
        xyz[:, 1] = (self.R - self.R_PRIME*(1 - np.cos(angles[:,1])))*np.sin(angles[:,0])
        xyz[:, 2] = self.R_PRIME*np.sin(angles[:,1])

        return xyz

    @property
    def vertices(self,):
        return self._verts
    
    def _calc_polys(self,):
        # define polygon faces for the 3D plotter to work with
        polys = np.zeros((self.NR_SIDES*self.NR_LAYERS, 5, 3))

        k = 0
        for j in range(self.NR_LAYERS):
            for i in range(self.NR_SIDES):
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
    
if __name__ == '__main__':
    pass