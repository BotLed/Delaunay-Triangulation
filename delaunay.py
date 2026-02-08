from random import choice, randint, uniform
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, PatchCollection
import matplotlib.patches as patches
import matplotlib.cm as cm
from math import hypot, sqrt
from typing import List, Tuple
import numpy as np


class Triangle:
    def __init__(self, v1 : List[Tuple[int, int]], v2 : List[Tuple[int, int]], v3 : List[Tuple[int, int]]):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.circumcircle = self.calc_circumcircle(v1, v2, v3)
        self.edges = [tuple(sorted((self.v1,self.v2))), 
                      tuple(sorted((self.v2, self.v3))), 
                      tuple(sorted((self.v3, self.v1)))]

    def inCircle(self, p : List[Tuple[int, int]]):
        if self.circumcircle is None: # check added for when D is None
            return False
        
        px, py = p
        center_x = self.circumcircle[0][0]
        center_y = self.circumcircle[0][1] 
        radius = self.circumcircle[1]

        if (px - center_x)**2 + (py - center_y)**2 > radius**2: 
            return False 

        return True
     
    def getCircle(self):
        return self.circumcircle

    def calc_circumcircle(self, v1, v2, v3) -> Tuple[Tuple[float, float], float] | None:
        '''Calculates center and circumference of Triangle -> returns (center, radius)'''
        ax, ay = v1
        bx, by = v2
        cx, cy = v3

        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

        # colinear points
        if D == 0:
            return None

        ax2 = ax**2 + ay**2
        bx2 = bx**2 + by**2
        cx2 = cx**2 + cy**2

        # circumcenter coords
        Xc = (ax2 * (by - cy) + bx2 * (cy - ay) + cx2 * (ay - by)) / D
        Yc = (ax2 * (cx - bx) + bx2 * (ax - cx) + cx2 * (bx - ax)) / D
        center = (Xc, Yc)

        # circumcircle radius
        radius = sqrt((ax - Xc)**2 + (ay - Yc)**2)

        return (center, radius)


def generate_points(width : int, height : int, num_points : int, method : str):
    '''Generates array of points using random or uniform distribution'''
    points = []
    if method=="uniform":
        k = 10

        for i in range(num_points):
            best_p = None
            d_max = 0
            for p in range(k):
                p = (randint(0, width - 1), randint(0, height - 1))

                if len(points) == 0: # No points, use intiial point as baseline
                    points.append(p)
                    best_p = p
                    break

                d_min = float('inf')
                for x, y in points:
                    d = hypot(p[0]-x, p[1]-y)

                    if d < d_min:
                        d_min = d

                if d_min > d_max:
                    d_max = d_min
                    best_p = p

                i += 1

            points.append(best_p)
    else:
        for i in range(num_points):
            p = (randint(0, width - 1), randint(0, height -1))

            if p in points:
                continue

            points.append(p)

    #print(f"Test Case 1: {[(15, 161), (136, 172), (155, 170)]}") -> Near degenerate triangle (Do colinearity check)
    #print(f"Test Case 2: {[(0,0), (1,1), (2,2)]}")
    wfc_points = [(21, 184), (21, 184), (479, 288), (435, 13), (136, 468), (251, 98), (330, 364), (15, 387), (82, 26), (201, 249), (350, 190), (434, 379), (185, 363), (295, 495), (355, 45), (60, 287)]
    return points 


def plot_triangulation(triangulation, show_circumcircle=False, show_points=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    colormap = cm.get_cmap('RdPu')
    
    triangle_verts = []
    circle_patches = []
    all_points = []

    for t in triangulation:
        v = [t.v1, t.v2, t.v3]
        triangle_verts.append(v)
        all_points.extend(v)
        
        if show_circumcircle:
            (center_x, center_y), radius = t.getCircle()
            circle = patches.Circle((center_x, center_y), radius, 
                                    fill=False, edgecolor='red', 
                                    linestyle='--', alpha=0.3, linewidth=0.8)
            circle_patches.append(circle)
        
        face_colors = [colormap(uniform(0.3, 0.7)) for _ in range(len(triangle_verts))]

    # add triangle faces and edges
    poly_coll = PolyCollection(triangle_verts, 
                               facecolors=face_colors, 
                               edgecolors='white', 
                               alpha=0.5, 
                               zorder=1)
    ax.add_collection(poly_coll)

    # add circumcircles
    if show_circumcircle and circle_patches:
        patch_coll = PatchCollection(circle_patches, match_original=True, zorder=2)
        ax.add_collection(patch_coll)

    # add points
    if show_points:
        pts = np.array(all_points)
        unique_pts = np.unique(pts, axis=0)
        ax.scatter(unique_pts[:, 0], unique_pts[:, 1], color='black', s=20, zorder=3)

    ax.set_aspect('equal')
    ax.autoscale_view()
    #plt.title("Delaunay Triangulation")
    plt.axis('off')
    plt.savefig('transparent_plot.png', transparent=True, dpi=300)
    plt.show()


def delaunay_triangulation(width : int, height : int, num_points : int, point_gen_method="uniform", plot=False, **plot_kwargs):
    triangulation = []
    points = generate_points(width, height, num_points, point_gen_method)
    
    # super triangle
    st_v1 = (-width, -height)
    st_v2 = (2 * width, -height)
    st_v3 = (width / 2, 3 * height)

    super_triangle = Triangle(st_v1, st_v2, st_v3)
    triangulation.append(super_triangle)

    for vertex in points:
        bad_triangles = set()
        
        # find all invalid triangles
        for triangle in triangulation:
            if triangle.inCircle(vertex):
                bad_triangles.add(triangle)

        polygon = set()
        unique_edges = set()

        # find boundary of polygonal hole
        for btriangle in bad_triangles:
            for edge in btriangle.edges:
                if edge not in unique_edges:
                    unique_edges.add(edge)
                    polygon.add(edge)
                else:
                    polygon.remove(edge) # edge seen more than once = interior edge, remove
        
        # remove all bad triangles from triangulation
        triangulation = list(set(triangulation) - set(bad_triangles))

        # re-triangulate hole
        for edge in polygon:
            tri = Triangle(vertex, edge[0], edge[1])
            triangulation.append(tri)

    remove = []
    for triangle in triangulation:
        common_vertices = list(set([triangle.v1, triangle.v2, triangle.v3]) & set([st_v1, st_v2, st_v3]))

        # DEBUG
        # print("T vertices: ", set([triangle.v1, triangle.v2, triangle.v3]))
        # print("ST vertices: ", set([st_v1, st_v2, st_v3]))
        # print(common_vertices, "\n")

        if common_vertices:
            remove.append(triangle)

    triangulation = list(set(triangulation) - set(remove))

    # visualize and also violate SOLID
    if plot:
        plot_triangulation(triangulation, **plot_kwargs)

    return triangulation
