"""Offline convex collision-hull versus axis-aligned measured scene boxes."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


def load_boxes(path):
    data=json.loads(Path(path).read_text());boxes=[]
    for obj in data['collision_objects']:
        for shape,pose in zip(obj['primitives'],obj['primitive_poses']):
            q=pose['orientation']
            if shape['type']!=1 or any(abs(q[k])>1e-9 for k in('x','y','z'))or abs(q['w']-1)>1e-9:raise ValueError('Only axis-aligned recorded primitive boxes supported')
            c=np.array([pose['position'][k]for k in('x','y','z')]);half=np.array(shape['dimensions'])/2
            boxes.append((obj['id'],c-half,c+half))
    if not boxes:raise ValueError('No environment boxes recorded')
    return boxes


def intersections(vertices,boxes):
    vertices=np.asarray(vertices);low=vertices.min(0);high=vertices.max(0);hits=[]
    for name,lo,hi in boxes:
        if np.any(high<lo)or np.any(low>hi):continue
        hull=ConvexHull(vertices).equations
        result=linprog(np.zeros(3),A_ub=hull[:,:3],b_ub=-hull[:,3],bounds=list(zip(lo,hi)),method='highs')
        if result.status!=2:hits.append(name) # Solver uncertainty also fails closed.
    return hits
