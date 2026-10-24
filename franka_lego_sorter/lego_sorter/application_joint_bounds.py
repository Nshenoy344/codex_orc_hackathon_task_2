"""Application wrist-branch restrictions after a measured robot self-avoidance reflex.

These bounds do not modify Franka thresholds and do not certify self-collision
clearance. A fresh plan is required after operator recovery/repositioning.
"""
import numpy as np
from scipy.interpolate import BPoly

APPLICATION_WRIST_BOUNDS = {'fr3_joint5': (-1.5, 1.5), 'fr3_joint6': (None, 3.0)}


def validate_application_joint_bounds(trajectory):
    jt=trajectory.joint_trajectory
    names=list(jt.joint_names);points=list(jt.points)
    if not points:raise ValueError('Application joint guard requires a nonempty trajectory')
    values=np.asarray([p.positions for p in points],float)
    if values.shape!=(len(points),len(names)) or not np.isfinite(values).all():
        raise ValueError('Malformed trajectory in application joint guard')
    if len(points)>1:
        times=np.array([p.time_from_start.sec+p.time_from_start.nanosec/1e9 for p in points])
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Invalid trajectory times in application joint guard')
        derivatives=[]
        for p in points:
            d=[np.asarray(p.positions,float)]
            if len(p.velocities)==len(names):
                d.append(np.asarray(p.velocities,float))
                if len(p.accelerations)==len(names):d.append(np.asarray(p.accelerations,float))
            if not all(np.isfinite(v).all()for v in d):raise ValueError('Nonfinite trajectory derivatives')
            derivatives.append(d)
        curve=BPoly.from_derivatives(times,derivatives)
        values=curve(np.unique(np.r_[times,np.linspace(times[0],times[-1],2001)]))
    report={}
    for name,(lower,upper) in APPLICATION_WRIST_BOUNDS.items():
        if name not in names:continue
        v=values[:,names.index(name)];lo,hi=float(v.min()),float(v.max())
        report[name]={'minimum_rad':lo,'maximum_rad':hi,'application_lower_rad':lower,'application_upper_rad':upper}
        if (lower is not None and lo<lower-1e-9)or hi>upper+1e-9:
            raise ValueError(f'Application wrist branch restriction violated: {name} range[{lo:.6f},{hi:.6f}], permitted[{lower},{upper}]; fresh planning/repositioning required')
    return report
