"""Offline bounded stud-only trial geometry; never sends robot/hand commands.

Surface offsets must be independently measured in a shared base-frame normal.
A visible upper envelope is not a measured body top. All heights include stated
uncertainty and full open-jaw toe-corner tilt, rather than tip-center shortcuts.
"""
import numpy as np


def generate_trial(T_base_tcp, normal_base, upper_offset_m, upper_uncertainty_m,
                   body_upper_offset_m=None, body_uncertainty_m=None,
                   body_reference_verified=False, opening_m=.0168,
                   toe_extension_m=.0095, pad_length_m=.020,
                   jaw_thickness_m=.0135, usable_insertion_m=.002,
                   initial_clearance_m=.001, step_m=.0002, max_descent_m=.0018,
                   isolated_prototype_only=False, lower_neighbors_verified=False,
                   lower_neighbor_upper_offset_m=None):
    T=np.asarray(T_base_tcp,float);n=np.asarray(normal_base,float)
    if T.shape!=(4,4) or n.shape!=(3,) or not np.isfinite(T).all() or not np.isfinite(n).all():
        raise ValueError('Finite actual pose and measured normal required')
    n=n/np.linalg.norm(n)
    if n[2]<.95:raise ValueError('Trial requires near-horizontal upward surface normal')
    values=[upper_offset_m,upper_uncertainty_m,opening_m,toe_extension_m,pad_length_m,jaw_thickness_m,usable_insertion_m,initial_clearance_m,step_m,max_descent_m]
    if not np.isfinite(values).all() or min(values[1:])<=0 or max_descent_m>.0018 or step_m>.0002 or usable_insertion_m>.002:
        raise ValueError('Explicit finite conservative trial bounds required')
    toes=np.array([[x*pad_length_m/2,s*(opening_m/2+d),toe_extension_m]
                   for x in(-1,1)for s in(-1,1)for d in(0,jaw_thickness_m)])
    heights=(toes@T[:3,:3].T)@n
    tilt=float(np.ptp(heights))
    start=T.copy();start[2,3]=(upper_offset_m+upper_uncertainty_m+initial_clearance_m-min(heights)-n[:2]@T[:2,3])/n[2]
    result=dict(kind='offline_shallow_stud_trial_only',action_sent=False,production_qualified=False,
                trial_ready=False,opening_m=opening_m,grasp_goal_m=.0128,epsilon_outer_m=.0008,
                measured_body_width_reference_m=.01535,body_width_excluded_from_success=True,
                full_toe_normal_height_span_m=tilt,T_base_tcp_initial_above_upper=start.tolist(),
                requested_step_m=step_m,requested_max_descent_m=max_descent_m,steps=[],
                required_checks=['Independent dense arm/Hand/camera/cloud review per actual planned segment','Verify actual TCP endpoint and opening before descent','Fresh local upper/body measurement uncertainty','No automatic closure or force press'])
    if isolated_prototype_only:
        result['isolated_prototype_only']=True
        if not lower_neighbors_verified or lower_neighbor_upper_offset_m is None or not np.isfinite(lower_neighbor_upper_offset_m):
            result['reason']='Isolated prototype requires measured lower-neighbor upper bound and independent cloud clearance'
            return result
        # Prototype calibrates stud/body response by measured grasp width, never
        # authorizes a dense-pallet pick from unverified body/stud semantics.
        for i in range(int(np.floor(max_descent_m/step_m))+1):
            descent=min(float(i*step_m),max_descent_m)
            pose=start.copy();pose[2,3]-=descent/n[2]
            toe_heights=(toes@pose[:3,:3].T+pose[:3,3])@n
            clearance=float(min(toe_heights)-lower_neighbor_upper_offset_m)
            if clearance<.003:break
            result['steps'].append(dict(index=i,descent_m=descent,T_base_tcp=pose.tolist(),
                minimum_toe_lower_neighbor_clearance_m=clearance))
        result['trial_ready']=len(result['steps'])>1
        result['body_reference_verified']=False
        result['closure_protocol']='Open16.8mm verified before EACH descent; separately reviewed12.8mm close with outerepsilon.8mm; measured width>=13.6mm is not stud success, stop/reopen; never descend closed. No automatic actions.'
        result['reason']='Isolated empirical height/width calibration only; lower-neighbor bound and cloud clearance required, no dense-pallet authorization'
        return result
    if not body_reference_verified or body_upper_offset_m is None or body_uncertainty_m is None:
        result['reason']='No independently verified body-top bound; upper-envelope alone cannot authorize toe/body/screw clearance'
        result['minimal_additional_measurement']='Resolve body/stud side profile with close side-on ruler/caliper measurement or independently resolved RGBD body strip; record uncertainty. No nominal stud-height subtraction.'
        return result
    if not np.isfinite([body_upper_offset_m,body_uncertainty_m]).all() or body_uncertainty_m<0:
        raise ValueError('Finite measured body bound and uncertainty required')
    safe_body_upper=body_upper_offset_m+body_uncertainty_m
    for i in range(int(np.floor(max_descent_m/step_m))+1):
        pose=start.copy();pose[2,3]-=i*step_m/n[2]
        toe_heights=(toes@pose[:3,:3].T+pose[:3,3])@n
        lowest=float(min(toe_heights));deepest=float(upper_offset_m+upper_uncertainty_m-lowest)
        # Never enter the body and never let screw clearance consume toe budget.
        if lowest<safe_body_upper or max(0,deepest)+tilt+body_uncertainty_m>usable_insertion_m:
            break
        result['steps'].append(dict(index=i,descent_m=min(float(i*step_m),max_descent_m),T_base_tcp=pose.tolist(),
            minimum_toe_body_clearance_m=lowest-safe_body_upper,
            conservative_toe_insertion_m=max(0,deepest)+tilt+body_uncertainty_m))
    result['trial_ready']=len(result['steps'])>1
    result['reason']='Conditional offline geometry only; each physical stage needs independent path and measured endpoint verification'
    return result
