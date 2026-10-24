"""Offline review of the one explicit nominal-geometry noncontact approach30."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from validate_isolated_hover import review as review_path


def review(plan_path,geometry=None,opening=None):
    path=Path(plan_path);plan=json.loads(path.read_text());meta=plan['isolated_diagnostic_stage']
    if meta.get('stage')!='approach30':raise ValueError('Only noncontact approach30 is enabled; no grasp/descent-to-contact action')
    targets=Path(meta['targets']);source=json.loads(targets.read_text());plane=next(p for p in source['planes']if p['plane']==meta['plane'])
    if source.get('kind')!='explicit_nominal_geometry_diagnostic_only' or source.get('production_qualified')is not False:raise ValueError('Requires explicit nominal diagnostic geometry')
    for key,source_key,hash_key in [('snapshot','snapshot','snapshot_sha256'),('handeye','handeye_artifact','handeye_sha256')]:
        if Path(meta[key]).resolve()!=Path(source[source_key]).resolve() or hashlib.sha256(Path(meta[key]).read_bytes()).hexdigest()!=source[hash_key]:raise ValueError('Nominal diagnostic source evidence changed')
    target=np.asarray(plan['T_base_tcp_target']);expected=np.asarray(plane['poses']['hover30'])
    if not np.allclose(target,expected,atol=1e-8):raise ValueError('Approach30 must match pinned diagnostic pose exactly')
    start=np.asarray(plan['T_base_tcp_start']);length=np.linalg.norm(target[:3,3]-start[:3,3])
    if not 0<length<=.20 or target[2,3]>=start[2,3]or not np.allclose(target[:3,2],[0,0,-1],atol=1e-8):raise ValueError('Approach30 requires a vertical-tool descending segment atmost200mm')
    geometry=Path(geometry)if geometry else path.with_suffix('.scene_geometry.json')
    if not geometry.exists():
        measured=plane['measured_table_plane'];g={'visible_top_base_m':plane['assumed_nominal_stud_top_base_m'],'table_point_base_m':plane['floor_ray_point_base_m'],
            'measured_table_plane':{'normal_base':measured['normal_base'],**measured['quality']},
            'scene_reference_semantics':'Nominal LEGO11.4mm studtop above robust measured localfloor; not a measured brickbody or production reference.',
            'source_targets':str(targets.resolve()),'source_targets_sha256':hashlib.sha256(targets.read_bytes()).hexdigest()}
        geometry.write_text(json.dumps(g,indent=2))
    g=json.loads(geometry.read_text())
    if g['source_targets_sha256']!=hashlib.sha256(targets.read_bytes()).hexdigest():raise ValueError('Diagnostic target source changed')
    width=float(meta['gripper_width_m'])if opening is None else float(opening)
    r=review_path(path,'data/commissioning/fr3_stock_hand.urdf',meta['snapshot'],meta['handeye'],geometry,joint_space=True,opening=width)
    checks=r['checks'];checks.pop('minimum150mm_tip_clearance');checks.update(approach20mm_nominal_tip_clearance=r['minimum_modeled_tip_height_above_measured_brick_m']>=.020,
        approach_straight_line=r['max_cartesian_line_deviation_m']<=.0005,approach_progress=r['path_progress_min_m']>=-.0005 and r['path_progress_max_m']<=length+.0005)
    r['checks']={key:bool(value) for key,value in checks.items()}
    r['failed_checks']=[k for k,v in r['checks'].items()if not v];r['passed_numeric_checks']=not r['failed_checks'];r['purpose']='isolated_approach30'
    r['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();r['source_targets']=str(targets.resolve());r['source_targets_sha256']=hashlib.sha256(targets.read_bytes()).hexdigest()
    r['limits_of_review']='Noncontact approach30 only, based on measured localfloor and explicit nominal LEGO geometry. No contact, gripper action or production sorting authorization.'
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('plan');a=p.parse_args();r=review(a.plan);Path(a.plan).with_suffix('.validation.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items()if k not in('joint_limits','full_stock_hand_observed_cloud_sweep','full_arm_convex_hull_review')},indent=2));return 0 if r['passed_numeric_checks'] else 2
if __name__=='__main__':raise SystemExit(main())
