"""Measured twelve-position pallet templates; no inferred rectangular spacing.

Every first_place_pose is the actual TCP pose for placing one NEW brick in that
slot. Its Z already accounts for whether the template is a fixture or existing
brick layer. Layer pitch is independently measured, never inferred from kind.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SLOTS_PER_LAYER = 12
PALLET_DESTINATIONS = ('requested', 'upper', 'lower')


def rotate_layer_pattern(poses, rotation_center_base_m, angle_deg):
    """Rotate the entire observed pattern about an explicit measured base-Z axis.

    Heights are preserved here. Input poses must already use the correct first
    NEW-layer TCP height; this helper supplies no supportheight or layerpitch.
    """
    values=np.asarray(poses,float);center=np.asarray(rotation_center_base_m,float)
    if values.shape!=(12,4,4) or center.shape!=(3,) or not np.isfinite(values).all() or not np.isfinite(center).all() or not math.isfinite(float(angle_deg)):
        raise ValueError('Twelve finite poses and explicit measured rotation center required')
    a=math.radians(float(angle_deg));R=np.array([[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.],[0.,0.,1.]])
    result=values.copy();result[:,:3,3]=(values[:,:3,3]-center)@R.T+center
    result[:,:3,:3]=np.einsum('ij,njk->nik',R,values[:,:3,:3])
    return result


def first_new_layer_from_reference_aligned_poses(reference_aligned_first_new_poses, rotation_center_base_m):
    """Apply required90deg pattern rotation separately before registry creation."""
    return rotate_layer_pattern(reference_aligned_first_new_poses,rotation_center_base_m,90.)


def occupancy_for_count(count):
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('Committed pallet count must be a nonnegative integer')
    layer, slot = divmod(count, SLOTS_PER_LAYER)
    return dict(slot_counts=[layer + int(i < slot) for i in range(SLOTS_PER_LAYER)],
                completed_layers=layer, current_layer=layer, next_slot_index=slot,
                total_committed=count)


class TemplateRegistry:
    def __init__(self, templates, base_frame='fr3_link0', tcp_frame='fr3_hand_tcp'):
        if not isinstance(templates, dict) or not templates:
            raise ValueError('Measured pallet templates must be a nonempty destination mapping')
        if not base_frame or not tcp_frame:
            raise ValueError('Template base_frame and tcp_frame must be explicit')
        self.base_frame, self.tcp_frame = str(base_frame), str(tcp_frame)
        self._templates = {}
        for destination, template in templates.items():
            if destination not in PALLET_DESTINATIONS:
                raise ValueError(f'Unknown pallet destination {destination}; scrap is a tray, not a pallet')
            if not isinstance(template, dict):
                raise ValueError(f'{destination} template must be a mapping')
            kind = template.get('kind')
            if kind not in ('base', 'fixture', 'existing_brick_layer'):
                raise ValueError(f'{destination}: inspect and set template kind explicitly')
            if template.get('first_place_reference') != 'tcp_pose_for_first_new_layer':
                raise ValueError(f'{destination}: first_place_reference must explicitly account for template height')
            center=np.asarray(template.get('rotation_center_base_m'),float)
            if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError(f'{destination}: explicit measured rotation_center_base_m required')
            if template.get('layer_rotation_deg')!=90 or template.get('first_new_layer_rotation_from_reference_deg')!=90:
                raise ValueError(f'{destination}: firstnew and subsequent wholelayer rotations must explicitly be90degrees')
            pitch = template.get('layer_pitch_m')
            if isinstance(pitch, bool) or not isinstance(pitch, (int, float)) or not math.isfinite(pitch) or pitch <= 0:
                raise ValueError(f'{destination}: measure a positive finite layer_pitch_m')
            poses = np.asarray(template.get('first_place_poses'), dtype=float)
            if poses.shape != (SLOTS_PER_LAYER, 4, 4) or not np.isfinite(poses).all():
                raise ValueError(f'{destination}: exactly12 measured finite4x4 first_place_poses required')
            for pose in poses:
                rot = pose[:3, :3]
                if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7) or
                        not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or
                        not np.isclose(np.linalg.det(rot), 1., atol=1e-5)):
                    raise ValueError(f'{destination}: placement pose must be a rigid transform')
            # Coincident TCP centers cannot represent12 distinct brick locations.
            for i in range(SLOTS_PER_LAYER):
                for j in range(i):
                    if np.linalg.norm(poses[i, :3, 3]-poses[j, :3, 3]) < .001:
                        raise ValueError(f'{destination}: measured slots {j}/{i} coincide')
            self._templates[destination] = dict(kind=kind,
                first_place_reference='tcp_pose_for_first_new_layer',
                first_place_poses=poses.tolist(), layer_pitch_m=float(pitch),rotation_center_base_m=center.tolist(),layer_rotation_deg=90,first_new_layer_rotation_from_reference_deg=90)
        self._canonical = dict(format='measured_pallet_templates_v1', base_frame=self.base_frame,
                               tcp_frame=self.tcp_frame, destinations=self._templates)
        self.signature = hashlib.sha256(json.dumps(self._canonical, sort_keys=True,
                                                   allow_nan=False).encode()).hexdigest()

    @property
    def destinations(self):
        return set(self._templates)

    def to_dict(self):
        return deepcopy(self._canonical)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get('format') != 'measured_pallet_templates_v1':
            raise ValueError('Unknown template registry format')
        return cls(value['destinations'], value['base_frame'], value['tcp_frame'])

    @classmethod
    def load(cls, path):
        import yaml
        with Path(path).expanduser().open() as stream:
            return cls.from_dict(yaml.safe_load(stream))

    def placement(self, destination, total_committed):
        logical = occupancy_for_count(total_committed)
        if destination not in self._templates:
            raise ValueError(f'No measured template registered for {destination}')
        slot, layer = logical['next_slot_index'], logical['current_layer']
        template = self._templates[destination]
        poses=rotate_layer_pattern(template['first_place_poses'],template['rotation_center_base_m'],layer*90.)
        pose=poses[slot]
        # first_place_poses ALREADY describe90deg firstnew layer vs reference.
        # Each further wholepattern rotates another90deg before measuredZpitch.
        pose[2, 3] += layer * template['layer_pitch_m']
        return dict(slot_index=slot, layer_index=layer, ridge_index=slot//3,
                    ridge_position=slot%3, placement_pose=pose.tolist(),
                    template_signature=self.signature)
