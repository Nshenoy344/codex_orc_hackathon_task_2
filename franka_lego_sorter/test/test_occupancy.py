import unittest
import numpy as np
from lego_sorter.occupancy import TemplateRegistry, occupancy_for_count, rotate_layer_pattern, first_new_layer_from_reference_aligned_poses


def template(offset=0., pitch=.0112):
    poses = []
    # Deliberately nonuniform spacing and differing slot heights; no rectangular fit.
    for i in range(12):
        p = np.eye(4)
        p[:3, 3] = [.2+(i//3)*.023 + offset, -.1+(i%3)*.036+(i//3)*.001, .04+i*.0001]
        poses.append(p.tolist())
    return dict(kind='existing_brick_layer', first_place_reference='tcp_pose_for_first_new_layer',
                first_place_poses=poses, layer_pitch_m=pitch,rotation_center_base_m=[.235+offset,-.06,.04],layer_rotation_deg=90,first_new_layer_rotation_from_reference_deg=90)


class OccupancyTest(unittest.TestCase):
    def test_layer_fills_all12_before_next(self):
        self.assertEqual(occupancy_for_count(0)['slot_counts'], [0]*12)
        self.assertEqual(occupancy_for_count(11)['slot_counts'], [1]*11+[0])
        self.assertEqual(occupancy_for_count(12)['slot_counts'], [1]*12)
        self.assertEqual(occupancy_for_count(13)['slot_counts'], [2]+[1]*11)
        self.assertEqual(occupancy_for_count(13)['next_slot_index'], 1)

    def test_measured_slot_geometry_and_pitch_preserved(self):
        t = template()
        registry = TemplateRegistry({'requested': t})
        for i in range(12):
            first = registry.placement('requested', i)
            np.testing.assert_allclose(first['placement_pose'], t['first_place_poses'][i])
            self.assertEqual((first['ridge_index'],first['ridge_position']), divmod(i,3))
            next_layer = registry.placement('requested', i+12)
            expected = rotate_layer_pattern(t['first_place_poses'],t['rotation_center_base_m'],90.)[i]
            expected[2,3] += t['layer_pitch_m']
            np.testing.assert_allclose(next_layer['placement_pose'], expected)

    def test_template_geometry_signature_changes(self):
        first = TemplateRegistry({'requested': template()})
        second = TemplateRegistry({'requested': template(offset=.001)})
        self.assertNotEqual(first.signature, second.signature)
        self.assertEqual(first.signature, TemplateRegistry.from_dict(first.to_dict()).signature)

    def test_missing_height_reference_pitch_or_slots_rejected(self):
        for key in ('kind', 'first_place_reference', 'layer_pitch_m', 'first_place_poses'):
            t = template()
            del t[key]
            with self.assertRaises((TypeError, ValueError)):
                TemplateRegistry({'requested':t})
        t = template()
        t['first_place_poses'] = t['first_place_poses'][:11]
        with self.assertRaises(ValueError):
            TemplateRegistry({'requested':t})

    def test_coincident_slots_rejected(self):
        t = template()
        t['first_place_poses'][1] = t['first_place_poses'][0]
        with self.assertRaisesRegex(ValueError,'coincide'):
            TemplateRegistry({'requested':t})

class InterlockingPatternTest(unittest.TestCase):
    def test_whole_nonuniform_pattern_rotates_xy_and_orientation(self):
        t=template();reg=TemplateRegistry({'requested':t});center=np.array(t['rotation_center_base_m'])
        for k in range(5):
            angle=k*np.pi/2;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            for slot in range(12):
                source=np.array(t['first_place_poses'][slot]);actual=np.array(reg.placement('requested',12*k+slot)['placement_pose'])
                np.testing.assert_allclose(actual[:3,3],center+R@(source[:3,3]-center)+[0,0,k*t['layer_pitch_m']],atol=1e-12)
                np.testing.assert_allclose(actual[:3,:3],R@source[:3,:3],atol=1e-12)
        self.assertGreater(np.linalg.norm(np.array(reg.placement('requested',12)['placement_pose'])[:2,3]-np.array(t['first_place_poses'][0])[:2,3]),.01)
    def test_firstnew_conversion_separate_and_does_not_invent_height(self):
        t=template();original=np.array(t['first_place_poses']);converted=first_new_layer_from_reference_aligned_poses(original,t['rotation_center_base_m'])
        np.testing.assert_allclose(converted[:,2,3],original[:,2,3]);np.testing.assert_allclose(original,np.array(t['first_place_poses']))
        np.testing.assert_allclose(converted,rotate_layer_pattern(original,t['rotation_center_base_m'],90))
    def test_missing_center_rotation_policy_and_pitch_refused(self):
        for key in ('rotation_center_base_m','layer_rotation_deg','first_new_layer_rotation_from_reference_deg','layer_pitch_m'):
            t=template();del t[key]
            with self.assertRaises(ValueError):TemplateRegistry({'requested':t})
