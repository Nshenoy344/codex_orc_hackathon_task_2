from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from lego_sorter.camera import Frame
from lego_sorter.occupancy import TemplateRegistry
from lego_sorter.pallet_runner import PalletRunner, candidate_pick_pose, verify_source_removal, grasp_tcp_surface_offset, tcp_at_surface, surface_from_tcp, verify_destination_jaw_clearance, selected_grasp_width, check_grasp_insertion, verify_narrow_preopen


def pose(x=0., y=0., z=.6):
    result = np.diag([1., -1., -1., 1.])
    result[:3, 3] = [x, y, z]
    return result


def candidate(color='red', orientation='across_width'):
    return SimpleNamespace(color_id=color, graspable=True, clearance_fraction=.99,
        point_camera_m=[0., 0., .3], grasp_closing_axis_camera=[0., 1., 0.],
        grasp_width_m=.0158 if orientation == 'across_width' else .0318,
        grasp_orientation=orientation, box_px=[[30, 30], [70, 30], [70, 50], [30, 50]])


class Model:
    def validate_labels(self, labels):
        return True

    def validate_camera(self, metadata):
        return True

    def classify(self, image, mask):
        return SimpleNamespace(label='red')


class Robot:
    def __init__(self, grasp_ok=True):
        self.pose = pose()
        self.events = []
        self.width = 0.
        self.grasp_ok = grasp_ok

    def current_pose(self):
        return self.pose.copy()

    def move_linear(self, target):
        self.events.append(('linear', np.array(target).copy()))
        self.pose = np.array(target).copy()

    def move_pose(self, target):
        self.events.append(('transit', np.array(target).copy()))
        self.pose = np.array(target).copy()

    def open_gripper(self, width_m=None):
        self.width = .06 if width_m is None else width_m
        self.events.append(('open', self.width))

    def grasp(self, width_m=None):
        self.events.append(('grasp', width_m))
        self.width = width_m
        return self.grasp_ok

    def gripper_state(self):
        return {'width_m': self.width}


class PalletRunnerTests(unittest.TestCase):
    def config(self, path):
        return dict(job=dict(good_colors=['red', 'dark', 'orange'], bad_colors=['tan', 'brown'],
                             requested_colors=['red'], requested_count=13),
            journal_file=str(path), perception=dict(roi=[0, 0, 100, 80]),
            motion=dict(base_frame='fr3_link0', tcp_frame='fr3_hand_tcp', open_width_m=.06,
                        gripper_width_tolerance_m=.003),
            execution=dict(source_arrangement='pallet', workspace_min_m=[-.8, -.8, .01],
                workspace_max_m=[.8, .8, 1.], transit_height_m=.8, consensus_frames=2,
                max_observation_attempts=2, max_pallet_layers=2, settle_sec=0.,
                tcp_to_fingertip_m=.0095, finger_insertion_below_surface_m=.006, maximum_usable_tip_insertion_m=.008, gripper_opening_margin_m=.004,
                detection_position_tolerance_m=.004,
                destination_xy_tolerance_m=.006, destination_z_tolerance_m=.006,
                template_grasp_orientation='across_width', verify_destination=True))

    def templates(self):
        destinations = {}
        for name, center in [('requested', (.2, 0.)), ('upper', (0., .2)), ('lower', (0., -.2))]:
            slots = [pose(center[0]+(i//3)*.024, center[1]+(i%3)*.037, .3035).tolist()
                     for i in range(12)]
            destinations[name] = dict(kind='existing_brick_layer',
                first_place_reference='tcp_pose_for_first_new_layer',
                first_place_poses=slots, layer_pitch_m=.0114,rotation_center_base_m=[center[0]+.036,center[1]+.037,.3035],layer_rotation_deg=90,first_new_layer_rotation_from_reference_deg=90)
        return TemplateRegistry(destinations)

    def runner(self, path, robot=None, config=None):
        c = config or self.config(path)
        poses = dict(observation=pose(), pick_reference=pose(), scrap=pose(-.2, 0., .4))
        return PalletRunner(c, poses, np.eye(4), Model(), robot or Robot(),
                            SimpleNamespace(capture_burst=lambda: object()), self.templates())

    def run_mocked(self, runner, brick=None, source_error=None, destination_error=None):
        brick = brick or candidate()
        with patch.object(runner, 'observe_source', return_value=(brick, object(), pose())), \
                patch.object(runner, '_capture', return_value=(object(), pose())), \
                patch('lego_sorter.pallet_runner.verify_source_removal',
                      side_effect=source_error, return_value={'removed_fraction': 1.}), \
                patch.object(runner, '_verify_destination', side_effect=destination_error) as verify:
            result = runner.run(max_transfers=1)
        return result, verify

    def test_verified_transfer_uses_registered_slot_and_commits(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            state, verified = self.run_mocked(runner)
            self.assertEqual(state['counts']['requested'], 1)
            self.assertEqual(state['occupancy']['requested']['slot_counts'], [1]+[0]*11)
            self.assertEqual(state['composition']['requested'], {'red': 1})
            self.assertIsNone(state['pending'])
            self.assertEqual(verified.call_count, 1)
            self.assertIn(('grasp', .0158), runner.robot.events)
            expected = np.array(runner.templates.placement('requested', 0)['placement_pose'])
            self.assertTrue(any(kind == 'linear' and np.allclose(value, expected)
                                for kind, value in runner.robot.events if value is not None and kind != 'grasp'))

    def test_two_color_request_records_mixture_in_shared_measured_pallet(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'run.json'
            config = self.config(path)
            config['job']['requested_colors'] = ['red', 'dark']
            runner = self.runner(path, config=config)
            with patch.object(runner, 'observe_source', side_effect=[
                    (candidate('red'), object(), pose()), (candidate('dark'), object(), pose())]), \
                    patch.object(runner, '_capture', return_value=(object(), pose())), \
                    patch('lego_sorter.pallet_runner.verify_source_removal', return_value={}), \
                    patch.object(runner, '_verify_destination'):
                state = runner.run(max_transfers=2)
            self.assertEqual(state['composition']['requested'], {'red': 1, 'dark': 1})
            self.assertEqual(state['occupancy']['requested']['slot_counts'], [1,1]+[0]*10)
            self.assertEqual([t['slot_index'] for t in state['history']], [0,1])
            self.assertEqual(state['counts']['upper'], 0)
            self.assertEqual(state['counts']['lower'], 0)

    def test_thirteenth_verified_transfer_starts_next_layer_same_slot(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            for _ in range(12):
                transfer = runner.planner.reserve('red')
                runner.planner.mark_motion_started(transfer['id'])
                runner.planner.commit(transfer['id'])
            state, _ = self.run_mocked(runner)
            self.assertEqual(state['occupancy']['requested']['slot_counts'], [2]+[1]*11)
            self.assertEqual(state['history'][-1]['slot_index'], 0)
            self.assertEqual(state['history'][-1]['layer_index'], 1)
            target = np.array(state['history'][-1]['placement_pose'])
            self.assertAlmostEqual(target[2, 3], .3035+.0114)

    def test_dynamic_length_grasp_compensates_template_yaw(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            _, verified = self.run_mocked(runner, candidate(orientation='across_length'))
            self.assertIn(('grasp', .0318), runner.robot.events)
            baseline = np.array(runner.templates.placement('requested', 0)['placement_pose'])
            release = [value for kind, value in runner.robot.events if kind == 'linear'
                       and np.allclose(value[:3, 3], baseline[:3, 3])]
            self.assertEqual(len(release), 1)
            np.testing.assert_allclose(release[0][:3, :3],
                np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]) @ baseline[:3,:3], atol=1e-10)
            self.assertEqual(verified.call_count, 1)

    def test_source_failure_keeps_pending_without_destination_release(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                self.run_mocked(runner, source_error=RuntimeError('source changed'))
            self.assertEqual(runner.planner.state['processed'], 0)
            self.assertTrue(runner.planner.state['pending']['motion_started'])
            self.assertEqual(sum(event[0] == 'open' for event in runner.robot.events), 1)

    def test_destination_failure_does_not_advance_occupancy(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            with self.assertRaisesRegex(RuntimeError, 'placement absent'):
                self.run_mocked(runner, destination_error=RuntimeError('placement absent'))
            self.assertEqual(runner.planner.state['occupancy']['requested']['slot_counts'], [0]*12)
            self.assertIsNotNone(runner.planner.state['pending'])
            self.assertEqual(sum(event[0] == 'open' for event in runner.robot.events), 2)

    def test_grasp_failure_never_places_or_commits(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json', robot=Robot(grasp_ok=False))
            with self.assertRaisesRegex(RuntimeError, 'grasp failed'):
                self.run_mocked(runner)
            self.assertEqual(runner.planner.state['processed'], 0)
            self.assertIsNotNone(runner.planner.state['pending'])

    def test_unknown_candidates_reobserve_without_grasp_or_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            unknown = candidate(color=None)
            with patch('lego_sorter.pallet_runner.inspect_pallet',
                       return_value=SimpleNamespace(candidates=[unknown])) as inspect:
                with self.assertRaisesRegex(RuntimeError, 'No known individually accessible'):
                    runner.run(max_transfers=1)
            self.assertEqual(inspect.call_count, 2)
            self.assertFalse(any(e[0] in ('open', 'grasp') for e in runner.robot.events))
            self.assertIsNone(runner.planner.state['pending'])

    def test_confirmed_scrap_routes_to_tray_without_pallet_occupancy(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            state, verified = self.run_mocked(runner, candidate('tan'))
            self.assertEqual(state['counts']['scrap'], 1)
            self.assertEqual(state['composition']['scrap'], {'tan': 1})
            self.assertEqual(state['occupancy']['requested']['slot_counts'], [0]*12)
            self.assertEqual(verified.call_count, 0)
            self.assertIsNone(state['pending'])

    def test_consensus_tracks_same_candidate_not_highest_next_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            selected, distractor = candidate(), candidate()
            distractor.point_camera_m = [.04, 0., .29]  # Higher but a different brick.
            with patch('lego_sorter.pallet_runner.inspect_pallet', side_effect=[
                    SimpleNamespace(candidates=[selected]),
                    SimpleNamespace(candidates=[distractor, selected])]):
                observed, _, _ = runner.observe_source()
            np.testing.assert_allclose(observed.point_camera_m, selected.point_camera_m)
            self.assertFalse(any(e[0] == 'grasp' for e in runner.robot.events))

    def test_unknown_brick_never_routes_to_scrap(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            before = runner.planner.state
            self.assertEqual(runner._known_candidates(SimpleNamespace(candidates=[candidate(None)]), pose()), [])
            self.assertEqual(runner.planner.state, before)

    def test_invalid_verification_threshold_fails_before_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'run.json'
            robot = Robot()
            c = self.config(path)
            c['execution']['destination_surface_fraction'] = float('nan')
            with self.assertRaisesRegex(ValueError, 'destination_surface_fraction'):
                self.runner(path, robot=robot, config=c)
            self.assertEqual(robot.events, [])

    def test_source_verification_removal_and_surrounding_motion(self):
        intr = dict(fx=300., fy=300., ppx=50., ppy=40.)
        before_depth = np.full((80, 100), .4, np.float32)
        before_depth[30:51, 30:71] = .3
        after_depth = np.full((80, 100), .4, np.float32)
        before = Frame(np.zeros((80,100,3), np.uint8), before_depth, intr, 0.)
        after = Frame(before.color_bgr.copy(), after_depth, intr, 0.)
        result = verify_source_removal(before, after, candidate(), np.eye(4), np.eye(4), {})
        self.assertGreater(result['removed_fraction'], .99)
        after.depth_m[:] = .5  # Entire support moved, not a verified single removal.
        with self.assertRaisesRegex(RuntimeError, 'surroundings changed'):
            verify_source_removal(before, after, candidate(), np.eye(4), np.eye(4), {})

    def test_stock_tcp_offset_keeps_physical_tips_inside_one_body(self):
        settings = dict(tcp_to_fingertip_m=.0095, finger_insertion_below_surface_m=.006)
        offset = grasp_tcp_surface_offset(settings)
        self.assertAlmostEqual(offset, .0035)
        target = candidate_pick_pose(candidate(), pose(), np.eye(4), pose(), .0095, .006)
        self.assertAlmostEqual(target[2,3], .3035)
        fingertip_z = target[2,3] - settings['tcp_to_fingertip_m']
        self.assertAlmostEqual(.3-fingertip_z, .006)
        self.assertGreater(fingertip_z, .3-.0096)

    def test_tip_offset_cannot_be_applied_to_an_upward_pointing_tool(self):
        with self.assertRaisesRegex(ValueError, 'near-downward'):
            candidate_pick_pose(candidate(), pose(), np.eye(4), np.eye(4), .0095, .006)

    def test_tilted_tool_places_actual_tip_exactly_below_measured_surface(self):
        angle = np.radians(5.)
        tilt = np.array([[1.,0.,0.],[0.,np.cos(angle),-np.sin(angle)],
                         [0.,np.sin(angle),np.cos(angle)]])
        reference = pose()
        reference[:3,:3] = tilt @ reference[:3,:3]
        target = candidate_pick_pose(candidate(), pose(), np.eye(4), reference, .0095, .006)
        actual_tip = target[:3,3] + target[:3,:3] @ [0.,0.,.0095]
        np.testing.assert_allclose(actual_tip, [0.,0.,.294], atol=1e-12)
        self.assertGreater(abs(target[1,3]), .0008)
        np.testing.assert_allclose(surface_from_tcp(target,.0095,.006), [0.,0.,.3], atol=1e-12)

    def test_rotated_tilted_placement_preserves_measured_slot_contact(self):
        angle = np.radians(4.)
        tilt = np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],
                         [-np.sin(angle),0.,np.cos(angle)]])
        measured = pose(.2, .1, .3035)
        measured[:3,:3] = tilt @ measured[:3,:3]
        surface = surface_from_tcp(measured, .0095, .006)
        rotated = measured.copy()
        rotated[:3,:3] = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]) @ measured[:3,:3]
        compensated = tcp_at_surface(surface, rotated, .0095, .006)
        np.testing.assert_allclose(surface_from_tcp(compensated,.0095,.006),surface,atol=1e-12)
        self.assertGreater(np.linalg.norm(compensated[:2,3]-measured[:2,3]), .0005)

    def test_deprecated_depth_alone_cannot_start_pallet_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'run.json'
            c = self.config(path)
            del c['execution']['tcp_to_fingertip_m']
            del c['execution']['finger_insertion_below_surface_m']
            c['execution']['grasp_depth_below_surface_m'] = .006
            robot = Robot()
            with self.assertRaisesRegex(ValueError, 'legacy grasp depth is insufficient'):
                self.runner(path, robot=robot, config=c)
            self.assertEqual(robot.events, [])

    def test_placement_verification_uses_tip_corrected_top_surface(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            _, verified = self.run_mocked(runner)
            measured_surface = verified.call_args.args[1]
            expected_tcp = np.array(runner.templates.placement('requested', 0)['placement_pose'])
            self.assertAlmostEqual(measured_surface[2], expected_tcp[2,3]-.0035)
            self.assertAlmostEqual(measured_surface[2], .3)
            self.assertAlmostEqual(runner.settings['grasp_depth_below_surface_m'], .006)

    def test_nonphysical_tip_and_insertion_values_are_rejected(self):
        for tip, insertion in [(float('nan'), .006), (0., .006), (.1, .006), (.0095, .0096),
                               (.0095, 0.), (.0095, float('inf'))]:
            with self.subTest(tip=tip, insertion=insertion), self.assertRaises(ValueError):
                grasp_tcp_surface_offset(dict(tcp_to_fingertip_m=tip,
                    finger_insertion_below_surface_m=insertion))

    def test_preopen_and_seated_release_are_narrow_for_both_grasp_axes(self):
        for orientation, expected in [('across_width',.0198),('across_length',.0358)]:
            with self.subTest(orientation=orientation), tempfile.TemporaryDirectory() as folder:
                runner = self.runner(Path(folder)/'run.json')
                self.run_mocked(runner,candidate(orientation=orientation))
                openings = [value for kind,value in runner.robot.events if kind=='open']
                self.assertEqual(len(openings),2)
                for width in openings:
                    self.assertAlmostEqual(width,expected)
                    self.assertLess(width,.06)
                # Seated release occurs before the straight lift; no broad opening intervenes.
                second = [i for i,event in enumerate(runner.robot.events) if event[0]=='open'][1]
                self.assertEqual(runner.robot.events[second+1][0],'linear')

    def test_measured_same_layer_neighbor_blocks_jaw_sweep_before_pickup(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            definition = runner.templates.to_dict()
            # 33mm center spacing leaves17.2mm edge gap, less than16+2mm jaw sweep.
            definition['destinations']['requested']['first_place_poses'][1][1][3] = .033
            tight = TemplateRegistry.from_dict(definition)
            destination = np.array(tight.placement('requested',0)['placement_pose'])
            occupied = [0,1]+[0]*10
            with self.assertRaisesRegex(ValueError,'intersects occupied slot 1'):
                verify_destination_jaw_clearance(destination,.0158,tight,'requested',occupied,
                    runner.settings,runner.e)
            # The same brick in the previous layer is below the physical tip.
            higher = np.array(tight.placement('requested',12)['placement_pose'])
            report = verify_destination_jaw_clearance(higher,.0158,tight,'requested',occupied,
                runner.settings,runner.e)
            self.assertEqual(report['checked_slots'],[])

    def test_known_occupied_destination_rejects_transfer_before_grasp(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            transfer=runner.planner.reserve('red')
            runner.planner.mark_motion_started(transfer['id'])
            runner.planner.commit(transfer['id'])
            # The next target is slot1. Widen physical fingers beyond available21.2mm gap.
            runner.settings['finger_clearance_m']=.025
            with self.assertRaisesRegex(ValueError,'jaw sweep intersects'):
                self.run_mocked(runner)
            self.assertIsNone(runner.planner.state['pending'])
            self.assertEqual(runner.planner.state['processed'],1)
            self.assertFalse(any(kind in ('open','grasp') for kind,_ in runner.robot.events))

    def test_measured_two_mm_tip_and_14_5_mm_gap_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'run.json'
            config = self.config(path)
            config['execution'].update(finger_insertion_below_surface_m=.0015,
                maximum_usable_tip_insertion_m=.002, gripper_opening_margin_m=.001)
            config['pallet'] = dict(finger_clearance_m=.0135, measured_interbrick_gap_m=.0145,
                                    grasp_width_source='configured_2x4_nominal')
            runner = self.runner(path, config=config)
            self.assertAlmostEqual(runner.settings['grasp_depth_below_surface_m'], .0015)
            self.assertAlmostEqual(runner.settings['jaw_opening_margin_m'], .001)
            config['execution']['finger_insertion_below_surface_m'] = .0021
            with self.assertRaisesRegex(ValueError, 'exceeds the measured usable tip'):
                self.runner(path, config=config)

    def test_opening_sweep_cannot_exceed_measured_gap(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'run.json'
            config = self.config(path)
            config['pallet'] = dict(finger_clearance_m=.0135, measured_interbrick_gap_m=.0145)
            # Existing4mm total opening would require15.5mm, exceeding14.5mm measuredgap.
            with self.assertRaisesRegex(ValueError, 'opening sweep exceeds measured'):
                self.runner(path, config=config)

    def test_nominal_width_preopen_does_not_follow_noisy_contour(self):
        for orientation, nominal in [('across_width',.0158),('across_length',.0318)]:
            with self.subTest(orientation=orientation), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'run.json'
                config = self.config(path)
                config['execution']['gripper_opening_margin_m'] = .001
                config['pallet'] = dict(grasp_width_source='configured_2x4_nominal')
                runner = self.runner(path, config=config)
                brick = candidate(orientation=orientation)
                brick.grasp_width_m = nominal-.002  # Noisy measured face width.
                self.run_mocked(runner,brick)
                self.assertIn(('grasp',nominal),runner.robot.events)
                openings = [width for kind,width in runner.robot.events if kind=='open']
                self.assertEqual(len(openings),2)
                for width in openings:
                    self.assertAlmostEqual(width,nominal+.001)
                self.assertEqual(brick.grasp_width_m,nominal-.002)  # Evidence remains measured.

    def test_usable_tip_measurement_is_not_silently_defaulted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'run.json'
            config = self.config(path)
            del config['execution']['maximum_usable_tip_insertion_m']
            with self.assertRaisesRegex(ValueError, 'explicit finite physical measurement'):
                self.runner(path, config=config)

    def test_vertical_measured_tip_guard_accepts_1_5_mm_insertion(self):
        actual = tcp_at_surface([0.,0.,.3],pose(),.0095,.0015)
        result = check_grasp_insertion(actual,[0.,0.,.3],.0095,.002,.0328,.020,.0135)
        self.assertAlmostEqual(result['minimum_penetration_m'],.0015)
        self.assertAlmostEqual(result['maximum_penetration_m'],.0015)

    def test_two_degree_tool_tilt_cannot_hide_deeper_lower_fingertip(self):
        angle = np.radians(2.)
        orientation = pose()
        orientation[:3,:3] = np.array([[1.,0.,0.],[0.,np.cos(angle),-np.sin(angle)],
                                      [0.,np.sin(angle),np.cos(angle)]]) @ orientation[:3,:3]
        actual = tcp_at_surface([0.,0.,.3],orientation,.0095,.0015)
        with self.assertRaisesRegex(RuntimeError,'fingertip penetration unsafe'):
            check_grasp_insertion(actual,[0.,0.,.3],.0095,.002,.0328,.020,.0135)

    def test_actual_low_endpoint_stops_before_close_with_pending_transfer(self):
        class LowEndpointRobot(Robot):
            def move_linear(self,target):
                super().move_linear(target)
                if abs(target[0,3])<1e-10 and abs(target[1,3])<1e-10 and target[2,3]<.35:
                    self.pose[2,3]-=.001  # Endpoint1mm below planned1.5mm shallow insertion.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'run.json'
            config = self.config(path)
            config['execution'].update(finger_insertion_below_surface_m=.0015,
                                      maximum_usable_tip_insertion_m=.002)
            robot=LowEndpointRobot()
            runner=self.runner(path,robot=robot,config=config)
            with self.assertRaisesRegex(RuntimeError,'fingertip penetration unsafe'):
                self.run_mocked(runner)
            self.assertIsNotNone(runner.planner.state['pending'])
            self.assertTrue(runner.planner.state['pending']['motion_started'])
            self.assertEqual(runner.planner.state['processed'],0)
            self.assertFalse(any(kind=='grasp' for kind,_ in robot.events))
            self.assertEqual(sum(kind=='open' for kind,_ in robot.events),1)

    def test_planned_tilt_failure_occurs_before_reservation_or_opening(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'run.json'
            config = self.config(path)
            config['execution'].update(finger_insertion_below_surface_m=.0015,
                                      maximum_usable_tip_insertion_m=.002)
            runner=self.runner(path,config=config)
            angle=np.radians(2.)
            runner.poses['pick_reference'][:3,:3]=np.array([
                [1.,0.,0.],[0.,np.cos(angle),-np.sin(angle)],
                [0.,np.sin(angle),np.cos(angle)]]) @ runner.poses['pick_reference'][:3,:3]
            with self.assertRaisesRegex(RuntimeError,'fingertip penetration unsafe'):
                self.run_mocked(runner,candidate(orientation='across_length'))
            self.assertIsNone(runner.planner.state['pending'])
            self.assertEqual(runner.robot.events,[])

    def test_actual_high_endpoint_does_not_close_above_the_brick(self):
        actual = tcp_at_surface([0.,0.,.3],pose(),.0095,.0015)
        actual[2,3]+=.002
        with self.assertRaisesRegex(RuntimeError,'fingertip penetration unsafe'):
            check_grasp_insertion(actual,[0.,0.,.3],.0095,.002,.0168,.020,.0135)

    def test_measured_preopen_retains_actual_14_5_mm_gap_budget(self):
        settings=dict(finger_clearance_m=.0135,measured_interbrick_gap_m=.0145)
        report=verify_narrow_preopen(.01685,.0158,.0168,settings)
        self.assertAlmostEqual(report['opening_tracking_error_m'],.00005)
        self.assertAlmostEqual(report['swept_thickness_m'],.014025)
        self.assertGreater(report['remaining_measured_gap_m'],.0004)
        with self.assertRaisesRegex(RuntimeError,'Measured narrow preopen'):
            verify_narrow_preopen(.0188,.0158,.0168,settings)

    def test_broad_action_success_cannot_authorize_descent_after_overopen(self):
        class OverOpenRobot(Robot):
            def open_gripper(self,width_m=None):
                super().open_gripper(width_m)
                self.width+=.002  # Generic3mm action tolerance could have accepted this.
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'run.json'
            config=self.config(path)
            config['execution'].update(finger_insertion_below_surface_m=.0015,
                maximum_usable_tip_insertion_m=.002,gripper_opening_margin_m=.001)
            config['pallet']=dict(finger_clearance_m=.0135,measured_interbrick_gap_m=.0145)
            robot=OverOpenRobot()
            runner=self.runner(path,robot=robot,config=config)
            with self.assertRaisesRegex(RuntimeError,'Measured narrow preopen'):
                self.run_mocked(runner)
            self.assertTrue(runner.planner.state['pending']['motion_started'])
            self.assertEqual(runner.planner.state['processed'],0)
            self.assertEqual([kind for kind,_ in robot.events],['open'])

    def test_underopen_cannot_descend_even_if_gap_budget_is_smaller(self):
        with self.assertRaisesRegex(RuntimeError,'does not clear'):
            verify_narrow_preopen(.0157,.0158,.0168,{'finger_clearance_m':.0135})

    def test_preopen_actual_sweep_rechecks_measured_gap_at_runtime(self):
        # Both values are within100um command tracking, but the independently
        # measured gap is tighter than the configured command requires.
        with self.assertRaisesRegex(RuntimeError,'consumes the physical interbrick gap'):
            verify_narrow_preopen(.01685,.0158,.0168,
                {'finger_clearance_m':.0135,'measured_interbrick_gap_m':.014})

    def test_broad_preopen_tracking_tolerance_is_not_accepted_by_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'run.json'
            config=self.config(path)
            config['execution'].update(gripper_opening_margin_m=.001,
                                      gripper_opening_tracking_tolerance_m=.003)
            robot=Robot()
            with self.assertRaisesRegex(ValueError,'no larger than half'):
                self.runner(path,robot=robot,config=config)
            self.assertEqual(robot.events,[])

    def test_camera_retreat_is_not_mistaken_for_source_removal(self):
        intr = dict(fx=300., fy=300., ppx=50., ppy=40.)
        before = Frame(np.zeros((80,100,3), np.uint8), np.full((80,100), .3, np.float32), intr, 0.)
        after = Frame(before.color_bgr.copy(), np.full((80,100), .31, np.float32), intr, 0.)
        camera_after = np.eye(4)
        camera_after[2,3] = -.01  # Same unchanged scene is now 10mm farther away.
        with self.assertRaisesRegex(RuntimeError, 'removal unverified'):
            verify_source_removal(before, after, candidate(), np.eye(4), camera_after, {})

    def test_destination_verifies_surface_and_rejects_wrong_height(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.runner(Path(folder)/'run.json')
            intr = dict(fx=600., fy=600., ppx=100., ppy=100.)
            frame = Frame(np.zeros((200,200,3), np.uint8), np.full((200,200), .3, np.float32), intr, 0.)
            with patch.object(runner, '_capture', return_value=(frame, pose())):
                runner._verify_destination(pose(), np.array([0.,0.,.3]), 'red', [1.,0.,0.])
                frame.depth_m[:] = .32
                with self.assertRaisesRegex(RuntimeError, 'surface is absent'):
                    runner._verify_destination(pose(), np.array([0.,0.,.3]), 'red', [1.,0.,0.])


if __name__ == '__main__':
    unittest.main()
