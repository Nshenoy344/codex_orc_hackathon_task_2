import json
from pathlib import Path
import tempfile
import unittest
from lego_sorter.occupancy import TemplateRegistry
from lego_sorter.planner import SortPlanner, PlannerError, PendingTransferError, UnknownColorError, reconcile_journal


class PlannerTest(unittest.TestCase):
    def config(self, **kwargs):
        return dict(good_colors=['red', 'dark_red', 'orange'], bad_colors=['tan', 'brown'],
                    requested_colors=['red'], source_count=10, **kwargs)

    def transfer(self, planner, color):
        t = planner.reserve(color)
        planner.mark_motion_started(t['id'])
        planner.commit(t['id'])
        return t

    def test_single_color_routes_and_independent_heights(self):
        p = SortPlanner(self.config())
        self.assertEqual([p.route(c) for c in ['red', 'dark_red', 'orange', 'tan', 'brown']],
                         ['requested', 'upper', 'lower', 'scrap', 'scrap'])
        first = self.transfer(p, 'dark_red')
        second = self.transfer(p, 'dark_red')
        self.assertEqual(first['stack_count_before'], 0)
        self.assertEqual(second['stack_count_before'], 1)
        self.assertEqual(p.state['counts']['requested'], 0)
        self.assertEqual(p.source_remaining, 8)

    def test_mixture_routes_arbitrary_order(self):
        cfg = self.config()
        cfg['requested_colors'] = ['red', 'orange']
        p = SortPlanner(cfg)
        self.assertEqual(p.route('dark_red'), 'upper')
        self.assertEqual(self.transfer(p, 'orange')['destination'], 'requested')
        self.assertEqual(self.transfer(p, 'red')['stack_count_before'], 1)

    def test_unknown_never_scrap_or_pending(self):
        p = SortPlanner(self.config())
        with self.assertRaises(UnknownColorError):
            p.reserve('unknown')
        self.assertIsNone(p.state['pending'])

    def test_pending_restart_refused_and_pre_motion_abort(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'journal.json'
            p = SortPlanner(self.config(), path)
            t = p.reserve('red')
            self.assertTrue(path.exists())
            with self.assertRaises(PendingTransferError):
                SortPlanner(self.config(), path)
            p.abort(t['id'])
            resumed = SortPlanner(self.config(), path)
            self.assertEqual(resumed.state['processed'], 0)

    def test_motion_abort_and_unverified_commit_refused(self):
        p = SortPlanner(self.config())
        t = p.reserve('red')
        with self.assertRaises(PlannerError):
            p.commit(t['id'])
        p.mark_motion_started(t['id'])
        with self.assertRaises(PendingTransferError):
            p.abort(t['id'])
        with self.assertRaises(PlannerError):
            p.commit(t['id'], verified_release=False)
        self.assertEqual(p.state['processed'], 0)

    def test_quota_stop(self):
        p = SortPlanner(self.config(requested_count=1))
        self.transfer(p, 'tan')
        self.assertFalse(p.complete)
        self.transfer(p, 'red')
        self.assertTrue(p.complete)
        with self.assertRaises(PlannerError):
            p.reserve('orange')

    def test_source_count_stop_and_resume(self):
        cfg = self.config()
        cfg['source_count'] = 1
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'journal.json'
            p = SortPlanner(cfg, path)
            self.transfer(p, 'brown')
            self.assertTrue(SortPlanner(cfg, path).complete)
            changed = dict(cfg, requested_colors=['orange'])
            with self.assertRaises(PlannerError):
                SortPlanner(changed, path)

    def test_sequence_and_bad_requested_refused(self):
        with self.assertRaises(ValueError):
            SortPlanner(dict(self.config(), sequence=['red', 'orange']))
        with self.assertRaises(ValueError):
            SortPlanner(dict(self.config(), requested_colors=['tan']))

    def test_double_commit_refused(self):
        p = SortPlanner(self.config())
        t = self.transfer(p, 'red')
        with self.assertRaises(PendingTransferError):
            p.commit(t['id'])
        self.assertEqual(p.state['processed'], 1)

    def test_reconcile_placed_commits_once_and_audits(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'journal.json'
            p = SortPlanner(self.config(), path)
            t = p.reserve('red')
            p.mark_motion_started(t['id'])
            state = reconcile_journal(self.config(), path, t['id'], 'placed')
            self.assertEqual(state['counts']['requested'], 1)
            self.assertEqual(state['processed'], 1)
            self.assertIsNone(state['pending'])
            self.assertEqual(state['history'][-1]['reconciliation'], 'placed')
            self.assertEqual(SortPlanner(self.config(), path).source_remaining, 9)
            with self.assertRaises(PendingTransferError):
                reconcile_journal(self.config(), path, t['id'], 'placed')

    def test_reconcile_returned_preserves_counts_after_motion(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'journal.json'
            p = SortPlanner(self.config(), path)
            t = p.reserve('tan')
            p.mark_motion_started(t['id'])
            state = reconcile_journal(self.config(), path, t['id'], 'returned')
            self.assertEqual(state['processed'], 0)
            self.assertEqual(state['counts']['scrap'], 0)
            self.assertIsNone(state['pending'])
            self.assertEqual(state['history'][-1]['reconciliation'], 'returned')
            self.assertEqual(SortPlanner(self.config(), path).source_remaining, 10)

    def test_reconcile_wrong_id_config_and_outcome_do_not_change_journal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'journal.json'
            p = SortPlanner(self.config(), path)
            t = p.reserve('red')
            p.mark_motion_started(t['id'])
            original = path.read_text()
            with self.assertRaises(PendingTransferError):
                reconcile_journal(self.config(), path, 'wrong', 'placed')
            with self.assertRaises(PlannerError):
                reconcile_journal(dict(self.config(), requested_colors=['orange']), path, t['id'], 'placed')
            with self.assertRaises(ValueError):
                reconcile_journal(self.config(), path, t['id'], 'discarded')
            self.assertEqual(path.read_text(), original)

    def test_reconcile_requires_existing_journal_and_started_placement(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'journal.json'
            with self.assertRaises(PlannerError):
                reconcile_journal(self.config(), path, 'missing', 'returned')
            p = SortPlanner(self.config(), path)
            t = p.reserve('red')
            with self.assertRaises(PlannerError):
                reconcile_journal(self.config(), path, t['id'], 'placed')
            state = reconcile_journal(self.config(), path, t['id'], 'returned')
            self.assertIsNone(state['pending'])

    def test_corrupt_pending_snapshot_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'journal.json'
            p = SortPlanner(self.config(), path)
            t = p.reserve('red')
            state = json.loads(path.read_text())
            state['pending']['stack_count_before'] = 99
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(PlannerError, 'snapshots'):
                reconcile_journal(self.config(), path, t['id'], 'returned')

    def test_actual_composition_reports_mixture_and_separate_scrap_shades(self):
        cfg = dict(self.config(), requested_colors=['red', 'orange'])
        p = SortPlanner(cfg)
        for color in ['red', 'orange', 'red', 'dark_red', 'tan', 'brown', 'tan']:
            self.transfer(p, color)
        expected = dict(requested=dict(red=2, orange=1), upper=dict(dark_red=1),
                        lower={}, scrap=dict(tan=2, brown=1))
        self.assertEqual(p.composition, expected)
        self.assertEqual(p.state['composition'], expected)
        p.state['composition']['requested']['red'] = 900
        self.assertEqual(p.composition['requested']['red'], 2)

    def test_composition_excludes_pending_and_returned_but_counts_reconciled_placed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'journal.json'
            p = SortPlanner(self.config(), path)
            t = p.reserve('red')
            p.mark_motion_started(t['id'])
            self.assertEqual(p.composition['requested'], {})
            state = reconcile_journal(self.config(), path, t['id'], 'returned')
            self.assertEqual(state['composition']['requested'], {})
            p = SortPlanner(self.config(), path)
            t = p.reserve('red')
            p.mark_motion_started(t['id'])
            state = reconcile_journal(self.config(), path, t['id'], 'placed')
            self.assertEqual(state['composition']['requested'], {'red': 1})
            self.assertEqual(json.loads(path.read_text())['composition']['requested'], {'red': 1})

    def test_composition_reconstructs_older_journal_and_refuses_counter_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'journal.json'
            p = SortPlanner(self.config(), path)
            self.transfer(p, 'red')
            state = json.loads(path.read_text())
            state.pop('composition')
            path.write_text(json.dumps(state))
            self.assertEqual(SortPlanner(self.config(), path).state['composition']['requested'], {'red': 1})
            state['history'] = []
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(PlannerError, 'color history'):
                SortPlanner(self.config(), path)

    def templates(self, offset=0.):
        templates = {}
        for d, name in enumerate(('requested','upper','lower')):
            poses = []
            for i in range(12):
                poses.append([[1.,0.,0., .2+d*.15+(i//3)*.022+offset],
                              [0.,1.,0., -.1+(i%3)*.035+(i//3)*.001],
                              [0.,0.,1., .03+i*.0001], [0.,0.,0.,1.]])
            templates[name] = dict(kind='fixture',
                first_place_reference='tcp_pose_for_first_new_layer',
                layer_pitch_m=.0113, first_place_poses=poses,rotation_center_base_m=[.233+d*.15+offset,-.064,.03],layer_rotation_deg=90,first_new_layer_rotation_from_reference_deg=90)
        return TemplateRegistry(templates)

    def test_pallet_layer_progression_and_destinations_independent(self):
        cfg = dict(self.config(), source_count=40)
        registry = self.templates()
        p = SortPlanner(cfg, templates=registry)
        for i in range(13):
            t = self.transfer(p, 'red')
            self.assertEqual((t['slot_index'],t['layer_index']), (i%12,i//12))
            self.assertEqual(t['placement_pose'],registry.placement('requested',i)['placement_pose'])
        self.transfer(p, 'dark_red')
        self.transfer(p, 'dark_red')
        self.transfer(p, 'orange')
        self.transfer(p, 'tan')
        occupancy = p.state['occupancy']
        self.assertEqual(occupancy['requested']['slot_counts'], [2]+[1]*11)
        self.assertEqual(occupancy['requested']['completed_layers'],1)
        self.assertEqual(occupancy['upper']['slot_counts'],[1,1]+[0]*10)
        self.assertEqual(occupancy['lower']['slot_counts'],[1]+[0]*11)
        self.assertNotIn('scrap',occupancy)

    def test_pending_reservation_does_not_advance_pallet_occupancy(self):
        cfg = dict(self.config(),source_count=40)
        p = SortPlanner(cfg,templates=self.templates())
        t = p.reserve('red')
        self.assertEqual((t['slot_index'],t['layer_index']), (0,0))
        self.assertEqual(p.state['occupancy']['requested']['slot_counts'],[0]*12)
        p.abort(t['id'])
        self.assertEqual(p.reserve('red')['slot_index'],0)

    def test_template_geometry_change_refuses_existing_journal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'journal.json'
            p = SortPlanner(self.config(),path,self.templates())
            self.transfer(p,'red')
            self.assertEqual(SortPlanner(self.config(),path,self.templates()).state['occupancy']['requested']['next_slot_index'],1)
            with self.assertRaisesRegex(PlannerError,'configuration differs'):
                SortPlanner(self.config(),path,self.templates(offset=.001))
            with self.assertRaises(PlannerError):
                SortPlanner(self.config(),path)

    def test_pallet_pending_reconciliation_advances_exact_slot_only_when_placed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'journal.json'
            registry = self.templates()
            p = SortPlanner(self.config(),path,registry)
            t = p.reserve('dark_red')
            p.mark_motion_started(t['id'])
            returned = reconcile_journal(self.config(),path,t['id'],'returned',registry)
            self.assertEqual(returned['occupancy']['upper']['next_slot_index'],0)
            p = SortPlanner(self.config(),path,registry)
            t = p.reserve('dark_red')
            p.mark_motion_started(t['id'])
            placed = reconcile_journal(self.config(),path,t['id'],'placed',registry)
            self.assertEqual(placed['occupancy']['upper']['slot_counts'],[1]+[0]*11)
            self.assertEqual(placed['occupancy']['requested']['slot_counts'],[0]*12)

    def test_missing_required_destination_template_rejected(self):
        registry = self.templates().to_dict()
        del registry['destinations']['lower']
        with self.assertRaisesRegex(ValueError,'missing destinations'):
            SortPlanner(self.config(),templates=TemplateRegistry.from_dict(registry))


if __name__ == '__main__':
    unittest.main()
