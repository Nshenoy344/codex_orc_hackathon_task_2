import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np
import yaml
from .geometry import matrix
from .planner import SortPlanner, reconcile_journal
from .occupancy import TemplateRegistry


def load_config(path):
    path = Path(path).resolve()
    with path.open() as f:
        c = yaml.safe_load(f)
    # The provided config lives in repo/config; data paths use repository root.
    root = path.parent.parent if path.parent.name == 'config' else path.parent
    c.setdefault('templates_file', 'data/templates.yaml')
    for key in ('journal_file', 'poses_file', 'handeye_file', 'color_model_file', 'templates_file'):
        p = Path(c[key]).expanduser()
        c[key] = str(p if p.is_absolute() else root / p)
    return c


def load_poses(c):
    with open(c['poses_file']) as f:
        document = yaml.safe_load(f)
    if document.get('base_frame') != c['motion']['base_frame'] or document.get('tcp_frame') != c['motion']['tcp_frame']:
        raise ValueError('Taught poses base/TCP frames differ from motion configuration')
    return document['poses']


def load_handeye(c, *, commissioning_only=False):
    """Load a physical calibration, or explicitly inspect a numerical candidate.

    Standard sorting never enables commissioning_only. Point-trial tooling may
    opt into an accepted numerical candidate without silently authorizing a job.
    """
    with open(c['handeye_file']) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError('Hand-eye artifact must be a mapping')
    p = data.get('parameters', {})
    for key in ('base_frame', 'tcp_frame'):
        if p.get(key) != c['motion'][key]:
            raise ValueError(f'Hand-eye {key} differs from motion frame')
    if 'T_tcp_color' not in data:
        raise ValueError('Need T_tcp_color for depth aligned to color, not native depth extrinsics')
    transform = matrix(data['T_tcp_color'])
    diagnostics = data.get('diagnostics', {})
    if not isinstance(diagnostics, dict):
        raise ValueError('Hand-eye diagnostic provenance must be a mapping')
    if diagnostics.get('accepted') is False:
        raise ValueError('Rejected numerical hand-eye candidate cannot be used for motion')
    if data.get('physical_commissioning_verified') is not True:
        if not commissioning_only:
            raise ValueError('Hand-eye calibration is not physically commissioned; standard sorting refuses numerical candidates')
        if diagnostics.get('accepted') is not True:
            raise ValueError('Commissioning-only use requires an accepted numerical hand-eye candidate')
    return transform


def load_handeye_candidate(c):
    """Explicit commissioning-only path; never used by the physical sorting CLI."""
    return load_handeye(c, commissioning_only=True)


def load_templates(c):
    templates = TemplateRegistry.load(c['templates_file'])
    if templates.base_frame != c['motion']['base_frame'] or templates.tcp_frame != c['motion']['tcp_frame']:
        raise ValueError('Measured template frames differ from motion configuration')
    routes=SortPlanner(c['job']).leftover_routes
    required={'requested'} | set(routes.values())
    if not required <= templates.destinations:
        raise ValueError(f'Missing measured output templates: {sorted(required-templates.destinations)}')
    return templates


def model(c):
    from .perception import ColorModel
    colors = c['colors']
    result = ColorModel.load(c['color_model_file'], colors['max_distance'], colors['min_margin'])
    required = set(c['job']['good_colors'] + c['job']['bad_colors'])
    result.validate_labels(required)
    if not required <= set(result.samples):
        raise ValueError(f'Missing taught colors: {sorted(required-set(result.samples))}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Franka LEGO sorting orchestrator')
    parser.add_argument('--config', default='config/demo.yaml')
    sub = parser.add_subparsers(dest='command', required=True)
    sim = sub.add_parser('simulate', help='Offline routing simulation; no robot/camera imports')
    sim.add_argument('--colors', default='300121,6135191,6117418,6487946,4211201,300121')
    sim.add_argument('--requested', nargs='+', help='Override requested color set')
    sub.add_parser('check', help='Check local commissioned inputs without hardware access')
    sub.add_parser('inspect-calibration-candidate', help='Read-only numerical candidate inspection; does not authorize physical sorting')
    sub.add_parser('pending', help='Read the current transfer journal without hardware access')
    recovery = sub.add_parser('reconcile', help='Record physical recovery of a stopped pending transfer')
    recovery.add_argument('transfer_id')
    recovery.add_argument('outcome', choices=['placed', 'returned'])
    recovery.add_argument('--physical-state-verified', action='store_true')
    sub.add_parser('robot-status', help='Read-only ROS endpoints, TF, joint state check')
    teach = sub.add_parser('teach-pose', help='Record current ROS TCP pose; sends no commands')
    teach.add_argument('name', choices=['observation','pick_reference','scrap'])
    run = sub.add_parser('run', help='Camera-guided physical sorting')
    run.add_argument('--execute', action='store_true', help='Enable actuator commands')
    run.add_argument('--max-transfers', type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    c = load_config(args.config)
    if args.command == 'inspect-calibration-candidate':
        transform=load_handeye_candidate(c)
        print(json.dumps(dict(commissioning_only=True,
            full_sorting_authorized_by_this_command=False,
            calibration_file=c['handeye_file'],T_tcp_color=transform.tolist()),indent=2))
        return 0
    if args.command == 'pending':
        print(Path(c['journal_file']).read_text())
        return 0
    if args.command == 'reconcile':
        if not args.physical_state_verified:
            parser.error('Inspect and restore the physical state first; then pass --physical-state-verified')
        state = reconcile_journal(c['job'], c['journal_file'], args.transfer_id, args.outcome,
                                  templates=load_templates(c))
        print(json.dumps(state, indent=2))
        return 0
    if args.command == 'simulate':
        if args.requested:
            c['job']['requested_colors'] = args.requested
        plan = SortPlanner(c['job'])
        for color in args.colors.split(','):
            if plan.complete:
                break
            t = plan.reserve(color.strip())
            print(f"{color}: {t['destination']}")
            plan.mark_motion_started(t['id'])
            plan.commit(t['id'], verified_release=True)
        print(json.dumps({key:plan.state[key] for key in ('counts','composition','occupancy')}, indent=2))
        return 0
    if args.command == 'check':
        issues = []
        for name, fn in [('poses', lambda: load_poses(c)), ('calibration', lambda: load_handeye(c)),
                         ('colors', lambda: model(c)), ('output templates', lambda: load_templates(c)),
                         ('journal/routing', lambda: SortPlanner(c['job'], c['journal_file'],
                             templates=load_templates(c) if Path(c['templates_file']).exists() else None))]:
            try:
                fn()
                print(name+': OK')
            except Exception as exc:
                issues.append(name+': '+str(exc))
        for key in ('transit_height_m','workspace_min_m','workspace_max_m','source_arrangement'):
            if c['execution'].get(key) is None:
                issues.append('execution.'+key+': not taught')
        if c['perception'].get('roi') is None:
            issues.append('perception.roi: not taught')
        if c['job'].get('source_count') is None and c['job'].get('requested_count') is None:
            issues.append('job: set source_count or requested_count')
        for issue in issues:
            print('MISSING: '+issue)
        return 1 if issues else 0
    if args.command == 'run':
        if not args.execute:
            parser.error('Physical sorting requires run --execute; use simulate or check first')
        if args.max_transfers is not None and args.max_transfers < 1:
            parser.error('--max-transfers must be positive')
        # Resolve all missing commissioning files before creating ROS entities.
        poses, tcp_color, colors = load_poses(c), load_handeye(c), model(c)
        templates=load_templates(c)
    from .motion import RosRobot
    robot = RosRobot(c)
    try:
        if args.command == 'robot-status':
            report = robot.preflight()
            print(json.dumps(report, indent=2))
            return 0 if report['ready'] else 1
        if args.command == 'teach-pose':
            pose = robot.current_pose()
            path = Path(c['poses_file'])
            if path.exists():
                document = yaml.safe_load(path.read_text())
                load_poses(c)  # refuse mixing TCP/base frames
            else:
                document = dict(base_frame=c['motion']['base_frame'], tcp_frame=c['motion']['tcp_frame'], poses={})
            document['poses'][args.name] = pose.tolist()
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            temp.write_text(yaml.safe_dump(document, sort_keys=False))
            temp.replace(path)
            print(f'Saved {args.name} to {path}; no robot motion sent')
            return 0
        from .pallet_runner import PalletRunner
        from .camera import RealSenseCamera
        # Validate geometry and journal before opening camera or sending commands.
        runner = PalletRunner(c, poses, tcp_color, colors, robot, None, templates)
        report = robot.preflight()
        if not report['ready']:
            raise RuntimeError('Robot endpoints/state not ready: '+json.dumps(report))
        with RealSenseCamera(c['camera']) as camera:
            runner.camera = camera
            state = runner.run(args.max_transfers)
        print(json.dumps({key:state[key] for key in ('counts','composition','occupancy')}, indent=2))
        quota = c['job'].get('requested_count')
        if quota is not None and state['counts']['requested'] < quota and runner.planner.complete:
            print('Source exhausted before requested quota was met', file=sys.stderr)
            return 2
        return 0
    finally:
        robot.close()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; inspect any pending transfer journal before resuming', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logging.exception('Stopped: %s', exc)
        sys.exit(1)
