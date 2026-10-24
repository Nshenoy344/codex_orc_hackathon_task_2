"""Deterministic color routing and crash-visible transfer accounting.

A pending transfer is saved BEFORE any arm/gripper motion. After restart, a human
must reconcile physical placement rather than accidentally executing it twice.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid
from .occupancy import TemplateRegistry, occupancy_for_count, PALLET_DESTINATIONS


class PlannerError(RuntimeError):
    pass


class PendingTransferError(PlannerError):
    pass


class UnknownColorError(PlannerError):
    pass


def _positive_count(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer or null')
    return value


class SortPlanner:
    """config: good_colors, bad_colors, requested_colors, optional counts.

    requested_count limits the total mixture size, not each color. Ordered
    sequences require extra buffering and are deliberately rejected. The source
    count must include every brick, including scrap. state_path=None is useful
    only for simulation; hardware execution should always supply a journal path.
    """
    def __init__(self, config, state_path=None, templates=None, *, _allow_pending=False):
        self.config = deepcopy(config)
        self.good = self._labels(config.get('good_colors', []), 'good_colors')
        self.bad = self._labels(config.get('bad_colors', []), 'bad_colors')
        self.requested = self._labels(config.get('requested_colors', []), 'requested_colors')
        if not self.good or not self.requested:
            raise ValueError('good_colors and requested_colors must be nonempty')
        if set(self.good) & set(self.bad):
            raise ValueError('Good and bad colors overlap')
        if not set(self.requested) <= set(self.good):
            raise ValueError('Requested colors must be good colors')
        if config.get('sequence'):
            raise ValueError('Ordered sequences unsupported: mixture accepts colors in arbitrary order')
        self.requested_count = _positive_count(config.get('requested_count'), 'requested_count')
        self.source_count = _positive_count(config.get('source_count'), 'source_count')
        leftovers = [c for c in self.good if c not in self.requested]
        if len(leftovers) > 2:
            raise ValueError('Only upper/lower slots available for at most two leftover colors')
        self.leftover_routes = dict(zip(leftovers, ('upper', 'lower')))
        if templates is not None and not isinstance(templates, TemplateRegistry):
            templates = (TemplateRegistry.from_dict(templates) if templates.get('format')
                         else TemplateRegistry(templates))
        self.templates = templates
        if templates is not None:
            required = {'requested'} | set(self.leftover_routes.values())
            if not required <= templates.destinations:
                raise ValueError('Measured templates missing destinations: ' + ', '.join(sorted(required-templates.destinations)))
        self.path = Path(state_path).expanduser() if state_path is not None else None
        signature = dict(good_colors=self.good, bad_colors=self.bad,
                         requested_colors=self.requested, requested_count=self.requested_count,
                         source_count=self.source_count)
        if self.templates is not None:
            signature['template_signature'] = self.templates.signature
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        self._state = dict(version=1, config_signature=self.signature,
                           counts=dict(requested=0, upper=0, lower=0, scrap=0),
                           processed=0, pending=None, history=[])
        if self.path is not None and self.path.exists():
            with self.path.open() as stream:
                self._state = json.load(stream)
            self._validate_loaded()
            if self._state['pending'] is not None and not _allow_pending:
                raise PendingTransferError(
                    f'Unresolved transfer in {self.path}: inspect the robot and journal before resuming')

    @staticmethod
    def _labels(values, name):
        if not isinstance(values, (list, tuple)) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f'{name} must be a list of nonempty strings')
        if len(values) != len(set(values)):
            raise ValueError(f'{name} has duplicate labels')
        return list(values)

    def _validate_loaded(self):
        s = self._state
        if not isinstance(s, dict):
            raise PlannerError('Journal must be a JSON object')
        if s.get('version') != 1 or s.get('config_signature') != self.signature:
            raise PlannerError('Journal configuration differs; use a new journal for a new sorting run')
        if set(s.get('counts', {})) != {'requested', 'upper', 'lower', 'scrap'}:
            raise PlannerError('Invalid destination counters in journal')
        for value in list(s['counts'].values()) + [s.get('processed')]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PlannerError('Invalid journal counter')
        if sum(s['counts'].values()) != s['processed']:
            raise PlannerError('Journal counter totals do not agree')
        if 'pending' not in s or not isinstance(s.get('history'), list):
            raise PlannerError('Invalid transfer journal')
        if self.source_count is not None and s['processed'] > self.source_count:
            raise PlannerError('Journal exceeds source count')
        if self.requested_count is not None and s['counts']['requested'] > self.requested_count:
            raise PlannerError('Journal exceeds requested count')
        composition = self._composition(s)
        if any(sum(composition[slot].values()) != count for slot, count in s['counts'].items()):
            raise PlannerError('Committed color history disagrees with destination counters')
        self._occupancy(s)
        pending = s['pending']
        if pending is not None:
            required = {'id', 'color', 'destination', 'stack_count_before',
                        'source_count_before', 'motion_started'}
            if (not isinstance(pending, dict) or not required.issubset(pending) or
                    not isinstance(pending['id'], str) or not pending['id'] or
                    not isinstance(pending['motion_started'], bool)):
                raise PlannerError('Invalid pending transfer record')
            try:
                expected_destination = self.route(pending['color'])
            except (TypeError, UnknownColorError) as exc:
                raise PlannerError('Invalid pending transfer color') from exc
            if pending['destination'] != expected_destination:
                raise PlannerError('Pending destination does not match color route')
            if (pending['stack_count_before'] != s['counts'][expected_destination] or
                    pending['source_count_before'] != self.source_remaining):
                raise PlannerError('Pending transfer counter snapshots disagree with journal')
            if self.complete:
                raise PlannerError('Pending transfer exists after sorting run completion')
            self._validate_placement(pending, expected_destination, s['counts'][expected_destination])

    def _placement(self, destination, count):
        if destination == 'scrap':
            return dict(slot_index=None, layer_index=None, ridge_index=None, ridge_position=None)
        if self.templates is not None:
            return self.templates.placement(destination, count)
        layer, slot = divmod(count, 12)
        return dict(slot_index=slot, layer_index=layer, ridge_index=slot//3, ridge_position=slot%3)

    def _validate_placement(self, entry, destination, count):
        expected = self._placement(destination, count)
        for key, value in expected.items():
            # Unregistered old journals support routing-only inspection, never
            # physical pallet execution. A registry changes the journal signature.
            if key not in entry and self.templates is None:
                continue
            if entry.get(key) != value:
                raise PlannerError(f'Journal placement {key} disagrees with pallet template/occupancy')

    def _occupancy(self, state):
        committed = dict.fromkeys(PALLET_DESTINATIONS, 0)
        for entry in state['history']:
            if entry.get('reconciliation') == 'returned':
                continue
            destination = entry['destination']
            if destination in committed:
                self._validate_placement(entry, destination, committed[destination])
                committed[destination] += 1
        return {destination: occupancy_for_count(committed[destination])
                for destination in PALLET_DESTINATIONS}

    def _composition(self, state):
        composition = {slot: {} for slot in ('requested', 'upper', 'lower', 'scrap')}
        committed_ids = set()
        for entry in state['history']:
            if not isinstance(entry, dict):
                raise PlannerError('Invalid color history entry')
            outcome = entry.get('reconciliation')
            if outcome == 'returned':
                continue
            if outcome not in (None, 'placed'):
                raise PlannerError('Invalid reconciliation outcome in history')
            color, destination, transfer_id = (entry.get(k) for k in ('color', 'destination', 'id'))
            if not isinstance(transfer_id, str) or not transfer_id or transfer_id in committed_ids:
                raise PlannerError('Invalid or duplicate committed transfer in history')
            if not isinstance(color, str) or destination != self.route(color):
                raise PlannerError('Committed color history has an invalid destination')
            committed_ids.add(transfer_id)
            colors = composition[destination]
            colors[color] = colors.get(color, 0) + 1
        return composition

    @property
    def composition(self):
        """Actual committed shade counts, excluding pending/returned transfers."""
        return self._composition(self._state)

    @property
    def state(self):
        state = deepcopy(self._state)
        state['composition'] = self._composition(state)
        state['occupancy'] = self._occupancy(state)
        state['template_signature'] = self.templates.signature if self.templates else None
        return state

    @property
    def complete(self):
        return ((self.requested_count is not None and
                 self._state['counts']['requested'] >= self.requested_count) or
                (self.source_count is not None and self._state['processed'] >= self.source_count))

    @property
    def source_remaining(self):
        return None if self.source_count is None else self.source_count - self._state['processed']

    def route(self, color):
        if color in self.bad:
            return 'scrap'
        if color in self.requested:
            return 'requested'
        if color in self.leftover_routes:
            return self.leftover_routes[color]
        raise UnknownColorError(f'Unrecognized or uncertain color {color!r}; reobserve before motion')

    def _save(self, state):
        state['composition'] = self._composition(state)
        state['occupancy'] = self._occupancy(state)
        state['template_signature'] = self.templates.signature if self.templates else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            filename = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent, delete=False) as f:
                    filename = f.name
                    json.dump(state, f, indent=2)
                    f.write('\n')
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(filename, self.path)
                parent_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            finally:
                if filename and os.path.exists(filename):
                    os.unlink(filename)
        self._state = state

    def reserve(self, color):
        if self._state['pending'] is not None:
            raise PendingTransferError('A transfer is already pending')
        if self.complete:
            raise PlannerError('Sorting run is complete')
        destination = self.route(color)
        pending = dict(id=uuid.uuid4().hex, color=color, destination=destination,
                       stack_count_before=self._state['counts'][destination],
                       source_count_before=self.source_remaining, motion_started=False)
        pending.update(self._placement(destination, pending['stack_count_before']))
        state = deepcopy(self._state)
        state['pending'] = pending
        self._save(state)
        return deepcopy(pending)

    def _pending(self, transfer_id):
        pending = self._state['pending']
        if pending is None or pending['id'] != transfer_id:
            raise PendingTransferError('No matching pending transfer')
        return pending

    def mark_motion_started(self, transfer_id):
        self._pending(transfer_id)
        state = deepcopy(self._state)
        state['pending']['motion_started'] = True
        self._save(state)

    def commit(self, transfer_id, verified_release=True):
        pending = self._pending(transfer_id)
        if not verified_release:
            raise PlannerError('Placement/release must be verified before committing counts')
        if not pending['motion_started']:
            raise PlannerError('Cannot commit a transfer before motion starts')
        state = deepcopy(self._state)
        state['counts'][pending['destination']] += 1
        state['processed'] += 1
        state['history'].append(deepcopy(pending))
        state['pending'] = None
        self._save(state)
        return self.state

    def abort(self, transfer_id):
        pending = self._pending(transfer_id)
        if pending['motion_started']:
            raise PendingTransferError('Motion already started; reconcile physical placement before recovery')
        state = deepcopy(self._state)
        state['pending'] = None
        self._save(state)


def reconcile_journal(config, state_path, transfer_id, outcome, templates=None):
    """Record an explicitly verified physical recovery without moving hardware.

    Call only after the previous runner has stopped and physical state has been
    inspected. ``placed`` means exactly one pending brick is at its reserved
    destination, source has one fewer brick, and gripper is empty. ``returned``
    means source/destination stacks match pre-transfer counts and gripper is
    empty. Moving a brick elsewhere is neither outcome: restore one of these
    states before reconciliation. A changed config/id or a second reconciliation
    is rejected. Normal constructors still refuse unresolved pending transfers.
    """
    if outcome not in ('placed', 'returned'):
        raise ValueError("Reconciliation outcome must be 'placed' or 'returned'")
    path = Path(state_path).expanduser()
    if not path.is_file():
        raise PlannerError(f'No existing transfer journal at {path}')
    planner = SortPlanner(config, path, templates, _allow_pending=True)
    pending = planner._pending(transfer_id)
    stamp = datetime.now(timezone.utc).isoformat()
    if outcome == 'placed':
        if not pending['motion_started']:
            raise PlannerError('Cannot reconcile placed: journal records that motion never started')
        # One atomic update includes both count commit and reconciliation audit.
        state = deepcopy(planner._state)
        state['counts'][pending['destination']] += 1
        state['processed'] += 1
        state['history'].append(dict(deepcopy(pending), reconciliation='placed',
                                     reconciled_at_utc=stamp))
    else:
        state = deepcopy(planner._state)
        state['history'].append(dict(deepcopy(pending), reconciliation='returned',
                                     reconciled_at_utc=stamp))
    state['pending'] = None
    planner._save(state)
    return planner.state
