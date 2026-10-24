import numpy as np
from lego_sorter.shallow_stud_trial import generate_trial


def test_unknown_body_cannot_authorize_search():
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=.3
    r=generate_trial(T,[0,0,1],.05,.0005)
    assert not r['trial_ready'] and not r['steps']


def test_verified_bounds_generate_only_bounded_steps():
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=.3
    r=generate_trial(T,[0,0,1],.05,.0001,.048,.0001,True)
    assert r['trial_ready']
    assert max(x['descent_m']for x in r['steps'])<=.0018
    assert min(x['minimum_toe_body_clearance_m']for x in r['steps'])>=0
    assert max(x['conservative_toe_insertion_m']for x in r['steps'])<=.002


def test_isolated_empirical_protocol_requires_measured_lower_neighbors():
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=.3
    r=generate_trial(T,[0,0,1],.05,.0005,isolated_prototype_only=True)
    assert not r['trial_ready']
    r=generate_trial(T,[0,0,1],.05,.0005,isolated_prototype_only=True,
                     lower_neighbors_verified=True,lower_neighbor_upper_offset_m=.039)
    assert r['trial_ready'] and r['isolated_prototype_only']
    assert not r['body_reference_verified']
    assert max(s['descent_m']for s in r['steps'])<=.0018
