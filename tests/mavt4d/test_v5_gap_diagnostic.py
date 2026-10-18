from dataclasses import replace
import pytest
import torch
from mavt4d.object_state.v5.config import V5Config
from mavt4d.object_state.v5.model import V5Model


def model():
    torch.manual_seed(31)
    return V5Model(V5Config(slots=3,width=12,evidence_width=12,spatial_heads=3,resampler_heads=3,
                           spatial_blocks=1,content_tokens=2,dynamic_tokens=2,samples_per_ray=4))


def test_frozen_gap_scene_has_real_crossing_exact_motion_and_two_cameras():
    from mavt4d.object_state.v5.diagnostic import make_gap_scene
    a=make_gap_scene(17,0); b=make_gap_scene(17,0)
    assert a['rgb'].shape==(24,2,3,64,64)
    assert a['crossing'] and a['same_appearance']
    assert torch.equal(a['rgb'],b['rgb'])
    assert a['object_poses'].shape==(24,2,4,4)
    torch.testing.assert_close(a['camera_c2w'][0,0],torch.eye(4),atol=1e-6,rtol=1e-6)
    # Relative displacement changes sign along the designed crossing axis.
    delta=a['object_poses'][:,0,:3,3]-a['object_poses'][:,1,:3,3]
    assert (delta[0]*a['crossing_axis']).sum()*(delta[-1]*a['crossing_axis']).sum()<0
    assert not make_gap_scene(17,1)['crossing']


def test_gap_observation_matches_natural_rgb_normalization(monkeypatch):
    from mavt4d.object_state.v5.diagnostic import make_gap_scene, _run_gap_batch
    sample=make_gap_scene(17,0)
    original_rgb=sample['rgb'].clone()
    captured=[]
    net=model()
    def capture(state, observation, **kwargs):
        captured.append(observation.images)
        raise RuntimeError('captured first observation')
    monkeypatch.setattr(net,'observe',capture)
    with pytest.raises(RuntimeError,match='captured first observation'):
        _run_gap_batch(net,[sample],4,17)
    torch.testing.assert_close(captured[0],sample['rgb'][None,0,:1,None]*2-1)
    torch.testing.assert_close(sample['rgb'],original_rgb,rtol=0,atol=0)


def test_budget_expiry_and_small_suite_never_claim_complete(tmp_path):
    from mavt4d.object_state.v5.diagnostic import prepare_gap_suite, evaluate_gap_diagnostic
    prepare_gap_suite(tmp_path,seed=17,source_count=4)
    first=evaluate_gap_diagnostic(model(),17,seconds=0,suite_dir=tmp_path)
    assert not first['complete'] and first['assignment_accuracy'] is None
    assert first['evaluated_cases']==0 and first['required_cases']==800
    out=evaluate_gap_diagnostic(model(),17,seconds=60,suite_dir=tmp_path,batch_size=4)
    assert out['evaluated_cases']==16
    assert not out['complete'] and out['assignment_accuracy'] is None
    assert out['partial_assignment_accuracy'] is not None
    assert set(out['gaps'])=={'2','4','8','16'}
    assert out['status_labeled_rays']>0 and out['rgb_rays']>0


def test_hidden_pixels_and_heldout_camera_never_change_filter_state():
    from mavt4d.object_state.v5.diagnostic import make_gap_scene, _run_gap_batch
    sample=make_gap_scene(17,0)
    hidden={**sample,'rgb':sample['rgb'].clone(),'depth':sample['depth'].clone()}
    hidden['rgb'][:,1]=1-hidden['rgb'][:,1]
    hidden['rgb'][4:8,0]=1-hidden['rgb'][4:8,0]
    hidden['depth'][4:8,0]*=3
    net=model().eval()
    with torch.no_grad():
        sa,ma=_run_gap_batch(net,[sample],4,17)
        sb,mb=_run_gap_batch(net,[hidden],4,17)
    torch.testing.assert_close(sa.c,sb.c,rtol=0,atol=0)
    torch.testing.assert_close(sa.d,sb.d,rtol=0,atol=0)
    assert ma['assignment_count']>0 and ma['rgb_rays']>0


def test_suite_hash_detects_changed_content(tmp_path):
    from mavt4d.object_state.v5.diagnostic import prepare_gap_suite, evaluate_gap_diagnostic
    prepare_gap_suite(tmp_path,seed=17,source_count=1)
    path=tmp_path/'source-0000.pt'
    with path.open('ab') as f: f.write(b'changed')
    with pytest.raises(ValueError,match='hash'):
        evaluate_gap_diagnostic(model(),17,seconds=60,suite_dir=tmp_path)


def test_full_coverage_requires_all_cases_and_finite_metrics(tmp_path,monkeypatch):
    import mavt4d.object_state.v5.diagnostic as diagnostic
    monkeypatch.setattr(diagnostic,'make_gap_scene',lambda seed,index:{
        'schema':diagnostic.SCHEMA,'source_seed':diagnostic._source_seed('dev',seed,index),
        'source_index':index,'crossing':index%4==0})
    diagnostic.prepare_gap_suite(tmp_path,source_count=200)
    def scored(model,samples,gap,seed,deadline=None):
        n=len(samples)
        return None,{'assignment_correct':n,'assignment_count':2*n,
                     'assignment_valid_cases':n,'rgb_abs_sum':.1*n,'rgb_rays':n,
                     'status_labeled_rays':n,'status_correct':n}
    monkeypatch.setattr(diagnostic,'_run_gap_batch',scored)
    result=diagnostic.evaluate_gap_diagnostic(model(),suite_dir=tmp_path,seconds=60)
    assert result['complete'] and result['assignment_accuracy']==.5
    assert result['evaluated_cases']==800 and result['crossing_cases']==200
    def invalid(*args,**kwargs):
        state,metrics=scored(*args,**kwargs); metrics['rgb_abs_sum']=float('nan')
        return state,metrics
    monkeypatch.setattr(diagnostic,'_run_gap_batch',invalid)
    with pytest.raises(FloatingPointError,match='nonfinite'):
        diagnostic.evaluate_gap_diagnostic(model(),suite_dir=tmp_path,seconds=60)
