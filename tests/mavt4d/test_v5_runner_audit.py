"""Regression tests for the reviewed v5 runner/evaluator contracts."""
from concurrent.futures import ThreadPoolExecutor
import json
import pytest
import torch
from mavt4d.object_state.v5 import runner, evaluation


def test_draw_plan_is_independent_of_microbatch_and_source_order():
    a = runner.logical_draw_plan(step=4000, effective=256, weights={'openvid': .5, 'dl3dv': .5}, seed=17)
    b = runner.logical_draw_plan(step=4000, effective=256, weights={'dl3dv': .5, 'openvid': .5}, seed=17)
    assert sorted(a, key=lambda p:p['draw_id']) == sorted(b, key=lambda p:p['draw_id'])
    assert len({p['draw_id'] for p in a}) == 256
    assert len({p['seed'] for p in a}) == 256
    assert len({p['profile'] for p in a if p['source']=='openvid'}) == 3
    assert all(p['seed']==p['plan_seed'] for p in a)
    assert json.loads(json.dumps(a)) == a


def test_capacity_rejects_short_video_as_g_h6():
    clip = {'source':'openvid', 'time_valid':True, 'plan':{'profile':'S-short','future':[1,2,3,4]}}
    with pytest.raises(ValueError, match='G.*6'):
        runner.validate_capacity_clip('openvid', clip)
    clip['plan']={'profile':'G','future':list(range(6)),'observed':[True]*3+[False]*6+[True]*3}
    assert runner.validate_capacity_clip('openvid',clip)['future_targets']==6


def test_collapse_counter_requires_actual_two_diagnostics_and_survives_resume():
    state = runner.OwnershipMonitor()
    for _ in range(500): state.update(.01, 1e-8)
    assert not state.should_soften
    state.record_diagnostic(.6)
    assert not state.should_soften
    state.record_diagnostic(.5)
    assert state.should_soften
    restored = runner.OwnershipMonitor.from_dict(state.to_dict())
    assert restored.should_soften
    restored.update(.2, 1e-8)
    assert not restored.should_soften
    restored.update(float('nan'), float('nan'))
    assert restored.bad_updates==0


def test_copy_last_uses_latest_admitted_pixel_and_does_not_read_hidden_rgb():
    clip={'rgb':torch.arange(3.).view(3,1,1,1,1).expand(3,1,3,1,2).clone(),
          'rgb_valid':torch.ones(3,1,1,1,2,dtype=torch.bool),
          'plan':{'prefix':[0,1], 'observed':[True,True]}}
    target={'pixels':torch.tensor([[[.5,.5],[1.5,.5]]]), 'view':0,
            'rgb':torch.zeros(1,2,3),'rgb_valid':torch.ones(1,2,dtype=torch.bool)}
    history=[{'frame':0,'input_mask':torch.tensor([[[[[True,False]]]]])},
             {'frame':1,'input_mask':torch.tensor([[[[[False,True]]]]])}]
    before, valid=evaluation.copy_last(clip,target,frame=2,history=history)
    assert torch.equal(before['rgb'][0,:,0],torch.tensor([0.,1.]))
    assert valid['rgb_valid'].all()
    clip['rgb'][1,:,:,:,0]=99
    after,_=evaluation.copy_last(clip,target,frame=2,history=history)
    assert torch.equal(before['rgb'],after['rgb'])


def test_status_metrics_report_missing_separately_and_tie_aware_auc():
    meter=evaluation.ReadoutMetrics()
    target={'rgb':torch.zeros(1,4,3),'rgb_valid':torch.ones(1,4,dtype=torch.bool),
            'status':torch.tensor([[0,1,2,-1]])}
    meter.add({'rgb':torch.zeros(1,4,3),'status_logits':torch.zeros(1,4,3)},target)
    result=meter.result()
    assert result['status_missing_count']==1
    assert result['status_0_auroc']==.5
    assert result['status_2_auroc']==.5
    assert result['status_ece']==pytest.approx(0)
    assert result['status_risk_coverage_count']==3


def test_gap_protocol_requires_real_coverage_and_declares_baselines_untrained():
    with pytest.raises(ValueError,match='200'):
        evaluation.build_gap_protocol(['only-one'])
    protocol=evaluation.build_gap_protocol([str(i) for i in range(200)])
    assert protocol['gaps']==[2,4,8,16]
    assert protocol['steps']==24 and protocol['cameras']==2
    assert protocol['crossing_sources_required']==50
    assert all(arm['status']=='requires_independent_training' for arm in evaluation.baseline_protocol()['trained_arms'])


def test_readiness_follows_counted_slot_tier_and_real_supervision():
    receipt={'schema':'v5-readiness-v1',
             'slot_census':{'total_steps':100,'steps_over_16':21,'steps_over_32':2,'schema':'v5-slot-census-v2','input_distribution':'uniform_single_camera','input_views':['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']},
             'role_coverage':{'sources':{'arkitscenes':{'trusted_draws':2,'reuse_valid':3,'canonical_valid':3}}}}
    with pytest.raises(ValueError,match='slot tier'):
        runner.validate_readiness(receipt,{})
    runner.validate_readiness(receipt,{'slots':32,'content_tokens':4,'dynamic_tokens':4})
    receipt['role_coverage']['sources']['arkitscenes']['canonical_valid']=0
    with pytest.raises(ValueError,match='reuse/canonical'):
        runner.validate_readiness(receipt,{'slots':32,'content_tokens':4,'dynamic_tokens':4})


def test_census_uses_gt_frustum_and_unique_instance_ids():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('census',Path(__file__).resolve().parents[2]/'scripts/mavt4d/census_v5_slots.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import numpy as np
    def ann(identity, xyz):
        return {'instance_token':identity,'translation':xyz,'size':[1,1,1],'rotation':[1,0,0,0]}
    tables=SimpleNamespace(keyframe_data={'s':{'CAM_FRONT':'cam'}},
        sample_data={'cam':{'width':100,'height':100}},
        annotations_of={'s':['a','duplicate','behind','outside']},
        sample_annotation={'a':ann('one',[0,0,10]),'duplicate':ann('one',[0,0,10]),
                           'behind':ann('two',[0,0,-10]),'outside':ann('three',[1000,0,10])},
        ego_to_global=lambda _:np.eye(4),sensor_to_ego=lambda _:np.eye(4),
        raw_intrinsics=lambda _:np.array([[100,0,50],[0,100,50],[0,0,1]]))
    assert module.visible_instances(tables,'s',['CAM_FRONT'])=={'one'}


def test_openvid_plan_respects_celebv_half_cap():
    choices=runner.logical_draw_plan(step=4000,effective=256,weights={'openvid':.253,'openimages':.747},seed=1)
    seeds=[p['seed'] for p in choices if p['source']=='openvid']
    assert sum(s%2==0 for s in seeds)<=len(seeds)//2


def test_draw_plan_rounding_ties_are_source_order_independent():
    weights={'openimages':.15,'openvid':.25,'nuscenes':.3,'dl3dv':.15,'arkitscenes':.15}
    a=runner.logical_draw_plan(step=0,effective=256,weights=weights,seed=17)
    b=runner.logical_draw_plan(step=0,effective=256,weights=dict(reversed(list(weights.items()))),seed=17)
    assert a==b


def test_train_update_retains_draws_across_physical_partition(monkeypatch):
    from types import SimpleNamespace
    records=[]
    class Data:
        def draw(self,source,seed,**kw):
            return {'source':source,'draw_seed':seed,'plan':{'profile':kw['profile']}}
    def signature(clip):return clip['source'],clip['plan']['profile']
    def loss(model,clips,**kw):
        ids=[c['draw_id'] for c in clips];records.extend(ids)
        value=sum((c['draw_seed']%100)/100 for c in clips)/len(clips)
        return model.weight.sum()*value,{'current':value}
    monkeypatch.setattr(runner,'clip_signature',signature)
    monkeypatch.setattr(runner,'compute_loss',loss)
    result=[]
    for capacity in (1,32):
        model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(1)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        records.clear()
        with ThreadPoolExecutor(max_workers=3) as pool:
            metrics=runner.train_update(model,optimizer,Data(),step=4000,effective=256,
                capacities={'openvid':capacity},weights={'openvid':1.},seed=17,rays=1,pool=pool)
        result.append((set(records),metrics['loss'],model.weight.detach().clone(),metrics['sampler/plan_sha256']))
    assert result[0][0]==result[1][0]
    assert result[0][1]==pytest.approx(result[1][1],abs=1e-6)
    assert torch.allclose(result[0][2],result[1][2],atol=1e-6)
    assert result[0][3]==result[1][3]


def test_bucket_capacity_selects_measured_warmup_and_conservative_mature():
    capacities={'openvid':{'S/H1/bridge':32,'G/H6/bridge':8}}
    assert runner.capacity_for(capacities,'openvid',step=0)==32
    assert runner.capacity_for(capacities,'openvid',step=1000)==8
    assert runner.capacity_for({'openvid':4},'openvid',step=0)==4
    assert runner.halve_capacities(capacities)=={'openvid':{'S/H1/bridge':16,'G/H6/bridge':4}}


def test_census_streams_json_array_with_embedded_delimiters():
    import importlib.util, io
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('census',Path(__file__).resolve().parents[2]/'scripts/mavt4d/census_v5_slots.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=[{'s':'escaped " comma , brackets ][','token':str(i)} for i in range(3)]
    assert list(module.iter_json_array(io.StringIO(json.dumps(data)),chunk_size=5))==data
    with pytest.raises(ValueError):
        list(module.iter_json_array(io.StringIO('[{"invalid":'),chunk_size=5))


def test_status_classwise_calibration_unknown_inferred_and_hallucination():
    result=evaluation.status_calibration(torch.tensor([[.1,.8,.1],[.4,.4,.2],[.05,.05,.9]]),torch.tensor([1,2,2]))
    assert all('status_'+str(k)+'_ece' in result for k in range(3))
    assert result['status_unknown_vs_inferred_auroc']==1.
    assert result['status_unknown_hallucination_rate']==.5
    assert result['status_unknown_count']==2


def test_diagnostic_monitor_ignores_partial_accuracy():
    monitor=runner.OwnershipMonitor(bad_updates=500)
    runner.record_diagnostic_report(monitor,{'complete':False,'partial_assignment_accuracy':.1,'assignment_accuracy':None})
    assert monitor.diagnostics==[]
    runner.record_diagnostic_report(monitor,{'complete':True,'assignment_accuracy':.8})
    runner.record_diagnostic_report(monitor,{'complete':True,'assignment_accuracy':.7})
    assert monitor.should_soften


def test_census_uses_camera_draw_distribution_not_union():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('census',Path(__file__).resolve().parents[2]/'scripts/mavt4d/census_v5_slots.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.summarize_camera_counts({'front':[1,20], 'back':[1,2]})
    assert result['total_steps']==4
    assert result['steps_over_16']==1
    assert result['fraction_over_16']==.25
    assert result['per_camera']['front']['steps_over_16']==1
    assert result['per_camera']['back']['steps_over_16']==0
    assert len(module.census.__kwdefaults__['cameras'])==6


def test_capacity_probe_masks_cover_all_observed_endpoints():
    clip={'source':'openvid','time_valid':True,'draw_seed':20,
          'plan':{'profile':'G','prefix':list(range(12)), 'observed':[True]*3+[False]*6+[True]*3,'future':list(range(12,18))}}
    probes=runner.capacity_probe_batch(clip,8,mask=True)
    assert len({p['draw_seed'] for p in probes})==8
    assert {p['force_tube_frame'] for p in probes}=={0,1,2,9,10,11}
    assert all(p['force_bridge'] and p['force_tube_mask'] for p in probes)
    from mavt4d.object_state.v5.training import draw_tube_masks
    for probe in probes:
        probe['rgb']=torch.zeros(18,1,3,32,32)
    masks,_,_=draw_tube_masks(probes,step=4000,seed=17,device=torch.device('cpu'))
    assert set(masks)=={0,1,2,9,10,11}


def test_smoke_loads_exact_model_config_and_rejects_tiny_override(tmp_path):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('smoke',Path(__file__).resolve().parents[2]/'scripts/mavt4d/smoke_v5.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'config.json';path.write_text(json.dumps({'model':{'slots':32,'content_tokens':4,'dynamic_tokens':4}}))
    cfg=module.resolve_model_config(path,tiny=False)
    assert (cfg.slots,cfg.content_tokens,cfg.dynamic_tokens)==(32,4,4)
    with pytest.raises(ValueError,match='tiny'):
        module.resolve_model_config(path,tiny=True)
