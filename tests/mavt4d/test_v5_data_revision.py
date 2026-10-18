"""Data recipe regression contracts fixed before the revised full run."""
import copy
import torch
from mavt4d.object_state.v5.natural import static_view_plan, normalize_clip_gauge
from mavt4d.object_state.v5.training import ray_bounds, _observation, clip_signature


def test_static_views_keep_a_target_and_revisit_without_time():
    for profile, count in [('S',2),('L',3),('G',3)]:
        p=static_view_plan(12,profile,seed=19)
        assert sum(p['observed'])==count
        assert len(p['heldout'])==1
        assert not set(p['prefix']) & set(p['heldout'])
        assert p['future']==[] and p['temporal_supervision'] is False
        if profile=='G':
            assert 1<=p['observed'].count(False)<=3
            assert p['observed'][0] and p['observed'][-1]


def test_static_view_observation_is_one_image():
    c={'rgb':torch.rand(8,1,3,8,8),'rgb_valid':torch.ones(8,1,1,8,8,dtype=torch.bool),
       'times':torch.arange(8).double(),'time_valid':False}
    o=_observation([c],3,'cpu')
    assert o.images.shape[2]==1
    torch.testing.assert_close(o.images[0,0,0],c['rgb'][3,0]*2-1)


def test_gauge_uses_first_admitted_camera_and_transforms_heldout():
    poses=torch.eye(4).expand(5,1,4,4).clone(); poses[:,0,0,3]=torch.arange(5.)+10
    c={'rgb':torch.rand(5,1,3,8,8),'camera_c2w':poses,'camera_valid':torch.ones(5,1,dtype=torch.bool),
       'time_valid':False,'plan':{'prefix':[1,2,3],'observed':[True,False,True],'heldout':[4]}}
    normalize_clip_gauge(c)
    torch.testing.assert_close(c['camera_c2w'][1,0],torch.eye(4))
    assert c['camera_c2w'][4,0,0,3]==3
    torch.testing.assert_close(c['scene_to_external']@c['camera_c2w'][4,0],poses[4,0])


def test_ray_bounds_are_domain_and_gauge_dependent():
    clips=[{'source':'nuscenes'},{'source':'arkitscenes'},{'source':'dl3dv'}]
    near,far=ray_bounds(clips,torch.tensor([True,True,False]))
    torch.testing.assert_close(near,torch.tensor([[.1],[.05],[.05]]))
    torch.testing.assert_close(far,torch.tensor([[120.],[20.],[20.]]))
    _,far=ray_bounds(clips,torch.tensor([False,False,False]))
    assert far.tolist()==[[20.],[20.],[20.]]


def test_static_evidence_requires_exact_source_protocol_and_sequence():
    from mavt4d.object_state.v5.adapters.decode import static_capture_evidence
    row={'source':'arkitscenes','sequence':'scan1','depth_provenance':'measured','scene_static':True}
    assert not static_capture_evidence(row)
    row['static_scene_evidence']={'protocol':'arkitscenes-static-capture-v1','reference':'arxiv:2111.08897',
       'scope':'sequence','sequence':'scan1','claim':'publisher_capture_assumption_sensor_checked'}
    assert static_capture_evidence(row)
    row['sequence']='scan2'
    assert not static_capture_evidence(row)


def test_per_draw_masks_survive_microbatch_partition():
    from mavt4d.object_state.v5.training import draw_tube_masks
    base={'rgb':torch.rand(8,1,3,16,16),'time_valid':True,
          'plan':{'prefix':[1,3,5],'observed':[True,True,True],'future':[]}}
    clips=[dict(base,draw_seed=n) for n in range(20)]
    full,counts,_=draw_tube_masks(clips,step=4000,seed=77,device='cpu')
    for j,c in enumerate(clips):
        one,n,_=draw_tube_masks([c],step=4000,seed=77,device='cpu')
        assert n.item()==counts[j].item()
        for frame,mask in full.items():
            expected=one.get(frame,torch.ones_like(mask[:1]))
            assert torch.equal(mask[j:j+1],expected)


def test_ineligible_profile_is_resampled_without_shortened_prefix():
    from mavt4d.object_state.v5.natural import sample_plan
    p=sample_plan(torch.arange(18).double()*.25,'G',seed=17,horizon=6)
    assert p['profile']=='S' and len(p['prefix'])==3 and len(p['future'])==6


def test_static_heldout_pixels_never_enter_filter_but_have_readout_loss():
    from test_v5_objectives import tiny_model, clip
    from mavt4d.object_state.v5.training import observe_prefix, compute_loss
    model=tiny_model().eval(); c=clip(camera=True); c['time_valid']=False
    c['plan']=static_view_plan(10,'G',seed=19)
    other=copy.deepcopy(c); other['rgb'][c['plan']['heldout']]=0
    a,_,_=observe_prefix(model,[c]); b,_,_=observe_prefix(model,[other])
    for key,value in a.tensor_dict().items():
        torch.testing.assert_close(value,b.tensor_dict()[key])
    records=[]
    _,metrics=compute_loss(model,[c],step=4000,rays=2,bridge=False,readout_callback=lambda **kw:records.append(kw))
    assert metrics['future_valid']==0 and metrics['heldout_views']==1
    assert [r['frame'] for r in records if r['kind']=='heldout']==c['plan']['heldout']
    assert all(float(r['state'].t)==0 for r in records)


def test_draw_masks_keep_total_loss_microbatch_invariant():
    from test_v5_objectives import tiny_model, clip
    from mavt4d.object_state.v5.training import compute_loss
    model=tiny_model().eval(); clips=[dict(clip(),draw_seed=n) for n in (2,3,4,5)]
    with torch.no_grad():
        full,_=compute_loss(model,clips,step=4000,rays=2,seed=77,bridge=False)
        parts=[compute_loss(model,[c],step=4000,rays=2,seed=77,bridge=False)[0] for c in clips]
    torch.testing.assert_close(full,torch.stack(parts).mean(),rtol=2e-5,atol=2e-6)
