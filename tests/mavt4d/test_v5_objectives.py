"""Objective contracts: no unavailable metadata or target may become an input."""
import copy
import unittest
import torch
from mavt4d.object_state.v5.training import compute_loss, observe_prefix
from mavt4d.object_state.v5.objectives import per_example_mean, reliability_objective
from mavt4d.object_state.v5.history import history_labels
from mavt4d.object_state.v5.config import V5Config
from mavt4d.object_state.v5.model import V5Model


def tiny_model():
    return V5Model(V5Config(slots=2, content_tokens=2, dynamic_tokens=2,
        width=16,evidence_width=24,spatial_blocks=1,spatial_heads=3,
        resampler_heads=4,samples_per_ray=3,query_chunk_size=4,window_size=2,relay_tokens=2))


def clip(camera=False, depth=False):
    result={'rgb':torch.rand(10,1,3,16,16),'rgb_valid':torch.ones(10,1,1,16,16,dtype=torch.bool),
            'times':torch.arange(10).double()*.5,'source':'arkitscenes',
            'plan':{'prefix':[1,3],'observed':[True,True],'future':[4,5,6,7,8,9]},'time_valid':True}
    if camera:
        result.update(intrinsics=torch.tensor([[16.,0.,8.],[0.,16.,8.],[0.,0.,1.]]).expand(10,1,3,3).clone(),
                      camera_c2w=torch.eye(4).expand(10,1,4,4).clone(),camera_valid=torch.ones(10,1,dtype=torch.bool),
                      metric_valid=torch.ones(10,1,dtype=torch.bool))
    if depth:
        result.update(depth=torch.ones(10,1,1,16,16), input_depth=torch.ones(10,1,1,16,16),
                      depth_valid=torch.ones(10,1,1,16,16,dtype=torch.bool),
                      input_depth_valid=torch.ones(10,1,1,16,16,dtype=torch.bool),depth_provenance='measured',scene_static=True)
    return result


class ObjectiveContracts(unittest.TestCase):
    def test_camera_only_and_all_six_future_endpoints(self):
        model=tiny_model(); c=clip(camera=True)
        records=[]
        loss,metrics=compute_loss(model,[c],step=4000,rays=2,seed=17,bridge=False,
            readout_callback=lambda **kw: records.append((kw['kind'],kw['frame'])))
        self.assertEqual([f for k,f in records if k=='future'],c['plan']['future'])
        loss.backward(); self.assertTrue(torch.isfinite(loss))

    def test_future_target_perturbation_cannot_change_prefix(self):
        model=tiny_model().eval(); c=clip(camera=True,depth=True); other=copy.deepcopy(c)
        other['rgb'][4:]=0; other['depth'].fill_(99); other['camera_c2w'][4:,:,:,3]=99
        a,_,_=observe_prefix(model,[c],seed=3); b,_,_=observe_prefix(model,[other],seed=3)
        for key,value in a.tensor_dict().items():
            torch.testing.assert_close(value,b.tensor_dict()[key])

    def test_unknown_fps_disables_forecast(self):
        model=tiny_model(); c=clip(); c['time_valid']=False
        _,metrics=compute_loss(model,[c],step=4000,rays=2,bridge=False)
        self.assertEqual(metrics['future_valid'],0)

    def test_normalization_is_partition_invariant(self):
        x=torch.tensor([[1.,4.,3.],[2.,9.,8.]],requires_grad=True)
        v=torch.tensor([[True,False,False],[True,True,False]])
        full=per_example_mean(x,v)
        split=(per_example_mean(x[:1],v[:1])+per_example_mean(x[1:],v[1:]))/2
        torch.testing.assert_close(full,split)

    def test_reliability_target_is_detached_and_missing_is_absent(self):
        prediction=torch.tensor([.7,.2],requires_grad=True); target=torch.tensor([.1,.9],requires_grad=True)
        loss=reliability_objective(prediction,target,torch.tensor([True,False]))
        loss.backward(); self.assertIsNone(target.grad); self.assertEqual(prediction.grad[1],0)

    def test_static_surface_history_and_mask(self):
        c=clip(camera=True,depth=True)
        pixel=torch.tensor([[[8.5,8.5]]])
        history=[{'frame':1,'input_mask':torch.ones(1,1,1,16,16,dtype=torch.bool)}]
        labels=history_labels([c],3,pixel,history,current_index=None,view=0)
        self.assertEqual(labels.item(),1)
        c['scene_static']=False
        self.assertEqual(history_labels([c],3,pixel,history,current_index=None,view=0).item(),-1)

if __name__=='__main__': unittest.main()

class BridgeContracts(unittest.TestCase):
    def test_dropped_branch_starts_from_empty_independent_state(self):
        model=tiny_model(); c=clip(camera=True)
        inputs=[]; original=model.observe
        def recording(state,observation,**kwargs):
            inputs.append((observation.intrinsics is None,state.c.detach().clone()))
            return original(state,observation,**kwargs)
        model.observe=recording
        loss,metrics=compute_loss(model,[c],step=4000,rays=1,seed=4,bridge=True)
        first_blind=next(state for blind,state in inputs if blind)
        self.assertEqual(first_blind.abs().sum().item(),0.)
        self.assertEqual(metrics['bridge_active'],1.)
        loss.backward(); self.assertTrue(torch.isfinite(loss))

class AdditionalObjectiveContracts(unittest.TestCase):
    def test_heldout_depth_does_not_hide_observed_rgb_status(self):
        c=clip(camera=True,depth=True); c['scene_static']=False
        c['input_depth_valid'].zero_()
        history=[{'frame':3,'input_mask':torch.ones(1,1,1,16,16,dtype=torch.bool)}]
        labels=history_labels([c],3,torch.tensor([[[8.5,8.5]]]),history,current_index=0,view=0)
        self.assertEqual(labels.item(),0)
        history[0]['input_mask'].zero_()
        labels=history_labels([c],3,torch.tensor([[[8.5,8.5]]]),history,current_index=0,view=0)
        self.assertEqual(labels.item(),2)

    def test_full_loss_microbatch_partition_invariance(self):
        model=tiny_model().eval(); a=clip(); b=clip(); a['group_id']='a'; b['group_id']='b'
        with torch.no_grad():
            full,_=compute_loss(model,[a,b],step=4000,rays=2,seed=17,bridge=False)
            left,_=compute_loss(model,[a],step=4000,rays=2,seed=17,bridge=False)
            right,_=compute_loss(model,[b],step=4000,rays=2,seed=17,bridge=False)
        torch.testing.assert_close(full,(left+right)/2,rtol=2e-5,atol=2e-6)

class CapabilityBucketing(unittest.TestCase):
    def test_mixed_available_metadata_cannot_silently_drop_supervision(self):
        from mavt4d.object_state.v5.training import clip_signature
        rgb=clip(); camera=clip(camera=True); depth=clip(camera=True,depth=True)
        self.assertNotEqual(clip_signature(rgb),clip_signature(camera))
        self.assertNotEqual(clip_signature(camera),clip_signature(depth))
        measured=copy.deepcopy(depth); reconstructed=copy.deepcopy(depth)
        reconstructed['depth_provenance']='reconstructed'
        self.assertNotEqual(clip_signature(measured),clip_signature(reconstructed))
        with self.assertRaisesRegex(ValueError,'microbatch'):
            compute_loss(tiny_model(),[camera,depth],step=4000,rays=1,bridge=False)

class SecondPassRegression(unittest.TestCase):
    def test_indoor_metric_and_relative_ray_bounds_follow_domain(self):
        from mavt4d.object_state.v5.training import _read_targets
        model=tiny_model(); a=clip(camera=True); b=copy.deepcopy(a)
        b['camera_valid'].zero_(); b['metric_valid'].zero_()
        state,_,_=observe_prefix(model,[a,b])
        _,_,query=_read_targets(model,state,[a,b],3,rays=2,seed=3)
        far=torch.as_tensor(query.far).expand(2,2)
        torch.testing.assert_close(far,torch.tensor([[20.,20.],[20.,20.]]))

    def test_unknown_fps_never_drives_physical_dynamics(self):
        model=tiny_model(); c=clip(); c['time_valid']=False
        dts=[]; original=model.predict
        def recording(state,dt,*args,**kwargs):
            dts.append(dt.detach().clone()); return original(state,dt,*args,**kwargs)
        model.predict=recording
        observe_prefix(model,[c])
        self.assertTrue(dts)
        self.assertTrue(all(torch.equal(dt,torch.zeros_like(dt)) for dt in dts))

    def test_empty_endpoint_cannot_shift_bridge_correspondence(self):
        from unittest.mock import patch
        from mavt4d.object_state.v5.training import _bridge_state
        model=tiny_model(); c=clip(camera=True,depth=True)
        c['input_depth'][1].fill_(2); c['input_depth'][3].fill_(7)
        branch=model.initial_state(1,time=1.5)
        diag={'proposal_depth':torch.ones(1,1,2),'depth_prob':torch.full((1,1,2),.5)}
        history=[{'frame':frame,'time':torch.tensor([frame*.5]),
                  'camera':(c['intrinsics'][frame],c['camera_c2w'][frame]),
                  'input_mask':torch.full((1,1,1,16,16),frame==3,dtype=torch.bool),
                  'diagnostics':{} if frame==1 else diag} for frame in (1,3)]
        calls=[]
        def capture(state,scene,metric,valid,times):
            calls.append((metric,valid,times)); return state
        with patch('mavt4d.object_state.v5.gauge.calibrate_state',side_effect=capture):
            _bridge_state(model,branch,[c],history,[{},diag])
        self.assertTrue(calls[0][1].all())
        torch.testing.assert_close(calls[0][0][...,2],torch.full((1,1),7.))
        torch.testing.assert_close(calls[0][2],torch.full((1,1),1.5,dtype=calls[0][2].dtype))

    def test_tubelet_context_surface_counts_as_prior_evidence(self):
        model=tiny_model(); c=clip(camera=True,depth=True)
        c['plan']={'prefix':[1],'observed':[True],'future':[]}
        c['input_depth'][1].fill_(.5)
        _,history,_=observe_prefix(model,[c])
        labels=history_labels([c],2,torch.tensor([[[8.5,8.5]]]),history,current_index=None,view=0)
        self.assertEqual(labels.item(),1)

class ActualAdmissionRegression(unittest.TestCase):
    def test_bypassed_encoder_context_cannot_certify_inferred(self):
        model=tiny_model(); c=clip(camera=True,depth=True)
        c['plan']={'prefix':[1],'observed':[True],'future':[]}
        c['rgb_valid'][1].zero_()
        _,history,_=observe_prefix(model,[c])
        labels=history_labels([c],2,torch.tensor([[[8.5,8.5]]]),history,current_index=None,view=0)
        self.assertNotEqual(labels.item(),1)
        self.assertFalse(any(record['input_mask'].any() for record in history))

class PatchAdmissionRegression(unittest.TestCase):
    def test_discarded_endpoint_patch_context_cannot_certify_inferred(self):
        model=tiny_model(); c=clip(camera=True,depth=True)
        c['rgb']=torch.rand(10,1,3,32,32)
        c['rgb_valid']=torch.ones(10,1,1,32,32,dtype=torch.bool)
        for key in ('depth','input_depth'):
            c[key]=torch.ones(10,1,1,32,32)
        for key in ('depth_valid','input_depth_valid'):
            c[key]=torch.ones(10,1,1,32,32,dtype=torch.bool)
        c['plan']={'prefix':[1],'observed':[True],'future':[]}
        c['rgb_valid'][1,:,:,:,:16]=False
        _,history,_=observe_prefix(model,[c])
        labels=history_labels([c],2,torch.tensor([[[8.5,8.5]]]),history,current_index=None,view=0)
        self.assertNotEqual(labels.item(),1)
        self.assertFalse(history[0]['input_mask'][...,:16].any())
        self.assertTrue(history[0]['input_mask'][...,16:].all())
