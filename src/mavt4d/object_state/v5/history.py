"""Supervision-only material-surface history. Never consumed by the encoder.

History retains only actually admitted prefix observations. Target depth may
locate a query surface, but cannot alter the filtering state or proposals.
"""
import torch
from .evidence import classify_visibility, visibility_from_depth, transport_rigid_points


@torch.no_grad()
def history_labels(clips,frame,pixels,history,*,current_index=None,view=0):
    device=pixels.device; b,q,_=pixels.shape
    result=torch.full((b,q),-1,device=device,dtype=torch.long)
    if not history:
        return result
    for batch,c in enumerate(clips):
        required=('depth','depth_valid','intrinsics','camera_c2w','input_depth')
        if any(key not in c for key in required):
            continue
        # Pseudo targets and missing provenance never certify observed surfaces.
        if c.get('depth_provenance') not in ('measured','reconstructed','sensor','lidar','metric_sensor'):
            continue
        if 'camera_valid' not in c or not bool(c['camera_valid'][frame,view]):
            continue
        h,w=c['rgb'].shape[-2:]
        x=pixels[batch,:,0].floor().long().clamp(0,w-1)
        y=pixels[batch,:,1].floor().long().clamp(0,h-1)
        target_z=c['depth'][frame,view,0].to(device)[y,x]
        target_valid=c['depth_valid'][frame,view,0].to(device)[y,x] & torch.isfinite(target_z) & (target_z>0)
        k=c['intrinsics'][frame,view].to(device)
        pose=c['camera_c2w'][frame,view].to(device)
        homogeneous=torch.cat((pixels[batch],torch.ones_like(target_z[:,None])),-1)
        camera=(torch.linalg.inv(k)@homogeneous.T).T*target_z[:,None]
        points=(pose[:3,:3]@camera.T).T+pose[:3,3]
        frames=[record['frame'] for record in history]
        transported=points[None,None].expand(1,len(frames),q,3)
        correspondence=target_valid[None,None].expand(1,len(frames),q).clone()
        if not c.get('scene_static',False):
            # Trusted rigid material mapping is opt-in, never derived from boxes
            # or model poses. Per-query identity comes from an exact instance map.
            if not all(key in c for key in ('instance_map','object_to_world','object_pose_valid')) or not c.get('rigid_correspondence_trusted',False):
                # Same-timestamp cross-view evidence needs no motion model.
                correspondence &= torch.tensor([f==frame for f in frames],device=device)[None,:,None]
                ids=torch.full((q,),-1,device=device,dtype=torch.long)
            else:
                ids=c['instance_map'][frame,view].to(device)[y,x]
                correspondence.zero_()
            transported=transported.clone()
            for identity in ids.unique().tolist():
                if identity<0 or identity>=c['object_to_world'].shape[1]:
                    continue
                selected=ids==identity
                poses=c['object_to_world'][:,identity].to(device)
                moved=transport_rigid_points(points[None,selected],poses[frame:frame+1],poses[frames][None])
                transported[:,:,selected]=moved
                trusted=c['object_pose_valid'][frames,identity].to(device)&c['object_pose_valid'][frame,identity].to(device)
                correspondence[:,:,selected]=target_valid[selected][None,None]&trusted[None,:,None]
        cameras=c['camera_c2w'][frames,0].to(device)[None]
        intrinsics=c['intrinsics'][frames,0].to(device)[None]
        camera_available=c['camera_valid'][frames,0].to(device)[None]
        cameras=torch.where(camera_available[...,None,None],cameras,torch.eye(4,device=device,dtype=cameras.dtype))
        intrinsics=torch.where(camera_available[...,None,None],intrinsics,torch.eye(3,device=device,dtype=intrinsics.dtype))
        depth=c['input_depth'][frames,0,0].to(device)[None]
        if 'input_depth_valid' in c:
            depth_valid=c['input_depth_valid'][frames,0,0].to(device)[None]
        else:
            depth_valid=torch.zeros_like(depth,dtype=torch.bool)
        pixel_valid=torch.stack([r['input_mask'][batch,0,0].to(device) for r in history])[None]
        correspondence &= c['camera_valid'][frames,0].to(device)[None,:,None]
        visible,checked=visibility_from_depth(transported,torch.linalg.inv(cameras),intrinsics,
            depth,depth_valid,pixel_valid,correspondence)
        if view==0 and current_index is not None and frames[current_index]==frame:
            # The target surface was measured at this exact input-camera pixel.
            # Withholding its depth from the encoder does not withhold its RGB.
            # This GT-only label must not be fed back to filtering/proposals.
            allowed=pixel_valid[0,current_index,y,x]
            visible[0,current_index]=target_valid & allowed
            checked[0,current_index]=target_valid
        result[batch]=classify_visibility(visible.transpose(1,2),checked.transpose(1,2),current_index)[0]
    return result
