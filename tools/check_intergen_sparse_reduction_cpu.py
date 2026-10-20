"""CPU numerical regression for sparse spatial and per-person contact reductions."""
from pathlib import Path
import hashlib,json,os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from tools.intergen_rotation_objective_sparse_v2 import selected_pair_mean,selected_contact_person_sum,rotation_objective
from tools.intergen_rotation_grouped_objective import rotation_objective as original
from dataset.interhuman_root_rotation import fk_positions

def main():
 torch.set_num_threads(1);torch.manual_seed(20261017)
 source=ROOT/'third_party/intergen_base_20261008';os.chdir(source);sys.path.insert(0,str(source))
 from models.losses import InterLoss,GeometricLoss
 pair=InterLoss('l2',22);contact_manager=GeometricLoss('l2',22,'audit')
 values=torch.ones(1,1,22,22,requires_grad=True);selected=torch.zeros_like(values,dtype=torch.bool);selected[...,0,0]=True
 corrected=selected_pair_mean(values,selected);old=(values*selected).flatten(2).mean(-1)
 reference=pair.mix_masked_mse(torch.ones(1,1,1,484),torch.zeros(1,1,1,484),torch.ones(1,1,1,1),torch.ones(1),dm_mask=selected.reshape(1,1,1,484))
 assert torch.allclose(corrected,reference,atol=1e-6)
 assert torch.allclose(corrected,torch.ones_like(corrected))
 corrected.sum().backward();assert torch.equal(values.grad,selected.float())
 dense=selected_pair_mean(values.detach(),torch.ones_like(selected));assert torch.equal(dense,torch.ones_like(dense))
 empty=selected_pair_mean(values.detach(),torch.zeros_like(selected));assert torch.equal(empty,torch.zeros_like(empty))
 feet=torch.ones(1,1,2,4,3,requires_grad=True);active=torch.zeros(1,1,2,4);active[...,0]=1
 new_contact=selected_contact_person_sum(feet.square(),active)
 ref_contact=sum(contact_manager.mix_masked_mse(feet[:,:,i],torch.zeros_like(feet[:,:,i]),torch.ones(1,1,1),torch.ones(1),contact_mask=active[:,:,i]) for i in (0,1))
 assert torch.allclose(new_contact,ref_contact,atol=1e-6)
 old_contact=(feet.square()*active[...,None]).flatten(2).mean(-1)
 new_contact.sum().backward();assert torch.isfinite(feet.grad).all()
 assert selected_contact_person_sum(feet.detach().square(),torch.zeros_like(active)).item()==0
 # End-to-end loss/gradient/padding checks with nondegenerate rotations and a fixed skeleton.
 b,t=4,6;target=torch.zeros(b,t,2,268);identity=torch.tensor([1.,0,0,1,0,0])
 target[...,132:258]=identity.repeat(21);target[...,262:]=identity
 target[:,:,1,2]=.5;target[...,258]=1
 offsets=torch.zeros(b,2,22,3);offsets[:,:,1:,1]=.1
 target[...,:66]=fk_positions(target,offsets).flatten(-2)
 pred=(target+.03*torch.randn_like(target)).requires_grad_();lengths=torch.tensor([6,5,4,3]);timesteps=torch.full((b,),10)
 args=(offsets,lengths,timesteps,torch.zeros(268),torch.ones(268))
 before=original(pred,target,*args);after=rotation_objective(pred,target,*args)
 changed={'fk_distance','fk_affinity','fk_contact','total'}
 assert all(torch.equal(before[k],after[k]) for k in before.keys()-changed)
 after['total'].backward();assert torch.isfinite(pred.grad).all()
 pred2=torch.cat([pred.detach(),torch.randn_like(pred)],1).requires_grad_();target2=torch.cat([target,torch.randn_like(target)],1)
 padded=rotation_objective(pred2,target2,*args)
 assert all(torch.allclose(after[k],padded[k],atol=1e-6) for k in after)
 padded['total'].backward();valid=torch.arange(t*2)[None]<lengths[:,None]
 assert pred2.grad[~valid].abs().max().item()==0
 assert not torch.cuda.is_initialized()
 result={'schema':'intergen_sparse_reduction_cpu_regression_v1','status':'passed','cuda_initialized':False,
 'sparse_pair_one_of484':{'old_weighted3':float(3*old),'corrected_weighted3':float(3*corrected),'official_same_mask_weighted3':float(3*reference),'dilution_factor':484},
 'contact_one_selected_per_person':{'old_weighted30':float(30*old_contact),'corrected_weighted30_person_sum':float(30*new_contact),'official_weighted30_person_sum':float(30*ref_contact)},
 'checks':['selected-pair denominator matches official on identical mask','sparse gradient zero outside selected pairs','dense/empty masks','contact per-person sum matches official on identical contacts','unaffected loss components identical','full-objective finite gradients','padding invariance','zero padded gradients'],
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'corrected_objective_sha256':hashlib.sha256((ROOT/'tools/intergen_rotation_objective_sparse_v2.py').read_bytes()).hexdigest(),
 'limits':'Denominator correction only; original GT-vs-pred distance/contact masks and grouped valid-frame reduction remain custom; no claim this alone explains FID gap'}
 out=ROOT/'output/hhc_sota_20261016/intergen_sparse_reduction_cpu_regression_20261017.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
