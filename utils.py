from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import RadialDepthwiseConv2d, CircularRingSpatialFilter


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def geometry_audit(model):
    report={}
    for name,module in model.named_modules():
        if isinstance(module,(RadialDepthwiseConv2d,CircularRingSpatialFilter)):
            w=module.effective_weight().detach().float().cpu(); support=module.support.detach().float().cpu()
            outside=(w*(1-support)).abs().max().item(); rotate=(w-torch.rot90(w,1,(-2,-1))).abs().max().item(); reflect=(w-torch.flip(w,(-1,))).abs().max().item()
            if max(outside,rotate,reflect)>1e-5: raise AssertionError(f'Geometric constraint failed at {name}.')
            report[name]=dict(kernel_size=module.kernel_size,number_of_rings=len(module.basis),max_outside_disk=outside,max_rot90_deviation=rotate,max_reflection_deviation=reflect)
    return report


def segmentation_loss(logits, labels):
    prob=logits.sigmoid(); dimensions=(1,2,3)
    tp=(prob*labels).sum(dimensions); fp=(prob*(1-labels)).sum(dimensions); fn=((1-prob)*labels).sum(dimensions)
    dice=1-((2*tp+1)/(2*tp+fp+fn+1)).mean(); tversky=1-((tp+1)/(tp+0.35*fp+0.65*fn+1)).mean(); bce=F.binary_cross_entropy_with_logits(logits,labels)
    return .35*bce+.40*dice+.25*tversky


def score_tensors(prob,masks,threshold):
    pred=(prob>=threshold).flatten(1); true=(masks>=.5).flatten(1)
    tp=(pred&true).sum(1).float(); fp=(pred&~true).sum(1).float(); fn=(~pred&true).sum(1).float(); gt=true.sum(1).float(); pr=pred.sum(1).float()
    dice=torch.where((2*tp+fp+fn)>0,(2*tp)/(2*tp+fp+fn).clamp_min(1),torch.ones_like(tp)); iou=torch.where((tp+fp+fn)>0,tp/(tp+fp+fn).clamp_min(1),torch.ones_like(tp))
    prec=torch.where((tp+fp)>0,tp/(tp+fp).clamp_min(1),torch.where(gt>0,torch.zeros_like(tp),torch.ones_like(tp))); rec=torch.where((tp+fn)>0,tp/(tp+fn).clamp_min(1),torch.ones_like(tp))
    return {'dice':dice,'iou':iou,'precision':prec,'recall':rec,'tp':tp,'fp':fp,'fn':fn,'gt_pixels':gt,'pred_pixels':pr}


def predict_prob(model,images,tta=False):
    p=torch.sigmoid(model(images))
    if tta:
        mirrored=torch.flip(images,dims=(-1,)); p_flip=torch.sigmoid(model(mirrored)); p=.5*(p+torch.flip(p_flip,dims=(-1,)))
    return p


@torch.no_grad()
def evaluate(model,loader,device,threshold=.5,tta=False):
    model.eval(); totals=defaultdict(list); loss_total=0
    for images,labels in loader:
        images=images.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True); p=predict_prob(model,images,tta)
        logits=torch.logit(p.clamp(min=1e-6,max=1-1e-6)); loss_total+=segmentation_loss(logits,labels).item()*images.shape[0]
        scores=score_tensors(p,labels,threshold)
        for key,tensor in scores.items(): totals[key].extend(tensor.detach().cpu().tolist())
    n=len(loader.dataset); result={'loss':loss_total/n,'n_images':n,'threshold':threshold,'tta_horizontal_flip':bool(tta)}
    for key in ('dice','iou','precision','recall'):
        result[key]=float(np.mean(totals[key])); result[key+'_std']=float(np.std(totals[key],ddof=1)) if n>1 else 0.0
    tp,fp,fn=[sum(totals[k]) for k in ('tp','fp','fn')]; result['pooled_dice']=float(2*tp/max(1,2*tp+fp+fn)); result['pooled_iou']=float(tp/max(1,tp+fp+fn)); result['empty_predictions']=int(sum(a==0 for a in totals['pred_pixels'])); result['n_below_0_5_dice']=sum(x<.5 for x in totals['dice'])
    return result,totals


@torch.no_grad()
def tune_threshold(model,val_loader,device,threshold_grid,tta=False):
    model.eval(); score_sums={float(t):0.0 for t in threshold_grid}; n=0
    for images,labels in val_loader:
        images,labels=images.to(device),labels.to(device); probs=predict_prob(model,images,tta)
        for t in score_sums: score_sums[t]+=score_tensors(probs,labels,t)['dice'].sum().item()
        n+=images.shape[0]
    scores={f'{t:.2f}':round(v/n,6) for t,v in score_sums.items()}; best=max(score_sums,key=lambda t:(score_sums[t],-abs(t-.5)))
    return best,scores


def hd95_assd(pred,gt):
    try: from scipy.ndimage import binary_erosion,distance_transform_edt
    except ImportError: return float('nan'),float('nan')
    pred=pred.astype(bool); gt=gt.astype(bool)
    if not pred.any() or not gt.any(): return float('nan'),float('nan')
    ps=pred^binary_erosion(pred); gs=gt^binary_erosion(gt); samples=np.concatenate((distance_transform_edt(~gs)[ps],distance_transform_edt(~ps)[gs]))
    return float(np.percentile(samples,95)),float(samples.mean())


@torch.no_grad()
def save_per_image_results(model,pairs,loader,device,threshold,out,tta):
    model.eval(); rows=[]; img_no=0
    for images,labels in loader:
        images=images.to(device); prob=predict_prob(model,images,tta).cpu(); scores=score_tensors(prob,labels,threshold)
        for j in range(len(images)):
            ip=pairs[img_no][0]; pred=prob[j,0].numpy()>=threshold; mask=labels[j,0].numpy()>=.5; hd95,assd=hd95_assd(pred,mask)
            rows.append({'image':Path(ip).name,'dice':scores['dice'][j].item(),'iou':scores['iou'][j].item(),'precision':scores['precision'][j].item(),'recall':scores['recall'][j].item(),'gt_pixels':int(scores['gt_pixels'][j].item()),'pred_pixels':int(scores['pred_pixels'][j].item()),'HD95_resized_pixels':hd95,'ASSD_resized_pixels':assd}); img_no+=1
    assert img_no==len(pairs)
    with open(out/'per_image_test_metrics.csv','w',newline='',encoding='utf-8') as fh:
        wr=csv.DictWriter(fh,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    return rows
