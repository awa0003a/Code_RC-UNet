from __future__ import annotations

import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import GeometryRCUNet
from dataset_preprocess import (MEAN, STD, OpticDiscDataset, find_pairs, get_patient_mapping,
                                make_grouped_splits, load_verified_manifest, ensure_no_subject_leak,
                                save_manifest)
from utils import (seed_everything, geometry_audit, segmentation_loss, evaluate, tune_threshold,
                   predict_prob, save_per_image_results)

# ========================= CONFIGURATION =========================
DATA_DIR = '/mnt/proj3/dd-24-125/awa0003/lungs_dataset/final'
OUTPUT_DIR = '/mnt/proj3/dd-24-125/awa0003/lungs_dataset/experiment_retina_v2'
PATIENT_CSV = None
GROUP_FROM_FILENAMES = True
EXISTING_SPLIT_CSV = None


MODE = 'radial'
IMAGE_SIZE = 384
BASE_FEATURES = 64
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.20
EARLY_STOP_PATIENCE = 20
SCHEDULER_PATIENCE = 7
VAL_FRAC = 0.15
TEST_FRAC = 0.15
NUM_WORKERS = 2
SEED = 42
AUGMENT_TRAIN = True
ALLOW_HORIZONTAL_FLIP = True
TTA_HORIZONTAL_FLIP = True
TUNE_THRESHOLD_ON_VAL = True
THRESHOLD_GRID = (0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70)
DEFAULT_THRESHOLD = 0.50
USE_AMP = True
GRAD_CLIP_NORM = 5.0
SAVE_WORST_N = 8


def train_one_model(model, train_loader, val_loader, device, out):
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=SCHEDULER_PATIENCE,min_lr=1e-6)
    use_amp=USE_AMP and device.type=='cuda'; scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    best_dice,best_epoch,no_improve=-1,0,0; hist=[]; ckpt_path=out/'best_model.pt'
    for epoch in range(1,EPOCHS+1):
        start=time.time(); model.train(); epoch_loss=0.0
        for images,masks in train_loader:
            images=images.to(device,non_blocking=True); masks=masks.to(device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            ctx=torch.autocast(device_type='cuda',dtype=torch.float16) if use_amp else nullcontext()
            with ctx: logits=model(images); loss=segmentation_loss(logits,masks)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP_NORM); scaler.step(optimizer); scaler.update(); epoch_loss+=loss.item()*images.size(0)
        train_loss=epoch_loss/len(train_loader.dataset); val,_=evaluate(model,val_loader,device,DEFAULT_THRESHOLD,False); scheduler.step(val['loss']); curr_lr=optimizer.param_groups[0]['lr']
        hist.append(dict(epoch=epoch,train_loss=train_loss,val_loss=val['loss'],val_dice=val['dice'],val_iou=val['iou'],lr=curr_lr,seconds=round(time.time()-start,2)))
        print(f'[{MODE}] {epoch:03d}/{EPOCHS}: train={train_loss:.4f}, val={val["loss"]:.4f}, Dice={val["dice"]:.4f}, lr={curr_lr:.2g}, {hist[-1]["seconds"]:.1f}s',flush=True)
        if val['dice']>best_dice+1e-7:
            best_dice,best_epoch,no_improve=val['dice'],epoch,0
            torch.save(dict(state=model.state_dict(),mode=MODE,epoch=epoch,best_val_dice=best_dice,image_size=IMAGE_SIZE,base=BASE_FEATURES,dropout=DROPOUT),ckpt_path)
        else:
            no_improve+=1
            if no_improve>=EARLY_STOP_PATIENCE:
                print(f'Early stopping at epoch {epoch}; best was {best_epoch}.'); break
    with open(out/'training_history.csv','w',newline='',encoding='utf-8') as fh:
        wr=csv.DictWriter(fh,fieldnames=list(hist[0])); wr.writeheader(); wr.writerows(hist)
    payload=torch.load(ckpt_path,map_location=device,weights_only=True); model.load_state_dict(payload['state'])
    return model,hist,best_epoch,best_dice


def save_plots(model,hist,test_pairs,device,threshold,out,mode,worst_rows):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    epochs=[r['epoch'] for r in hist]; fig,axs=plt.subplots(1,2,figsize=(12,4)); axs[0].plot(epochs,[r['train_loss'] for r in hist],label='Train'); axs[0].plot(epochs,[r['val_loss'] for r in hist],label='Validation'); axs[0].set(xlabel='Epoch',ylabel='BCE + Dice + Tversky',title='Training dynamics'); axs[0].legend(); axs[1].plot(epochs,[r['val_dice'] for r in hist],label='Validation Dice'); axs[1].set(xlabel='Epoch',ylabel='Mean per-image Dice',title='Checkpoint selection'); axs[1].legend(); fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=170); plt.close(fig)
    n=min(SAVE_WORST_N,len(worst_rows))
    if n>0:
        by_name={Path(ip).name:(ip,mp) for ip,mp in test_pairs}; fig,axes=plt.subplots(n,4,figsize=(14,max(3,3*n)),squeeze=False); model.eval()
        for ri,row in enumerate(worst_rows[:n]):
            ip,mp=by_name[row['image']]; ds=OpticDiscDataset([(ip,mp)],IMAGE_SIZE,False,ALLOW_HORIZONTAL_FLIP); x,y=ds[0]
            with torch.no_grad(): prob=predict_prob(model,x[None].to(device),TTA_HORIZONTAL_FLIP)[0,0].cpu().numpy()
            raw=(x*torch.tensor(STD).view(3,1,1)+torch.tensor(MEAN).view(3,1,1)).clamp(0,1).permute(1,2,0).numpy(); gt=y[0].numpy(); pred=(prob>=threshold).astype(np.float32)
            for col,(title,data,cmap) in enumerate([('Input',raw,None),('Ground truth',gt,'gray'),('Probability',prob,'viridis'),('Prediction',pred,'gray')]):
                axes[ri,col].imshow(data,cmap=cmap,vmin=0 if cmap else None,vmax=1 if cmap else None); axes[ri,col].set_title(title+(f' | Dice {row["dice"]:.3f}' if col==0 else ''),fontsize=9); axes[ri,col].axis('off')
        fig.suptitle('Worst test examples — DIAGNOSTIC ONLY'); fig.tight_layout(); fig.savefig(out/'worst_test_examples.png',dpi=150); plt.close(fig)
    kernels=[]
    # Swapped mode names here too.
    if mode in ('radial','circular'):
        for label,block in (('Encoder 3',model.enc3),('Bottleneck',model.bottleneck_conv)):
            for branch in block.branches: kernels.append((f'{label}: {branch.kernel_size}×{branch.kernel_size}',branch.effective_weight()[0,0].detach().float().cpu().numpy()))
    if mode in ('combined','circular'):
        z=model.bottleneck_attn.circular_spatial.effective_weight().detach().float().cpu().numpy(); kernels.extend([('Circular attention: mean',z[0,0]),('Circular attention: max',z[0,1])])
    if kernels:
        fig,axes=plt.subplots(1,len(kernels),figsize=(3*len(kernels),3.3),squeeze=False)
        for ax,(title,w) in zip(axes[0],kernels):
            limit=max(float(abs(w).max()),1e-6); im=ax.imshow(w,cmap='coolwarm',vmin=-limit,vmax=limit); ax.set_title(title,fontsize=9); fig.colorbar(im,ax=ax,shrink=.75)
        fig.tight_layout(); fig.savefig(out/'learned_geometric_kernels.png',dpi=160); plt.close(fig)


def main():
    if DATA_DIR.startswith('/content/drive/') and not Path('/content/drive/MyDrive').exists():
        from google.colab import drive; drive.mount('/content/drive')
    if MODE not in ('radial','combined','circular'): raise ValueError('Invalid MODE.')
    if IMAGE_SIZE<64 or IMAGE_SIZE%8: raise ValueError('IMAGE_SIZE must be >=64 and divisible by eight.')
    seed_everything(SEED); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); pairs=find_pairs(DATA_DIR); patient_map,group_source=get_patient_mapping(pairs,PATIENT_CSV,GROUP_FROM_FILENAMES)
    splits=load_verified_manifest(EXISTING_SPLIT_CSV,pairs,patient_map) if EXISTING_SPLIT_CSV else make_grouped_splits(pairs,patient_map,SEED,VAL_FRAC,TEST_FRAC)
    ids_per_split=ensure_no_subject_leak(splits,patient_map); out=Path(OUTPUT_DIR).expanduser()/MODE/time.strftime('%Y%m%d_%H%M%S'); out.mkdir(parents=True,exist_ok=True)
    settings={k:v for k,v in globals().copy().items() if k in {'DATA_DIR','OUTPUT_DIR','PATIENT_CSV','GROUP_FROM_FILENAMES','EXISTING_SPLIT_CSV','MODE','IMAGE_SIZE','BASE_FEATURES','BATCH_SIZE','EPOCHS','LEARNING_RATE','WEIGHT_DECAY','DROPOUT','EARLY_STOP_PATIENCE','SCHEDULER_PATIENCE','VAL_FRAC','TEST_FRAC','NUM_WORKERS','SEED','AUGMENT_TRAIN','ALLOW_HORIZONTAL_FLIP','TTA_HORIZONTAL_FLIP','TUNE_THRESHOLD_ON_VAL','THRESHOLD_GRID','DEFAULT_THRESHOLD','USE_AMP','GRAD_CLIP_NORM'}}; settings.update(device=str(device),torch_version=torch.__version__,group_source=group_source,patient_groups=ids_per_split)
    with open(out/'config.json','w') as fh: json.dump(settings,fh,indent=2)
    save_manifest(out/'split_manifest.csv',splits,patient_map); print(f'Found {len(pairs)} image/mask pairs. Subject IDs from {group_source}.'); print('Split image counts:',*(len(s) for s in splits)); print('Split subject counts:',ids_per_split,'| Overlap: ZERO')
    loaders=[]
    for idx,split in enumerate(splits):
        dataset=OpticDiscDataset(split,IMAGE_SIZE,augment=(idx==0 and AUGMENT_TRAIN),allow_horizontal_flip=ALLOW_HORIZONTAL_FLIP); loaders.append(DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=(idx==0),num_workers=NUM_WORKERS,pin_memory=(device.type=='cuda'),drop_last=False))
    train_loader,val_loader,test_loader=loaders; model=GeometryRCUNet(MODE,BASE_FEATURES,DROPOUT).to(device); num_params=sum(p.numel() for p in model.parameters() if p.requires_grad); before=geometry_audit(model); print(f'Mode={MODE} | Device={device} | Parameters={num_params:,} | Geometric kernels audited={len(before)}',flush=True)
    start=time.time(); model,hist,best_epoch,best_val_dice=train_one_model(model,train_loader,val_loader,device,out)
    threshold,threshold_scores=tune_threshold(model,val_loader,device,THRESHOLD_GRID,TTA_HORIZONTAL_FLIP) if TUNE_THRESHOLD_ON_VAL else (DEFAULT_THRESHOLD,{})
    val_final,_=evaluate(model,val_loader,device,threshold,TTA_HORIZONTAL_FLIP); test_result,_=evaluate(model,test_loader,device,threshold,TTA_HORIZONTAL_FLIP); rows=save_per_image_results(model,splits[2],test_loader,device,threshold,out,TTA_HORIZONTAL_FLIP)
    finite_hd=[r['HD95_resized_pixels'] for r in rows if np.isfinite(r['HD95_resized_pixels'])]; finite_assd=[r['ASSD_resized_pixels'] for r in rows if np.isfinite(r['ASSD_resized_pixels'])]
    test_result.update({'mode':MODE,'params':num_params,'best_epoch':best_epoch,'best_val_dice_at_0_5_no_tta':best_val_dice,'val_metrics_selected_threshold':val_final,'validation_threshold_scores':threshold_scores,'patient_group_source':group_source,'patient_group_counts':ids_per_split,'geometry_checks':geometry_audit(model),'mean_HD95_resized_pixels':float(np.mean(finite_hd)) if finite_hd else None,'mean_ASSD_resized_pixels':float(np.mean(finite_assd)) if finite_assd else None,'boundary_evaluable_images':len(finite_hd),'elapsed_seconds':round(time.time()-start,1),'interpretation':'Single seed. No performance improvement/significance claim without matched repeats.'})
    with open(out/'test_metrics.json','w') as fh: json.dump(test_result,fh,indent=2)
    rows=sorted(rows,key=lambda r:r['dice']); save_plots(model,hist,splits[2],device,threshold,out,MODE,rows); print('----- COMPLETED ONE MODEL RUN -----'); print(f'Best epoch: {best_epoch} | threshold chosen on validation: {threshold:.2f}'); print(f'TEST: Dice={test_result["dice"]:.4f}, IoU={test_result["iou"]:.4f}, precision={test_result["precision"]:.4f}, recall={test_result["recall"]:.4f}, failed (<0.5 Dice)={test_result["n_below_0_5_dice"]}'); print('Outputs:',out)
    return test_result


if __name__ == '__main__':
    main()
