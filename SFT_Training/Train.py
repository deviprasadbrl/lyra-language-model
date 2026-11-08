import math, torch
import torch.nn.functional as F
from accelerate import Accelerator
from Data_Pipeline import make_loaders
from Pre_Training.util import Decoder
from Pre_Training.Pre_Config import Config

# The code written down is specifically written for kaggle environement with kaggle paths. 
# So while using  this script change the paths as per your environment

cfg=Config()

BASE_CKPT="final_model.pt"
OUT_DIR="/kaggle/working" #change it as per your out_dir 
EPOCHS=3
MICRO=4
ACCUM=8
LR=5e-5
MIN_LR=5e-6
WARMUP=30
WEIGHT_DECAY=0.1
CLIP=1.0


def sft_loss(logits,labels,reduction="mean"):
   
    logits=logits[:,:-1].reshape(-1,logits.size(-1)).float()
    targets=labels[:,1:].reshape(-1)
    return F.cross_entropy(logits,targets,ignore_index=-100,reduction=reduction)


def get_lr(step, total):
    if step<WARMUP:
        return LR*(step+1)/WARMUP
    
    p=min(1.0,(step-WARMUP)/max(1,total-WARMUP))
    return MIN_LR+0.5*(LR-MIN_LR)*(1+math.cos(math.pi*p))


@torch.no_grad()
def evaluate(model,val_loader,accelerator):
    model.eval()
    total=0.0
    count=0

    for b in val_loader:
        ids=b["input_ids"].to(accelerator.device)
        labels=b["labels"].to(accelerator.device)
        with accelerator.autocast():
            logits=model(ids)

        total+=sft_loss(logits, labels, reduction="sum").item()
        count+=(labels[:, 1:]!=-100).sum().item()

    model.train()
    return total/count


def main():
    accelerator=Accelerator(mixed_precision="fp16",gradient_accumulation_steps=ACCUM)

    train_loader,val_loader=make_loaders(bs=MICRO)

    model=Decoder(
    vocab_size=cfg.vocab_size,
    d_model=cfg.d_model,
    num_heads=cfg.num_heads,
    num_kv=cfg.num_kv,
    num_blocks=cfg.num_blocks,
    intermediate_size=cfg.intermediate_size,
    max_length=cfg.max_length
)
    
    ckpt=torch.load(BASE_CKPT,map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    decay=[p for p in model.parameters() if p.ndim>=2]
    no_decay=[p for p in model.parameters() if p.ndim<2]

    optimizer=torch.optim.AdamW(
        [{"params":decay,"weight_decay":WEIGHT_DECAY},{"params":no_decay,"weight_decay": 0.0}],
        lr=LR,
        betas=(0.9,0.95)
        )

    model,optimizer,train_loader=accelerator.prepare(model,optimizer,train_loader)

    steps_per_epoch=math.ceil(len(train_loader)/ACCUM)
    total_steps=steps_per_epoch*EPOCHS

    accelerator.print(f"{steps_per_epoch} steps per epoch, {total_steps} total")

    best_val=float("inf")
    step=0

    model.train()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            for g in optimizer.param_groups:
                g["lr"]=get_lr(step,total_steps)

            with accelerator.accumulate(model):

                logits=model(batch["input_ids"])
                loss=sft_loss(logits,batch["labels"])

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(),CLIP)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                step+=1
                if step%10==0:
                    accelerator.print(f"epoch: {epoch+1} step: {step}/{total_steps} loss: {loss.item():.4f}")

        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            val_loss=evaluate(accelerator.unwrap_model(model),val_loader,accelerator)
            print(f"epoch: {epoch+1} | validation loss: {val_loss:.4f}")

            weights=accelerator.unwrap_model(model).state_dict()
            torch.save(weights,f"{OUT_DIR}/last_sft.pt")

            if val_loss<best_val:
                best_val=val_loss
                torch.save(weights,f"{OUT_DIR}/best_sft.pt")
                print("saved best_sft.pt")

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()