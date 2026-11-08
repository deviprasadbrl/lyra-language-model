import torch 
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from util import Decoder
from Pre_Config import  Config
from data import make_datasets
import os,re,glob,math,time,zlib

# The code written down is specifically written for kaggle environement with kaggle paths. 
# So while using  this script change the paths as per your environment

cfg=Config()

ROOT="corpus" #dataset
OUT="/kaggle/working" #change it as per your out_dir 
ORDER_PATH ="train_order_b2048.npy" #An data order created so we can use the same order throughout the project

MAX_TOKENS=3000000000   #3 billion tokens
BLOCK=cfg.max_length 
MAX_LR=3e-4
MIN_LR=3e-5
WARMUP=300

SAVE_EVERY=100
LOG_EVERY=20
EVAL_EVERY=500
STEPS_PER_SESSION=3000

acc=Accelerator(mixed_precision="fp16",gradient_accumulation_steps=cfg.accumulation)

RANK=acc.process_index
WORLD=acc.num_processes
dev=acc.device

torch.manual_seed(0)

train,val=make_datasets(ROOT,BLOCK,ORDER_PATH)

blocks_per_step=cfg.micro_batch*cfg.accumulation*WORLD
total_steps=min(len(train),MAX_TOKENS//BLOCK)//blocks_per_step

CHECK={
    "blocks_per_step":blocks_per_step,
    "total_steps":total_steps,
    "order_crc":zlib.crc32(train.index.tobytes()),
}

acc.print(
    f"world={WORLD} blocks={len(train):,}  "
    f"total_steps={total_steps}  "
    f"tokens={total_steps*blocks_per_step*BLOCK/1e9:.3f}B"
)

def get_lr(step):
    if step<WARMUP:
        return MAX_LR*(step+1)/WARMUP
    progress=min(1.0,(step-WARMUP)/(total_steps-WARMUP))
    return MIN_LR+0.5*(MAX_LR-MIN_LR)*(1+math.cos(math.pi*progress))

model=Decoder(
    vocab_size=cfg.vocab_size,
    d_model=cfg.d_model,
    num_heads=cfg.num_heads,
    num_kv=cfg.num_kv,
    num_blocks=cfg.num_blocks,
    intermediate_size=cfg.intermediate_size,
    max_length=cfg.max_length
)
acc.print(f"params:{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

decay_params=[p for p in model.parameters() if p.dim()>=2]
nodecay_params=[p for p in model.parameters() if p.dim()<2]


optimizer=optim.AdamW(
    [
        {"params":decay_params,"weight_decay": 0.1},
        {"params":nodecay_params,"weight_decay": 0.0},
    ],
    lr=MAX_LR,
    betas=(0.9,0.95),
)

ckpt_files=(glob.glob(f"{OUT}/ckpt_step*.pt")+glob.glob("/kaggle/input/**/ckpt_step*.pt",recursive=True))

def step_from_path(p):
    return int(re.findall(r"ckpt_step(\d+)",p)[-1])

path=max(ckpt_files, key=step_from_path, default=None)
step=0

if path is not None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["check"]==CHECK, (
        f"Data/schedule changed!\n"
        f"saved={ckpt['check']}\n"
        f"now={CHECK}"
    )
    model.load_state_dict(ckpt["model"])

model,optimizer=acc.prepare(model,optimizer)

if path is not None:
    optimizer.load_state_dict(ckpt["opt"])
    acc.scaler.load_state_dict(ckpt["scaler"])
    step = ckpt["step"]
    acc.print(f"RESUMED at step {step}/{total_steps} from {path}")
    del ckpt

start_step=step


def save_checkpoint(step: int):
    if not acc.is_main_process:
        return

    state={
        "model":acc.unwrap_model(model).state_dict(),
        "opt":optimizer.state_dict(),
        "scaler":acc.scaler.state_dict(),
        "step":step,
        "check":CHECK,
    }

    tmp=f"{OUT}/tmp.pt"
    torch.save(state, tmp)
    os.replace(tmp,f"{OUT}/ckpt_step{step:06d}.pt")

    # keep only the newest 2 checkpoints
    old=sorted(glob.glob(f"{OUT}/ckpt_step*.pt"))
    for f in old[:-2]:
        os.remove(f)

    print(f"[ckpt] saved step {step}",flush=True)


@torch.no_grad()
def evaluate():
    model.eval()
    results={}

    for split_name,ds in val.items():
        indices=range(RANK,len(ds) //cfg.micro_batch,WORLD)
        total=torch.zeros(2,device=dev)

        for i in indices:
            x,y=ds.batch(i,cfg.micro_batch)
            x,y=x.to(dev),y.to(dev)
            logits=model(x).float()
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),y.view(-1))
            total[0]+=loss
            total[1]+=1

        total=acc.reduce(total,reduction="sum")
        avg=(total[0]/total[1]).item()
        results[split_name]={"loss":round(avg,4),"perplexity":round(math.exp(avg), 2)}

    model.train()
    return results

loss_sum=torch.zeros((),device=dev)
n_acc=0 #number of batches used for the loss
t_log=time.time()
last_log=step

model.train()

while step<total_steps:
    lr=get_lr(step)
    for group in optimizer.param_groups:
        group["lr"]=lr

    for a in range(cfg.accumulation):
        batch_idx=step*cfg.accumulation*WORLD+a*WORLD+RANK
        x,y=train.batch(batch_idx,cfg.micro_batch)
        x,y=x.to(dev),y.to(dev)

        with acc.accumulate(model):
            logits=model(x).float()
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),y.view(-1))
            acc.backward(loss)

            if acc.sync_gradients:
                acc.clip_grad_norm_(model.parameters(),1.0)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_sum+=loss.detach()
        n_acc+=1

    step+=1
    session_over=(step-start_step)>=STEPS_PER_SESSION
    save_now=(step%SAVE_EVERY==0 or step==total_steps or session_over)

    # logging
    if step%LOG_EVERY==0 or save_now:
        avg_loss=acc.reduce(loss_sum/n_acc,reduction="mean").item()
        loss_sum.zero_()
        n_acc=0
        perplexity=math.exp(avg_loss)

        now=time.time()
        tok_per_sec=blocks_per_step*BLOCK*(step-last_log)/(now-t_log)
        t_log=now
        last_log=step

        acc.print(
            f"step {step}/{total_steps} | loss {avg_loss:.4f} | perplexity {perplexity:.2f} | "
            f"lr {lr:.2e} | {tok_per_sec:,.0f} tok/s"
            )

        if acc.is_main_process:
            with open(f"{OUT}/train_log.csv","a") as f:
                f.write(f"{step},train,{avg_loss:.4f},{perplexity:.2f}\n")

    # evaluation
    if step%EVAL_EVERY==0 or step==total_steps:
        val_loss=evaluate()
        acc.print("VAL",val_loss)
        if acc.is_main_process:
            with open(f"{OUT}/train_log.csv","a") as f:
                for name, r in val_loss.items():
                    f.write(f"{step},val_{name},{r['loss']},{r['perplexity']}\n")

    # checkpoint
    if save_now:
        save_checkpoint(step)

    if session_over and step<total_steps:
        acc.print("Session step limit reached. Checkpoint saved.")
        break

# final save
if step==total_steps and acc.is_main_process:
    torch.save(
        acc.unwrap_model(model).state_dict(),
        f"{OUT}/final_model.pt",
    )
    print("TRAINING COMPLETE")

acc.wait_for_everyone()