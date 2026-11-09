import json,torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader

''' The code written down is specifically written for kaggle environement with kaggle paths. 
    So while using  this script change the paths as per your environment '''


TRAIN_PATH="SFT_Datatrain.jsonl"
VAL_PATH="SFT_Data/eval.jsonl"

SYS,USER,ASSIST,EOS=50257,50258,50259,50256
MAX_LEN=2048

ROLE_TOKEN={"system":SYS,"user":USER}
BANNED=["<|system|>","<|user|>","<|assistant|>","<|endoftext|>"]

SOURCES=["chat", "system_follow", "identity", "multiturn", "cannot_do",
           "sys_instr", "arithmetic", "facts", "task_do", "unknown",
           "smalltalk", "sys_recall"]


def make_encoder(tokenizer):
    def encode(text):
        return tokenizer.encode(text).ids
    return encode


def valid_messages(msgs):
    if not isinstance(msgs, list) or len(msgs) < 3:
        return False
    
    roles=[m.get("role") for m in msgs]

    if roles[0]!="system" or roles[-1] != "assistant":
        return False
    
    for i,r in enumerate(roles[1:]):
        if r!=("user" if i%2==0 else "assistant"):
            return False
        
    for m in msgs:
        c=m.get("content")
        if not isinstance(c, str) or not c.strip():
            return False
        
        if any(b in c for b in BANNED):
            return False
    return True


def build_example(messages,encode):
    ids,labels=[],[]

    for m in messages:

        t=encode(m["content"])
        
        if m["role"] in ROLE_TOKEN:                 # system / user: model reads it, no loss
            seg=[ROLE_TOKEN[m["role"]]]+t
            ids+=seg
            labels+=[-100]*len(seg)
        
        else:                                       # assistant: role token masked, reply + EOS learned
            ids+=[ASSIST]
            labels+=[-100]

            ids+=t+[EOS]
            labels+=t+[EOS]
    
    return {"input_ids":ids,"labels":labels}


def process_file(path, encode):
    out,stats=[],Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            stats["rows"]+=1

            try:
                row=json.loads(line)
                msgs,source=row["messages"],row["source"]

            except (json.JSONDecodeError,KeyError,TypeError):
                stats["bad_json"]+=1
                continue

            if source not in SOURCES:
                stats["bad_source"]+=1
                continue

            if not valid_messages(msgs):
                stats["bad_format"]+=1
                continue

            ex=build_example(msgs, encode)
            if len(ex["input_ids"]) > MAX_LEN:
                stats["too_long"] += 1
                continue

            ex["source"]=source
            out.append(ex)

    stats["kept"]=len(out)
    return out,stats


def build_dataset(
    tok,
    train_path=TRAIN_PATH,
    val_path=VAL_PATH,
    out_path="sft_tokenized.pt",
):
    encode=make_encoder(tok)

    train,train_stats=process_file(train_path,encode)
    val,val_stats=process_file(val_path,encode)

    print("train:",dict(train_stats))
    print("val:",dict(val_stats))

    torch.save({"train": train, "val": val}, out_path)

    all_ex=train+val

    lens=sorted(len(e["input_ids"]) for e in all_ex)
    learned=sum(label != -100 for e in train for label in e["labels"])
    total=sum(len(e["input_ids"]) for e in train)

    print(
        f"train {len(train)} | val {len(val)} | "
        f"median len {lens[len(lens)//2]} | max len {lens[-1]}"
    )
    print(f"trained-on tokens: {learned} of {total} ({learned/total:.0%})")
    print("val per source:", dict(Counter(e["source"] for e in val)))


'''Once the json files are tokenized and saved inside .pt file we can directly use that to make batches using data loader the
    below code do the exact thing '''


class SFTDataset(Dataset):
    def __init__(self, ex):
        self.ex=ex

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


class SFTCollator:
    def __init__(self, pad_id=EOS):
        self.pad_id=pad_id

    def __call__(self, batch):
        L=max(len(x["input_ids"]) for x in batch)
        ids=torch.full((len(batch),L),self.pad_id,dtype=torch.long)
        lab=torch.full((len(batch),L),-100,dtype=torch.long)
        
        for i, x in enumerate(batch):
            n=len(x["input_ids"])

            ids[i,:n]=torch.tensor(x["input_ids"])
            lab[i,:n]=torch.tensor(x["labels"])
        return {"input_ids": ids, "labels": lab}


def make_loaders(bs,path="/kaggle/working/sft_tokenized.pt"):
    d=torch.load(path)
    collate=SFTCollator()

    train=DataLoader(SFTDataset(d["train"]),batch_size=bs, shuffle=True,collate_fn=collate,num_workers=2,pin_memory=True,drop_last=True)
    val=DataLoader(SFTDataset(d["val"]),batch_size=bs,shuffle=False,collate_fn=collate,num_workers=2,pin_memory=True)
    
    return train, val

'''For testing the Data Pipeline'''

# def sanity_check(loader, tok, n=3):
#     b=next(iter(loader))
    
#     for i in range(n):
#         ids,lab=b["input_ids"][i],b["labels"][i]
#         real=ids[(ids!=EOS)|(lab!=-100)]    # drop the padding
        
#         print("FULL   :", tok.decode(real.tolist(), skip_special_tokens=False))
#         print("LEARNED:", tok.decode(lab[lab != -100].tolist(), skip_special_tokens=False))
#         print("-" * 50)

