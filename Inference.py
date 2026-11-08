import torch
from tokenizers import Tokenizer
from Pre_Training.util import Decoder
from Pre_Training.Pre_Config import Config

cfg=Config()
CKPT="best_sft.pt"
TOKENIZER="my_tokenizer/tokenizer.json"

SYS,USER,ASSIST,EOS=50257,50258,50259,50256 #these are the token number of the special tokens
SYSTEM_PROMPT=("You are Lyra, a helpful assistant. "
                 "You are a 114-million-parameter model developed by Deviprasad BRL. Answer the user's question clearly using your knowledge. Do not decline simple factual or language tasks.")

# sampling settings (change these to experiment)
TEMPERATURE,TOP_K,TOP_P,REP_PENALTY,MAX_NEW=0.7,30,0.9,1.05,500

device="cuda" if torch.cuda.is_available() else "cpu"
tok=Tokenizer.from_file(TOKENIZER)

def encode(text): 
    return tok.encode(text).ids


def load_model(path=CKPT):
    model=Decoder(vocab_size=cfg.vocab_size,d_model=cfg.d_model,num_heads=cfg.num_heads,num_kv=cfg.num_kv,num_blocks=cfg.num_blocks,intermediate_size=cfg.intermediate_size,max_length=cfg.max_length)
    
    ckpt=torch.load(path,map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model.to(device).eval()


def messages_to_ids(messages):
    """Exactly the training format: role token + text, assistant turns end with EOS, then <|assistant|>."""
    ids=[]
    for m in messages:
        t=encode(m["content"])

        if m["role"]=="system":
            ids+=[SYS]+t

        elif m["role"]=="user":
            ids+=[USER]+t

        else:
            ids+=[ASSIST]+t+[EOS]

    return ids+[ASSIST]


def sample_next(logits,generated,temperature,top_k,top_p,rep_pen):
    logits=logits.float().clone()
    logits[[SYS, USER, ASSIST]]=-float("inf")     

    if rep_pen!=1.0 and generated:                         
        idx=torch.tensor(list(set(generated)), device=logits.device)
        l=logits[idx]
        logits[idx]=torch.where(l>0,l/rep_pen,l*rep_pen)
    
    if temperature<=0:
        return int(logits.argmax())
    
    logits=logits/temperature
    
    if top_k > 0:
        v,_=torch.topk(logits,min(top_k,logits.size(-1)))
        logits[logits<v[-1]]=-float("inf")

    probs=torch.softmax(logits,-1)
    
    if top_p<1.0:
        sp,si=torch.sort(probs,descending=True)
        sp[torch.cumsum(sp,-1)-sp>top_p]=0

        probs=torch.zeros_like(probs).scatter_(0,si,sp)
        probs=probs/probs.sum()
    return int(torch.multinomial(probs,1))


@torch.no_grad()
def generate(model,prompt_ids,temperature=TEMPERATURE,top_k=TOP_K,top_p=TOP_P,
             rep_pen=REP_PENALTY,max_new=MAX_NEW):
    
    ids,new=list(prompt_ids),[]
    
    for _ in range(max_new):
        x=torch.tensor([ids[-cfg.max_length:]],device=device)
        
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(device == "cuda")):
            logits = model(x)[0,-1]  

        nxt=sample_next(logits,new,temperature,top_k,top_p,rep_pen)
        
        if nxt==EOS:
            break

        ids.append(nxt)
        new.append(nxt)
    return new


def ask(model,question,context=None,history=None,extra_system=None,**kw):
    system=(SYSTEM_PROMPT+("\n"+extra_system if extra_system else "")+("\n\nContext:\n"+context if context else ""))

    msgs=[{"role": "system","content": system}]+(history or [])+[{"role": "user", "content": question}]

    return tok.decode(generate(model,messages_to_ids(msgs),**kw), skip_special_tokens=False).strip()


def chat():
    model=load_model()
    context,history=None,[]
    print("Commands: /ctx <passage>  set RAG context | /noctx  remove it | /clear  new conversation | /quit")
    while True:
        user=input("You: ").strip()
        
        if not user:
            continue

        if user=="/quit":
            break

        if user=="/clear":
            history=[]
            print("(conversation cleared)") 
            continue

        if user=="/noctx":
            context,history=None,[] 
            print("(context removed)")
            continue

        if user.startswith("/ctx "):
            context,history=user[5:].strip(),[]
            print("(context set, conversation cleared)")
            continue

        history.append({"role":"user","content":user})
        system=SYSTEM_PROMPT+("\n\nContext:\n"+context if context else "")
        
        while True: 
            msgs=[{"role": "system", "content": system}]+history
            ids=messages_to_ids(msgs)
            
            if len(ids)+MAX_NEW<=cfg.max_length or len(history)<=1:
                break

            history=history[2:]

        reply=tok.decode(generate(model,ids),skip_special_tokens=False).strip()
        print("Lyra:", reply)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    chat()