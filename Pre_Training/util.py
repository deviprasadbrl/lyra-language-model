import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self,d_model,eps=1e-6):
        super().__init__()
        self.eps=eps
        self.gamma=nn.Parameter(torch.ones(d_model))

    def forward(self,x):
        x_dtype=x.dtype

        x=x.to(torch.float32)

        square_mean=x.pow(2).mean(dim=-1,keepdim=True)
        rms=x*torch.rsqrt(square_mean+self.eps)

        weighted_rms=(self.gamma*rms).to(x_dtype)

        return weighted_rms

class SwiGLU(nn.Module):
    def __init__(self,d_model,intermediate_size):
        super().__init__()
        #The gate and vlaue weights are combined on the same projection
        self.gate_value=nn.Linear(d_model,2*intermediate_size,bias=False)

        self.out_gate=nn.Linear(intermediate_size,d_model,bias=False)

    def forward(self,x):
        combined=self.gate_value(x)
        gate,value=torch.chunk(combined,chunks=2,dim=-1)

        sigmoid_out=F.silu(gate)*value

        return self.out_gate(sigmoid_out)

class Rope(nn.Module):
    def __init__(self,head_dim,max_length,base=10000.0):
        super().__init__()

        speeds=1.0/(base**(torch.arange(0,head_dim,2).float()/head_dim))
        

        positions=torch.arange(max_length).float()
        angles=torch.outer(positions,speeds)

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self,x):
        seq_len=x.shape[-2]  # x:[B,H,T,D]
        cos=self.cos[:seq_len]
        sin=self.sin[:seq_len]

        cos=cos[None,None,:,:]   # [1, 1, T, D/2]
        sin=sin[None, None,:,:]   # Pytorch supports the before broadcasting but this for an safer approach

        x1=x[...,0::2]
        x2=x[...,1::2]

        new1=x1*cos-x2*sin
        new2=x1*sin+x2*cos
        out=torch.stack([new1,new2],dim=-1).flatten(-2)
        return out.to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self,d_model,num_q,num_kv,max_length):
        super().__init__()

        assert d_model%num_q==0 ,"d_model must be divisible by num_q"
        assert num_q%num_kv==0,"num_q must be divisible by num_kv"

        self.d_model=d_model
        self.num_q=num_q
        self.num_kv=num_kv
        self.max_length=max_length

        self.head_dim=self.d_model//self.num_q
        self.num_q_per_kv=self.num_q//self.num_kv

        self.scale=math.sqrt(self.head_dim)

        self.q_proj=nn.Linear(self.d_model,self.num_q*self.head_dim,bias=False)
        self.k_proj=nn.Linear(self.d_model,self.num_kv*self.head_dim,bias=False)
        self.v_proj=nn.Linear(self.d_model,self.num_kv*self.head_dim,bias=False)
        self.out_proj=nn.Linear(self.d_model,self.d_model,bias=False)
        self.rope=Rope(head_dim=self.head_dim,max_length=self.max_length)

    def forward(self,x,is_causal=False):
        batch,seq,_=x.shape

        assert seq<=self.max_length,"The seq length should not be greater than max_length"

        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        q=q.view(batch,-1,self.num_q,self.head_dim).transpose(1,2)
        k=k.view(batch,-1,self.num_kv,self.head_dim).transpose(1,2)
        v=v.view(batch,-1,self.num_kv,self.head_dim).transpose(1,2)

        q=self.rope(q)
        k=self.rope(k)
        
        if self.num_q_per_kv>1:
            k=k.repeat_interleave(self.num_q_per_kv,dim=1)
            v=v.repeat_interleave(self.num_q_per_kv,dim=1)


        score=torch.matmul(q,k.transpose(-2,-1))/self.scale



        if is_causal:
                Lq,Lk=score.size(-2),score.size(-1)
                causal=torch.ones(Lq,Lk,dtype=torch.bool,device=x.device).tril()
                score=score.masked_fill(~causal,torch.finfo(score.dtype).min)

        attention_weights=F.softmax(score,dim=-1)

        context=torch.matmul(attention_weights,v)            

        context=context.transpose(1,2).contiguous().view(batch,-1,self.d_model)

        out=self.out_proj(context)

        return out



class OptimizedGroupedQueryAttention(nn.Module):
    def __init__(self,d_model,num_q,num_kv,max_length):
        super().__init__()

        assert d_model%num_q==0 ,"d_model must be divisible by num_q"
        assert num_q%num_kv==0,"num_q must be divisible by num_kv"

        self.d_model=d_model
        self.num_q=num_q
        self.num_kv=num_kv
        self.max_length=max_length

        self.head_dim=self.d_model//self.num_q
        self.num_q_per_kv=self.num_q//self.num_kv

        self.q_proj=nn.Linear(self.d_model,self.num_q*self.head_dim,bias=False)
        self.k_proj=nn.Linear(self.d_model,self.num_kv*self.head_dim,bias=False)
        self.v_proj=nn.Linear(self.d_model,self.num_kv*self.head_dim,bias=False)
        self.out_proj=nn.Linear(self.d_model,self.d_model,bias=False)

        self.rope=Rope(head_dim=self.head_dim,max_length=self.max_length)

    def forward(self,x,is_causal=False):
        batch,seq,_=x.shape
        assert seq<=self.max_length,"The seq length should not be greater than max_length"

        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        q=q.view(batch,-1,self.num_q,self.head_dim).transpose(1,2)
        k=k.view(batch,-1,self.num_kv,self.head_dim).transpose(1,2)
        v=v.view(batch,-1,self.num_kv,self.head_dim).transpose(1,2)

        q=self.rope(q)
        k=self.rope(k)


        if self.num_q_per_kv>1:
            k=k.repeat_interleave(self.num_q_per_kv,dim=1)
            v=v.repeat_interleave(self.num_q_per_kv,dim=1)

        
        context=F.scaled_dot_product_attention(q,k,v,is_causal=is_causal,attn_mask=None)
        
        context=context.transpose(1,2).contiguous().view(batch,-1,self.d_model)

        out=self.out_proj(context)

        return out 


class Decoder_layer(nn.Module):
    def __init__(self,d_model,num_heads,num_kv,intermediate_size,max_length):
        super().__init__()

        self.GQA=OptimizedGroupedQueryAttention(d_model=d_model,num_q=num_heads,num_kv=num_kv,max_length=max_length)
        self.SwiGlu=SwiGLU(d_model=d_model,intermediate_size=intermediate_size)
        self.RMSNorm1=RMSNorm(d_model=d_model)
        self.RMSNorm2=RMSNorm(d_model=d_model)

    def forward(self,x):
        normed_x1=self.RMSNorm1(x)
        
        attention=self.GQA(normed_x1,is_causal=True)

        x=x+attention

        normed_x2=self.RMSNorm2(x)
        swiglu_out=self.SwiGlu(normed_x2)
        x=x+swiglu_out

        return x

class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,num_kv,num_blocks,intermediate_size,max_length):
        super().__init__()

        self.embed=nn.Embedding(vocab_size,d_model)
        self.layers=nn.ModuleList([Decoder_layer(d_model=d_model,num_heads=num_heads,num_kv=num_kv,intermediate_size=intermediate_size,max_length=max_length) for _ in range(num_blocks)])

        self.RMSNorm=RMSNorm(d_model=d_model)
        self.final_proj=nn.Linear(d_model,vocab_size,bias=False)
        self.final_proj.weight=self.embed.weight   

        self._init_weights(num_blocks) 

    def _init_weights(self, num_blocks):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        scaled_std = 0.02 / math.sqrt(2 * num_blocks)
        for layer in self.layers:
            nn.init.normal_(layer.GQA.out_proj.weight, mean=0.0, std=scaled_std)
            nn.init.normal_(layer.SwiGlu.out_gate.weight, mean=0.0, std=scaled_std)

    def forward(self,x):
        x=self.embed(x)
        
        for layer in self.layers:
            x=layer(x)

        x=self.RMSNorm(x)

        out=self.final_proj(x)

        return out
