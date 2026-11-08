from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int=50260
    d_model: int=768
    num_heads: int=12
    num_kv: int=4
    num_blocks: int=12
    intermediate_size: int=2048
    max_length: int=2048
    accumulation: int=16
    micro_batch :int=4
