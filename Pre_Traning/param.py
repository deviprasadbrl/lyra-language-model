
from util import Decoder
from Pre_Config import Config

config=Config()

decoder_model = Decoder(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    num_heads=config.num_heads,
    num_kv=config.num_kv,
    num_blocks=config.num_blocks,
    intermediate_size=config.intermediate_size,
    max_length=config.max_length
)
decoder_params = sum(p.numel() for p in decoder_model.parameters())


print(f"Decoder: {decoder_params:,}")
