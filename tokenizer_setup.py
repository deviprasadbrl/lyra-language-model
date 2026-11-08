from transformers import GPT2TokenizerFast


tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")


tokenizer.add_special_tokens({
    "additional_special_tokens":[
        "<|system|>",
        "<|user|>",
        "<|assistant|>"
    ]
})
tokenizer.save_pretrained("my_tokenizer")

print(len(tokenizer))
