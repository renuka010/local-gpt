import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


def load_llm():
    bnb_config = BitsAndBytesConfig(
        # load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_name = "/home/demo/.cache/kagglehub/models/mistral-ai/mistral/pyTorch/7b-instruct-v0.1-hf/1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # load_in_4bit=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    return pipe

pipe = load_llm()