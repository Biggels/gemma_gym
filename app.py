import os

import huggingface_hub
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()


def main():
    huggingface_hub.login(token=os.environ["HF_TOKEN"])

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507",
        dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", padding_side="left"
    )

    messages = [
        {
            "role": "system",
            "content": """You are a coding function that accepts instructions for a program as input and returns the exact text of a valid program as output, 
            beginning with ```python and ending with ```, with no extraneous comments or any non-code content. You will receive a description of a program now.""",
        },
        {
            "role": "user",
            "content": "Checks if a given string is a valid ISBN-10 number.",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_length = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs, max_new_tokens=1000, do_sample=True, temperature=0.9
    )
    output_text = tokenizer.batch_decode(
        outputs[:, input_length:], skip_special_tokens=True
    )[0]

    print(output_text)

    program = output_text.split("```python")[1].split("```")[0]
    with open("gen.py", "w") as file:
        file.write(program)


if __name__ == "__main__":
    main()
