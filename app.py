import os

import huggingface_hub
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()


def main():
    huggingface_hub.login(token=os.environ["HF_TOKEN"])

    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        "google/codegemma-2b",
        dtype="auto",
        device_map="auto",
        # quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "google/codegemma-2b", padding_side="left"
    )

    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a coding function that accepts instructions for a program as input and returns the exact text of a valid program as output, with no extraneous comments or any non-code content.",
    #     },
    #     {
    #         "role": "user",
    #         "content": "Write a python program that checks if a given string is a valid ISBN-10 number.",
    #     },
    # ]
    # inputs = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     tokenize=True,
    #     return_dict=True,
    #     return_tensors="pt",
    # ).to(model.device)

    # prompt = """You are a coding function that accepts instructions for a program as input and returns the exact text of a valid program as output, with no extraneous comments or any non-code content. Your instructions: write a python program that checks if a given string is a valid ISBN-10 number. <code>"""
    prompt = """Here is a python program that checks if a given string is a valid ISBN-10 number: ```python"""
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    # input_length = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs, max_new_tokens=1000, do_sample=True, temperature=0.9
    )
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    # print(
    #     tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)[0]
    # )
    # print(
    #     tokenizer.decode(
    #         outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    #     )
    # )


if __name__ == "__main__":
    main()
