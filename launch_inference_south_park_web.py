import argparse
import datetime
import fire
import glog as log
import gradio as gr
import os
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import sys


prompt_input = (
    "Below is a script from the American animated sitcom South Park. "
    "Write a response that completes Cartman's last line in the conversation.\n\n"
    "### Script:\n{previous}\n\n ### Cartman:"
)


def generate_prompt(character, previous):
    return prompt_input.format_map({'previous': character + ": " + previous})


def main(
    server_name: str = "0.0.0.0",
    share_gradio: bool = False,
):
    parser = argparse.ArgumentParser(description='Run a commandline inference on a Hugging Face Transformer model.')
    parser.add_argument('--model', type=str, default='',
                         required=True, help="Path to the input Hugging Face model.")
    args = parser.parse_args()
    model_path = args.model
    log.check(os.path.exists(args.model))
    
    load_type = torch.float16
    if torch.cuda.is_available():
        # Make to work for multiple GPUs.
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    print('Loading model: {:s}'.format(model_path))
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
    )
    print('Loading model done.')

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size != tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(tokenzier_vocab_size)

    model.to(device)
    model.eval()


    def evaluate(
        character,
        talk,
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        num_beams=4,
        max_new_tokens=2048,
        stream_output=False,
        **kwargs,
    ):
        prompt = generate_prompt(character, talk)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=1.3,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        print(generation_config)
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device), 
                attention_mask = inputs['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config=generation_config,
        )
        token_ids = generation_output[0]
        output = tokenizer.decode(token_ids, skip_special_tokens=True)
        print('Raw output ({:d}): {:s}'.format(len(output), output))
        response = output.split("### Cartman:")[1].strip()
        print('Eric Cartman: {:s}\n'.format(response))
        yield response

    gr.Interface(
        fn=evaluate,
        allow_flagging="never",
        inputs=[
            gr.components.Dropdown(character="character", value="Stan",
                                   choices=["Stan", "Kyle", "Kenny"]),
            gr.components.Textbox(
                lines=2,
                label="talk",
                placeholder="Talk to the character",
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.7, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.9, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=4, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2048, step=1, value=2048, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="🍋🦙 Chat with Eric Cartman",
        description='<img src="https://www.hollywoodreporter.com/wp-content/uploads/2021/10/south-park-4.jpg" height=150>',
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
