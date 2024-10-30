import time
import torch
from transformers import AutoTokenizer

#from model2 import Mamba2LMHeadModel
from model2_print import Mamba2LMHeadModel
#from jax_model2_print import Mamba2LMHeadModel

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()

model = Mamba2LMHeadModel.from_pretrained("state-spaces/mamba2-370m", device=device)
#model = Mamba2LMHeadModel.from_pretrained("state-spaces/mamba2-1.3b", device=device)
#model = Mamba2LMHeadModel.from_pretrained("state-spaces/mamba2-2.7b", device=device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id

generation_config = dict(
    max_new_length=200,
    temperature=1.0,
    top_k=30,
    top_p=1.0,
)

def generate(
    prompt: str,
    seed: int = 0,
    n_prediction : int = 100,
    show_perf: bool = True):
    """Generate streaming completion"""
    torch.manual_seed(seed)

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)[0]
    print(prompt, end="")

    start = time.process_time()
    n_generated = 0
    for i, (token_id, _hidden_state) in enumerate(model.generate(input_ids, **generation_config)):
        token = tokenizer.decode([token_id])
        if i == 0:
            now = time.process_time()
            prompt_eval_elapsed, start = now - start, now
        else:
            n_generated += 1
        if n_generated >= n_prediction:
            break
        print(token, end="", flush=True)
    if show_perf:
        elapsed = time.process_time() - start
        print('\n\n---')
        print(f'Prompt eval | tokens: {input_ids.shape[0]} | elapsed: {prompt_eval_elapsed:.2f}s | tok/s: {input_ids.shape[0] / prompt_eval_elapsed:.2f}')
        print(f'Generation | tokens: {n_generated} | elapsed: {elapsed:.2f}s | tok/s: {n_generated / elapsed:.2f}')

if __name__ == "__main__":
    generate("""On a dark September night, a fierce storm jolted Florida with winds and torrential rainfall. Less than two weeks later, Hurricane Milton has roared through the state, mercilessly hampering recovery efforts already underway.

After Helene’s strong winds, heavy rains and a wall of water took 20 lives in the state along its path from south to north, Milton has claimed at least 17 more, bringing the ocean’s fury ashore with several feet of storm surge, three months worth of rain in three hours to some areas and a deadly tornado outbreak as it churned from west to east.

The trail of destruction all the way from the """, n_prediction=1)

