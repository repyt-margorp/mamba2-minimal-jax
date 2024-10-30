from jax_model import Mamba, ModelArgs
from transformers import AutoTokenizer

# One of:
#     'state-spaces/mamba-2.8b-slimpj'
#     'state-spaces/mamba-2.8b'
#     'state-spaces/mamba-1.4b'
#     'state-spaces/mamba-790m'
#     'state-spaces/mamba-370m'
#     'state-spaces/mamba-130m'
pretrained_model_name = 'state-spaces/mamba-370m'

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
model, params = Mamba.from_pretrained(pretrained_model_name, tokenizer=tokenizer)

import jax
import jax.numpy as np

def jax_generate(model,
                 params, 
                 tokenizer,
                 prompt: str,
                 n_tokens_to_gen: int = 50,
                 sample: bool = True,
                 top_k: int = 40,
                 rng = jax.random.PRNGKey(7),
                 ):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids # In pytorch format
    input_ids = np.array(input_ids.numpy()) # In jax format

    for token_n in range(n_tokens_to_gen):
        indices_to_input = input_ids
        next_token_logits = model.apply(params, indices_to_input)[:, -1]

        probs = jax.nn.softmax(next_token_logits, axis=-1)

        if top_k is not None:
            (values, indices) = jax.lax.top_k(probs, k=top_k)
            mask = probs < np.expand_dims(values[:, -1], axis=1)
            probs = np.where(mask, 0.0, probs)
            probs = probs / probs.sum(axis=1, keepdims=True)

        if sample:
            # TODO, might not be 100% correct. 
            rng, subrng = jax.random.split(rng)
            next_indices = jax.random.categorical(subrng, jax.nn.log_softmax(probs), 1, shape=probs.shape[:-1]+(1,))
        else:
            next_indices = np.argmax(probs, axis=-1, keepdims=True)

        input_ids = np.concatenate([input_ids, next_indices], axis=1)
    
    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]

    return output_completions

sample=False

print(jax_generate(model, params, tokenizer, 'Mamba is the', sample=sample))
print(jax_generate(model, params, tokenizer, 'John: Hi!\nSally:', sample=sample))
print(jax_generate(model, params, tokenizer, 'The meaning of life is ', sample=sample))
print(jax_generate(model, params, tokenizer, 'def reverse_string(', sample=sample))