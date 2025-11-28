import torch
import self_attention_class
import casual_attention
import multi_head_attention_class

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],
    ]  # step
)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

keys = inputs @ W_key
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

keys_2 = keys[1]
attn_score_22 = query_2.dot(key_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)


d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

torch.manual_seed(789)
sa_v2 = self_attention_class.SelfAttention_v2(d_in, d_out)
print(sa_v2.forward(inputs))

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

context_length = attn_scores.shape[0]
masked_simple = torch.tril(torch.ones(context_length, context_length))
print(masked_simple)

masked_simple = attn_weights * masked_simple
print(attn_weights)
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print(attn_weights)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

torch.manual_seed(123)
context_length = batch.shape[1]
ca = casual_attention.CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs)


torch.manual_seed(123)
context_length = batch.shape[1]  # This is the number of tokens
d_in, d_out = 3, 2
mha = multi_head_attention_class.MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = multi_head_attention_class.MultiHeadAttention(
    d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# GPT-2 context 768
inputs_gpt = torch.Tensor(torch.rand(768, 3))
batch_gpt = torch.stack((inputs_gpt, inputs_gpt), dim=0)
print(batch_gpt.shape)

batch_size, context_length, d_in = batch_gpt.shape
print(context_length)
d_out = 12
mha = multi_head_attention_class.MultiHeadAttention(
    d_in, d_out, context_length, 0.0, num_heads=12
)
context_vecs = mha(batch_gpt)
print("context_gpt_vecs.shape:", context_vecs.shape)
