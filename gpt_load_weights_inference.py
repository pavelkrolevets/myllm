import torch
import tiktoken
from gpt_model import (
    GPTModel,
    GPT_CONFIG_124M,
    calc_loss_loader,
    train_model_simple,
    plot_losses,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
    generate,
)

torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Hello whats your name?", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4,
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
