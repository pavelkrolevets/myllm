import os
import torch
import tiktoken
from gpt_model import (
    GPTModel,
    GPT_CONFIG_124M,
    plot_losses,
    evaluate_model,
    generate_and_print_sample,
    calc_loss_batch,
)
from dataset_loader import create_dataloader_v1

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Find latest checkpoint in checkpoints directory
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in the given directory."""
    if not os.path.exists(checkpoint_dir):
        return None, 0

    latest_epoch = -1
    latest_path = None

    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("model_and_optimizer.pth") and ".ep" in filename:
            # Extract epoch number from filename like "model_and_optimizer.pth.ep20"
            try:
                epoch_str = filename.split(".ep")[-1]
                epoch_num = int(epoch_str)
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_path = os.path.join(checkpoint_dir, filename)
            except ValueError:
                continue

    # Also check for base checkpoint without epoch suffix
    base_path = os.path.join(checkpoint_dir, "model_and_optimizer.pth")
    if os.path.exists(base_path):
        try:
            checkpoint = torch.load(base_path, map_location=device)
            saved_epoch = checkpoint.get("epoch", 0)
            if saved_epoch > latest_epoch:
                latest_epoch = saved_epoch
                latest_path = base_path
        except Exception:
            pass

    return latest_path, latest_epoch


checkpoint_path, start_epoch = find_latest_checkpoint(checkpoint_dir)

num_epochs = 50

# Check if model is already fully trained
if start_epoch >= num_epochs:
    print(f"Model already trained to {start_epoch} epochs (target: {num_epochs}). Training complete!")
    # Load the model for inference
    checkpoint = torch.load(checkpoint_path, map_location=device) if checkpoint_path else None
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate sample to verify
    generate_and_print_sample(model, tokenizer, device, "Every effort moves you")
    print("Model is ready for inference.")
else:
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path} (epoch {start_epoch})")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = GPTModel(GPT_CONFIG_124M)
        model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.train()
    else:
        print(f"No checkpoint found in {checkpoint_dir}, training from scratch")
        model = GPTModel(GPT_CONFIG_124M)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
        start_epoch = 0
        model.train()

checkpoint_interval = 20

# Training loop with checkpoint saving every 20 epochs
train_losses, val_losses, tokens_seen = [], [], []
tokens_seen_count, global_step = 0, -1

for epoch in range(start_epoch, num_epochs):
    model.train()
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        loss.backward()
        optimizer.step()
        tokens_seen_count += input_batch.numel()
        global_step += 1

        if global_step % 5 == 0:
            train_loss, val_loss = evaluate_model(
                model, train_loader, val_loader, device, 5
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            tokens_seen.extend([tokens_seen_count])
            print(
                f"Ep {epoch+1} (Step {global_step:06d}): "
                f"Train loss {train_loss:.3f}, "
                f"Val loss {val_loss:.3f}"
            )

    # Generate sample
    generate_and_print_sample(model, tokenizer, device, "Every effort moves you")

    # Save checkpoint every 20 epochs
    if (epoch + 1) % checkpoint_interval == 0:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            f"{checkpoint_path}.ep{epoch+1}",
        )
        print(f"Checkpoint saved to {checkpoint_path}.ep{epoch+1} at epoch {epoch+1}")

# Save final checkpoint
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": num_epochs,
    },
    checkpoint_path,
)
print(f"Final checkpoint saved to {checkpoint_path}")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
