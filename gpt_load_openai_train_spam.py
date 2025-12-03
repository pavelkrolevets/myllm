import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
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
    evaluate_model,
    load_weights_into_gpt,
    calc_accuracy_loader_spam,
)
from gpt_download import download_and_load_gpt2
import os
import tensorflow as tf
import json
from gpt_download import load_gpt2_params_from_tf_ckpt


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


# df = pd.read_csv(
#     "sms_spam_collection/SMSSpamCollection.tsv",
#     sep="\t",
#     header=None,
#     names=["Label", "Text"],
# )
# print(df)
# print(df["Label"].value_counts())


# def create_balanced_dataset(df):
#     num_spam = df[df["Label"] == "spam"].shape[0]
#     ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
#     balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
#     return balanced_df


# balanced_df = create_balanced_dataset(df)
# print(balanced_df["Label"].value_counts())

# balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})


# def random_split(df, train_frac, validation_frac):
#     df = df.sample(frac=1, random_state=123).reset_index(drop=True)
#     train_end = int(len(df) * train_frac)
#     validation_end = train_end + int(len(df) * validation_frac)
#     train_df = df[:train_end]
#     validation_df = df[train_end:validation_end]
#     test_df = df[validation_end:]
#     return train_df, validation_df, test_df


# train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
# train_df.to_csv("train.csv", index=None)
# validation_df.to_csv("validation.csv", index=None)
# test_df.to_csv("test.csv", index=None)

tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)
print(train_dataset.max_length)

val_dataset = SpamDataset(
    csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer
)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
# settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
model_dir = os.path.join("gpt2", model_size)
tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
settings = json.load(
    open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8")
)
params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"],
)
print(token_ids_to_text(token_ids, tokenizer))

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"],
)
print(token_ids_to_text(token_ids, tokenizer))

for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"], out_features=num_classes
)
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)

with torch.no_grad():
    outputs = model(inputs)
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)

probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label:", label.item())
logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("Class label:", label.item())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.manual_seed(123)
train_accuracy = calc_accuracy_loader_spam(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader_spam(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader_spam(test_loader, model, device, num_batches=10)
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")
