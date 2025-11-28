import torch
import dataset_loader

vocab_size = 50257
output_dim = 256

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


