import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, max_seq_len, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, max_seq_len, embed_dim]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [batch_size, num_filters, max_seq_len-filter_size+1]
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # [batch_size, num_filters]
        x = torch.cat(x, 1)  # [batch_size, num_filters * len(filter_sizes)]
        x = self.dropout(x)
        logits = self.fc(x)  # [batch_size, num_classes]
        return logits

class ComplexCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Increased number of convolutional layers
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)),
                nn.ReLU(),
                nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=(fs, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1))
            )
            for fs in filter_sizes
        ])

        # Additional fully connected layers
        # self.fc1 = nn.Linear(len(filter_sizes) * n_filters, 256)  # Intermediate layer
        # self.fc2 = nn.Linear(256, output_dim)  # Final output layer
        # self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(14400, 256)  # Intermediate layer
        self.fc2 = nn.Linear(256, output_dim)  # Final output layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]

        # Apply convolution and pooling layers
        conved = [conv(embedded).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, (sent len - filter_sizes[n]) // 2]

        # Flatten the convolutional layer outputs
        flat_conved = [conv.view(conv.size(0), -1) for conv in conved]
        cat = self.dropout(torch.cat(flat_conved, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        # Fully connected layers
        hidden = self.fc1(cat)
        hidden = nn.ReLU()(hidden)
        hidden = self.dropout(hidden)
        return self.fc2(hidden)
    
    def embedded_to_flattened(self, embedded):
        conved = [conv(embedded).squeeze(3) for conv in self.convs]
        flat_conved = [conv.view(conv.size(0), -1) for conv in conved]
        cat = torch.cat(flat_conved, dim=1)
        return cat