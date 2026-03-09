## 导入相关包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import logging

logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

torch.manual_seed(1024)

@dataclass
class GPTConfig:
    vocab_size: int = 50274 # gpt官方tokenizer
    block_size: int = 512 # 文本最大长度
    batch_size: int = 2
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    hidden_dim: int = n_embd
    dropout: float = 0.1
    head_size: int = n_embd // n_head

# attention head
class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.hidden_dim, config.head_size, bias=False)
        self.value = nn.Linear(config.hidden_dim, config.head_size, bias=False)
        self.query = nn.Linear(config.hidden_dim, config.head_size, bias=False)

        # attention mask
        self.register_buffer("attention_mask",
                              torch.tril(torch.ones(config.block_size,
                                                    config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X):
        # X: BLC
        batch_size, seq_length, hidden_dim = X.size()
        k = self.key(X)
        v = self.value(X)
        q = self.query(X)
        weights = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
        weights = weights.masked_fill(self.attention_mask[:seq_length, :seq_length] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        output = weights @ v

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, X):
        head_outputs = [head(X) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        output = self.proj(concat)
        output = self.dropout(output)

        return output

# mlp layer
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, X):
        return self.net(X)

# transformer block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)

    def forward(self, X):
        X = X + self.attn(self.ln1(X))
        X = X + self.ffwd(self.ln2(X))

        return X

# GPT model
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # embedding, position, norm, block, mlp
        # position embedding: 从0，1升级到rope
        # norm: layer_norm -> rms norm
        # mlp -> swiglu
        # mha -> gqa
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 现在的slm模型会用tie wieght减少参数
        self.token_embedding.weight = self.lm_head.weight # important for tie weight

    def _init_weights(self, Module):
        if isinstance(Module, nn.Linear):
            torch.nn.init.normal_(Module.weight, mean=0.0, std=0.02)
            if Module.bias is not None:
                torch.nn.init.zeros_(Module.bias)
        elif isinstance(Module, nn.Embedding):
            torch.nn.init.normal_(Module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: token_ids, targets: token_ids
        batch, seq_length = idx.size()
        token_emb = self.token_embedding(idx)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=idx.device)
        position_emb = self.position_embedding(position_ids)
        x = token_emb + position_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            batch, seq_length, vocab_size = logits.size()
            logits = logits.view(batch * seq_length, vocab_size)
            targets = targets.view(batch * seq_length)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx):
        # idx: bacth, seq_length
        for _ in range(512):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# 构建输入的 dataset
class MyDataset(Dataset):
    def __init__(self, path, block_size=512):
        # 使用 mobvoi_seq_monkey_general_open_corpus.jsonl 数据集，
        # 读取前 1000 行
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        import json

        self.encoded_data = []

        self.max_lines = 50000
        raw_data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])

        # 将长文本分割成训练样本
        for i in range(0, len(full_encoded), self.block_size):
            # 多取一个 Token 作为目标
            chunk = full_encoded[i:i+self.block_size+1]
            # 如果长度不够，用 eos_token 填充
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)

# 训练循环
def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        # 将数据移到设备上
        x, y = x.to(device), y.to(device)

        # 前向传播
        logits, loss = model(x, targets=y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 调整学习率
        scheduler.step()

        total_loss += loss.item()

        # if batch_idx % 100 == 0:
        #     print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    return total_loss

def eval(model, val_loader, device):
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
    return val_loss


if __name__ == '__main__':
    # 模型训练
    # train data
    train_dataset = MyDataset('/data/yanrui/demo/gpt/mobvoi_seq_monkey_general_open_corpus.jsonl')

    # split traindataset to train and val
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)
    
    model = GPT(GPTConfig())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 打印模型一共有多少参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # 设置 cosine 学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    for epoch in range(100):
        train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device)
        val_loss = eval(model, val_loader, device)
        # print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        log_str = f"Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}"
        logging.info(log_str)

        # 保存模型
        avg_val_loss = val_loss / len(val_loader)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }
        # 保存每个epoch的模型
        if epoch % 10 == 0 and epoch != 0:
            torch.save(checkpoint, f'/data/yanrui/demo/gpt/checkpoints/model_epoch_{epoch}.pt')