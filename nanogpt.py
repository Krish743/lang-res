import torch
from torch import nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32
# head_size = 16

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()



chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text))



n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(6767)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])  #torch.stack just turns it into a tensor in this case.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch("train")


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(n_embed, n_embed)))
    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x) # (B,T,16)
        q = self.query(x) # (B,T,16)

        #now focuses on a token based on learned params instead of giving equal importance to each token
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, 16) @ (B, 16 , T) ---> (B, T, T)

        
        # wei = torch.zeros(T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0 , float('-inf')) # no peeking foward hehe
        wei = F.softmax(wei, dim=1)
        
        v = self.value(x)
        out = wei @ v # the v gets aggregated
        return out

class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.positional_embedding(torch.arange(T, device = device ))
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -block_size:] #we have to crop it to only have 8 char because we have positional encoding

            logits, loss = self(idx_cropped)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples =1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
model = BigramModel()
model = model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

batch_size = 32
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {iter} | Train loss {losses['train']:.4f} | val loss {losses['val']:.4f}')
    xb, yb= get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

print(loss.item())
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(model.generate(idx = context, max_new_tokens=500)[0].tolist()))

