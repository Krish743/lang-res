#all the em-dashes are user generated.

import torch
from torch import nn
from torch.nn import functional as F

batch_size = 64
block_size = 16  #context length
max_iters = 2000
eval_interval = 200
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 64
n_head = 4
n_layers = 2
dropout = .2
# head_size = 16

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('../data/langv2.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

torch.manual_seed(1337)


tokens = []
for line in text.splitlines():
    tokens.append("<BOS>")
    tokens.extend(line.split())
    tokens.append("<EOS>")

chars = sorted(list(set(tokens)))
vocab_size = len(chars)
stoi = {w:i for i, w in enumerate(chars)}
itos = {i:w for i, w in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s.split()]
decode = lambda l: ' '.join([itos[i] for i in l])

data = torch.tensor(
    [stoi[t] for t in tokens],
    dtype=torch.long
)

# data = torch.tensor(encode(text))


n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


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
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)

        #now focuses on a token based on learned params instead of giving equal importance to each token
        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)  # (B, T, head_size) @ (B, head_size , T) ---> (B, T, T)


        # wei = torch.zeros(T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0 , float('-inf')) # no peeking foward hehe, :T implemeted cause at the time of generation the T might be diff when starting
        wei = F.softmax(wei, dim=-1)  # the -inf elems gets the prob of 0 when passed thorugh softmax

        self.last_attention = wei.detach().cpu()

        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v # the v gets aggregated
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed) # num_heads * head_size = n_embed. so, its 32 -> 32
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) #combining all the features to form 32 features again.
        out = self.proj(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), #basicaly projection for ffwd, coupled straight into the sequential.
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.sa_head = MultiHeadAttention(n_head, n_embed//n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # adding residuals
        x = x + self.sa_head(self.ln1(x)) # we fork off, compute the sa and come back to the original pathway.
        x = x + self.ffwd(self.ln2(x))
        # we feed the x into LayerNorm before sending it to sa or ffwd.
        return x

class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        # self.sa_head = Head(n_embed)
        # self.sa_head = MultiHeadAttention(4, n_embed // 4)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(n_embed)
        # self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.positional_embedding(torch.arange(T, device = device ))
        x = tok_emb + pos_emb
        # x = self.sa_head(x)
        # x = self.ffwd(x)
        x = self.blocks(x)

        self.last_hidden = x.detach().cpu()

        x = self.ln(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #converting the shape casue the cross_entropy doenst take 3 dims, so we're combining all the elements from diff batches into a single vector dim.
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #!!!HEAVY LEARNING FROM HEAVY MISTAKE
        # idx_cropped = idx[:, -block_size:] -----> lol i tried to bring it up here thinking i was doing some crazy optimization -- welp, i was sending the autoregressive mech. on fire-- basically if it was out of the loop, it wont be able to use the newly generated tokens appended to it inside the loop -- so for every iter it would have just taken 0 as input basically "\n" xD
        for _ in range(max_new_tokens):

            idx_cropped = idx[:, -block_size:] #we have to crop it to only have 8 char because we have positional encoding
            #it will only have the last 8 elems/chars

            logits, loss = self(idx_cropped)
            logits = logits[:, -1, :] #taking only the last token, cause only the last token is used to generate the next token

            probs = F.softmax(logits, dim = -1)
             # will do softmax for each batch item -> here there is no sense of batch because the batch_size will be 1 but there is max_new_tokens. so, it would generate (1,vocab_size) softmax prob dist for every iter(basically for every new token)

            idx_next = torch.multinomial(probs, num_samples =1)
            if idx_next.item() == stoi["<EOS>"]:
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def main():
    model = BigramModel()
    model = model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr = lr) # AdamW is just adam-chan with weight decay decoupled from regular updatation.


    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f'step {iter} | Train loss {losses['train']:.4f} | val loss {losses['val']:.4f}')

        SAVE_STEPS = [100, 250, 500, 1000, 1250, 1500, 1999]

        if iter in SAVE_STEPS:
            torch.save(
                {
                    "iter": iter,
                    "model_state_dict": model.state_dict(),
                },
                f"langv2e1/ckpt_{iter}.pt"
            )
        xb, yb= get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    print(loss.item())
    context = torch.tensor(encode("na"), dtype = torch.long, device = device).unsqueeze(0)
    print(decode(model.generate(idx = context, max_new_tokens=2500)[0].tolist()))

if __name__ == '__main__':
    main()