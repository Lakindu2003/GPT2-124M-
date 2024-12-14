from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken

class CasualSelfAttention(nn.Module):
        
        def __init__(self, config):
                super().__init__()
                assert config.n_embd % config.n_head == 0

                #attention is basically a couple of matrix multiplication operations with learnable weights
                #this is equivalent to carry out many many linear regressions which is equivalent to using a linear layer.
                self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
                self.c_proj = nn.Linear(config.n_embd, config.n_embd)
               
                self.n_head = config.n_head
                self.n_embd = config.n_embd # maps the attention dims to the embedding dims.
                                            # Each embedding dim is expressed as a linear combination of all the attention dimensions and 
                                            # the appropriate weights are found via training.

                self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
                #torch.ones is used to generate a context_length*context_length matrix of 1s
                #torch.tril sets all values above the main diagonal to 0. This is used to mask the attention.
                #masked attention prevents data leakage.
               
                #torch.view() converts the mask matrix into batch_size*n_head*context_length*context_length
                #This conversion as the key and query matrices are in the form batch_size*n_head*context_length*context_length.

        def forward(self, x):
                #An entry in X corrsponds to a particular token that belongs to a particular batch,
                #which has a corresponding embedding vector
                B, T, C = x.size() #x is decomposed into batch_size, context_length, n_embd
                                   #each embedding dimension corresponds to a channel (that's why it's denoted by 'C')

                q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

                k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
                q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
                v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
                #initially, the matrix has dimensions batch_size*context_length*n_embd.
                #since we use multi-headed attention, we need to allocate the dimensions into different attention heads.
                #in order to so, we use torch.view()
                #transpose is used as the matrices have to be in the form batch_size*n_head*context_length*n_embd 
             
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #standard attention formula
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) #masking
                att = F.softmax(att, dim=-1) #standard attention formula
                y = att @ v #standard attention formula
                y = y.transpose(1, 2).contiguous().view(B, T, C) 
                # output projection
                y = self.c_proj(y) #final linear layer in attention block
                return y

class MLP(nn.Module):
    
    #no comments as this is explained in the Block class.
    def __init__(self, config):
           super().__init__()

           self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
           self.silu = nn.SiLU()
           self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        
    def forward(self, x):
           return self.c_proj(self.silu(self.c_fc(x)))

class Block(nn.Module):
        
        def __init__(self, config):
                super().__init__()

                self.ln_1 = nn.LayerNorm(config.n_embd) #normalisation layer before attention block
                self.attn = CasualSelfAttention(config) #attention block: key part of the transformer architecture
                self.ln_2 = nn.LayerNorm(config.n_embd) #normalisation layer before MLP
                self.mlp = MLP(config) #MLP where a SiLU activation is sandwiched between two linear layers
                #orginally GPT-2 used an approximated version of GELU, but I decided to use SiLU because it's more up-to-date.

        def forward(self, x):
                #LayerNorm + nnLayer + Skip/residual connections
                x += self.attn(self.ln_1(x)) 
                return x + self.mlp(self.ln_2(x))

@dataclass
class GPTConfig:
        block_size: int = 1024 #context length
        vocab_size: int = 50257 #number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
        n_layer: int = 12 #number of attention blocks
        n_head: int = 12 #number of heads for multiheaded attention
        n_embd: int = 768 #dimensions of the embedding and attention space

class GPT(nn.Module):
        
        def __init__(self, config):
                super().__init__()
                self.config = config

                self.transformer = nn.ModuleDict(dict(
                        wte = nn.Embedding(config.vocab_size, config.n_embd), #token embedding layer
                        wpe = nn.Embedding(config.block_size, config.n_embd), #positional embedding layer
                        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #list of parallelizable attention blocks
                        ln_f = nn.LayerNorm(config.n_embd) #final normalisation layer before the final Linear layer 
                ))
                self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) #final linear layer that maps the embedding space into token vocabulary.
                #i am guessing the softmax function is outside the GPTclass is because otherwise it would make the training more inefficient

        # just copy-pasting since there isn't much to it
        def forward(self, idx):
                # idx is of shape (B, T)
                B, T = idx.size()
                assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
                
                # forward the token and position embeddings
                pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape [T]
                pos_emb = self.transformer.wpe(pos) # positional embeddings of shape (T, n_embd)
                tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
                x = tok_emb + pos_emb #tok_emb is broadcasted so that’s its dimensions are compatible with pos_emb. The pos_embs are identical for each batch.
                
                # forward the blocks of the transformer
                for block in self.transformer.h:
                    x = block(x)
                # forward the final layernorm and the classifier
                x = self.transformer.ln_f(x)
                logits = self.lm_head(x) # (B, T, vocab_size)
                return logits #there aren't probabilities


num_return_sequences = 5
max_length = 30
device = "cuda"
seed = 42

torch.cuda.set_per_process_memory_fraction(0.99) 



model = GPT(GPTConfig())
model.eval()
model.to(device)

# tiktoken is the GPT-2 tokeniser from huggingface
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # [8,]
x = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # [5, 8]
x = x.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# let the model do it's thing!
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
while x.size(1) < max_length:
    
    # forward the model to get the logits
    # torch.no_grad() indicates to PyTorch that no backpropagation takes place. 
    # This reduces the amount of space used by PyTorch in caching the intermediate tensors needed for backpropagation. 
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)

        # take the logits corresponding to the last position (poistion closest to token being predicted)
        logits = logits[:, -1, :] # (B, vocab_size)
        
        # convert the logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # `torch.topk`, where k=50, is used to filter the 50 tokens with the highest probability 
        #and renormalize the probability distribution.
        # Use of topk ensures that tokens with really small probabilities aren't chosen.
        # probs here have shape (B, 24), tok_indices is (B, 24)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        
        # model the distribution as multinomial and sample the next token.
        # Sampling ensures that the same token isn't predicted all the time.
        ix = torch.multinomial(topk_probs, 1)

        # gather the corresponding tokens
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        
        # concatenate the predicted tokens into the tensor, so the model can predict another token.
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens) 
    print(f">> {decoded}")

