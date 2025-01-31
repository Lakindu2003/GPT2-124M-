from dataclasses import dataclass
import tiktoken, os, glob, sys, math, time, torch, datetime
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd, numpy as np
from hellaswag import render_example, iterate_examples
# import inspect

class CasualSelfAttention(nn.Module):
        
        def __init__(self, config):
                super().__init__()
                assert config.n_embd % config.n_head == 0

                #attention is basically a couple of matrix multiplication operations with learnable weights
                #this is equivalent to carry out many many linear regressions which is equivalent to using a linear layer.
                self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
                self.c_proj = nn.Linear(config.n_embd, config.n_embd)
                self.c_proj.NANOGPT_SCALE_INIT = 1 # flag to initialise residuals
               
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
             
        # ---------------- Flash attention code removal -------------------------
                # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #standard attention formula
                # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) #masking
                # att = F.softmax(att, dim=-1) #standard attention formula
                # y = att @ v #standard attention formula
        # -----------------------------------------------------------------------
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash-attention (7.6x faster than regular attention)
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
           self.c_proj.NANOGPT_SCALE_INIT = 1 # flag to initialise residuals

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
                x = x + self.attn(self.ln_1(x)) # += breaks .backward()
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

                # weight sharing scheme
                self.transformer.wte.weight = self.lm_head.weight # make both layers point to the same weight matrix
                                                                  # This reduces the number of parameters by around 30%

                self.apply(self._init_weights) # iterates over each module and initialises its weights

        # initialize weights
        def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                        std = 0.02
                        if hasattr(module, 'NANOGPT_SCALE_INIT'):
                                # since we use 'add', the variance increases as we go deeper into the model.
                                # In order to circumvent this, we divide the variance by std by the sqrt(number_of_layers).
                                std *= (2*self.config.n_layer)**-0.5
                        nn.init.normal_(module.weight, mean=0.0, std=std)
                        if module.bias is not None:
                               nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                        nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # no need to initialize layernorm as PyTorch default is what we want
                       

        # just copy-pasting since there isn't much to it
        def forward(self, idx, targets=None):
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
                
                loss = None
                if targets is not None:
                       # cross entropy can't take in multi-dimensional inputs (B*T). So we flatten the tensors.
                       # logits: (B*T, vocab_size), targets: (B*T)
                       loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) 

                return logits, loss #there aren't any probabilities

        def configure_optimizers(self, weight_decay, learning_rate, device):
                # start with all of the candidate parameters (that require grad)
                param_dict = {pn: p for pn, p in self.named_parameters()}
                param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

                # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
                # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
                decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
                nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
                optim_groups = [
                    {'params': decay_params, 'weight_decay': weight_decay},
                    {'params': nodecay_params, 'weight_decay': 0.0}
                ]

                num_decay_params = sum(p.numel() for p in decay_params)
                num_nodecay_params = sum(p.numel() for p in nodecay_params)
                print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
                print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

                # Create AdamW optimizer and use the fused version if it is available
                # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters 
                # use_fused = fused_available and 'cuda' in device
                # print(f"using fused AdamW: {use_fused}")
                print(f"using fused AdamW: {True}")  

                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
                return optimizer

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def load_tokens(filename):
       npt = np.load(filename)
       ptt = torch.tensor(npt, dtype=torch.long)
       return ptt

class DataLoaderLite: 
        def __init__(self, B, T, split):
                self.B = B
                self.T = T
                assert split in {'train', 'val'}

                # get the names of all the data shards
                data_root = "edu_fineweb10B"
                shards = os.listdir(data_root)
                shards = [s for s in shards if split in s]
                shards = sorted(shards)
                shards = [os.path.join(data_root, s) for s in shards]
                self.shards = shards
                assert len(shards) > 0, f"no shards found for split {split}"
                print(f"found {len(shards)} shards for split {split}")
                self.reset()
        
        def reset(self):
                # state, init at shard_0
                self.current_shard = 0
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = self.B * self.T
                

        def next_batch(self):
                B, T = self.B, self.T
                buf = self.tokens[self.current_position:self.current_position+B*T+1] # +1 because buf stores the target as well
                x = buf[:-1].view(B, T) # inputs
                y = buf[1:].view(B, T) # targets
                self.current_position += B*T # B*T: tokens in one mini-batch
                if self.current_position + B*T+1 > len(self.tokens): # indexes to next shard if current shard has no remaining tokens. 
                       self.current_shard = (self.current_shard+1) % len(self.shards) # makes sure that all shards are looped over
                       self.tokens = load_tokens(self.shards[self.current_shard])
                       self.current_position = B*T
                return x, y

# Little Shakespheare dataset
# class DataLoaderLite: 
#         def __init__(self, B, T):
#                 self.B = B
#                 self.T = T
                
#                 # encode the text and load the tokens into memory
#                 with open(r"input.txt", 'r') as f:
#                         text = f.read()
#                 enc = tiktoken.get_encoding('gpt2')
#                 tokens = enc.encode(text)
#                 self.tokens = torch.tensor(tokens)
#                 print(f"loaded {len(self.tokens)} tokens")
#                 print(f"1 epoch = {len(self.tokens)//(B*T)} batches")

#                 self.current_position = 0 # current position being indexed
        
#         def next_batch(self):
#                 B, T = self.B, self.T
#                 buf = self.tokens[self.current_position:self.current_position+B*T+1] # +1 because buf stores the target as well
#                 x = buf[:-1].view(B, T) # inputs
#                 y = buf[1:].view(B, T) # targets
#                 self.current_position += B*T # B*T: tokens in one mini-batch
#                 if self.current_position + B*T+1 > len(self.tokens): 
#                        self.current_position = 0 # resets index if no more tokens left
#                 return x, y
# ----------------------------------------------------------------------------------------------------------------------------------------------------
@torch._dynamo.disable
def generate_and_save_samples(model, enc, device, step, max_length=32, num_return_sequences=4, initial_text="Hello, I'm a language model,"):
    model.eval()
    tokens = enc.encode(initial_text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)
    
    while xgen.size(1) < max_length:
        with torch.no_grad():
            logits, loss = model(xgen)  # (B, T, vocab_size)       
            
            # take the logits corresponding to the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # convert the logits to probabilities
            probs = F.softmax(logits, dim=-1)       
            
            # get top k tokens
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            
            # sample next token
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)   
            
            # gather the corresponding tokens
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            
            # concatenate the predicted tokens
            xgen = torch.cat((xgen, xcol), dim=1) 

    # generate and save texts
    generated_texts = []
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens) 
        generated_texts.append({'Step': step, 'Sequence_Number': i, 'Generated_Text': decoded})
        print(f"sample {i}: {decoded}") 
    print("")
    
    df = pd.DataFrame(generated_texts)
    df.to_csv('outputs.csv', mode='a', index=False, header=False)

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

@torch._dynamo.disable
def evaluate_hellaswag(model, device, device_type, log_file, step):
    num_correct_norm = 0
    num_total = 0
    
    for i, example in enumerate(iterate_examples("val")):
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    
    acc_norm = num_correct_norm / num_total
    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
    
    with open(log_file, "a") as f:
        f.write(f"{step} hella {acc_norm:.4f}\n")
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Useful functions for loading trained model
def get_latest_checkpoint(log_dir):
    checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not checkpoints:
        return None
    
    # Extract step numbers from filenames and find max
    steps = []
    for checkpoint in checkpoints:
        try:
            # Extract number between 'model_' and '.pt'
            step = int(checkpoint.split('model_')[-1].split('.pt')[0])
            steps.append((step, checkpoint))
        except:
            continue
    
    if not steps:
        return None
    
    # Return the path with highest step number
    _, latest_checkpoint = max(steps)
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def load_resume_checkpoint(checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
#     print(checkpoint)

    # Create and load model
    fixed_state_dict = {}
    model = GPT(checkpoint['config'])
    for k, v in checkpoint['model'].items():
        if k.startswith('_orig_mod.'):
            fixed_state_dict[k.replace('_orig_mod.', '')] = v # Remove '_orig_mod.' prefix (10 characters)
        else:
            fixed_state_dict[k] = v
    
    model.load_state_dict(fixed_state_dict)
    model = model.to(device)
    model = torch.compile(model)  # recompile for speed
    
    # Create and load optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=checkpoint['learning_rate'],
        device=device
    )
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Create and restore dataloaders
    train_loader = DataLoaderLite(
        B=checkpoint['train_loader_state']['B'],
        T=checkpoint['train_loader_state']['T'],
        split='train'
    )
    train_loader.current_shard = checkpoint['train_loader_state']['current_shard']
    train_loader.current_position = checkpoint['train_loader_state']['current_position']
    train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])
    
    val_loader = DataLoaderLite(
        B=checkpoint['val_loader_state']['B'],
        T=checkpoint['val_loader_state']['T'],
        split='val'
    )
    val_loader.current_shard = checkpoint['val_loader_state']['current_shard']
    val_loader.current_position = checkpoint['val_loader_state']['current_position']
    val_loader.tokens = load_tokens(val_loader.shards[val_loader.current_shard])
    
    # Restore random states
    torch.set_rng_state(torch.ByteTensor(checkpoint['torch_rng_state'].cpu()))
    torch.cuda.set_rng_state(torch.ByteTensor(checkpoint['cuda_rng_state'].cpu()))
    cuda_states = []
    for state in checkpoint['cuda_rng_state_all']:
        byte_state = torch.ByteTensor(state.cpu())
        cuda_states.append(byte_state)
    torch.cuda.set_rng_state_all(cuda_states)

    return (
        model,
        optimizer,
        checkpoint['step'],
        checkpoint['val_loss'],
        train_loader,
        val_loader
    )
# --------------------------------------------------------------------------------------------------------------
#initialisation

if __name__ == '__main__':
        device = "cuda" 
        seed = 1337

        torch.cuda.set_per_process_memory_fraction(0.99) # prevents Python from using 100% VRAM and crashing 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.set_float32_matmul_precision('high') # set PyTorch precision to TF32
        torch.set_float32_matmul_precision('high')
        # torch.autograd.set_detect_anomaly(True)

        total_batch_size = 2**19 # ~0.5M number of tokens
        B = 2**3
        T = 1024
        # default: B=16, T=1024
        assert total_batch_size%(B*T) == 0,  "Make sure total_batch_size is divisible by B*T"
        grad_accum_steps = total_batch_size//(B*T)
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
        train_loader = DataLoaderLite(B=B, T=T, split="train") # (B, T) = (num_return_sequences, max_length)
        val_loader = DataLoaderLite(B=B, T=T, split="val")

        model = GPT(GPTConfig(vocab_size=50304, block_size=1024)) # changing hyperparameters to nice numbers
        model.to(device)

        model = torch.compile(model) # compiles the code, making it much faster. Requires linux.

        max_lr = 18e-4 # default: 6e-4
        min_lr = 0.1*max_lr
        warmup_steps = 100 # default (GPT 3's warmup steps): 715, 375M/2**19. But this is too mild.
        max_steps = 19073 # total_num_tokens_in_all_shards//total_batch_size ~ 10B/2**19
        def get_lr(it):
                if it < warmup_steps:
                        return max_lr*(it+1)/warmup_steps
                if it > max_steps:
                        return min_lr
                decay_ratio = (it - warmup_steps)/(max_steps-warmup_steps)
                assert 0 <= decay_ratio <= 1
                coeff = 0.5*(1.0 + math.cos(math.pi*decay_ratio))
                return min_lr + coeff*(max_lr - min_lr)

        # optimisation loop
        # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8)
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
        enc = tiktoken.get_encoding('gpt2')

        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log.txt")
        with open(log_file, "a") as f:
               pass
# -------------------------------------------------------------------------------------------------------------------
        # loading trained model
        latest_checkpoint = get_latest_checkpoint(log_dir)
        model, optimizer, start_step, val_loss, train_loader, val_loader = load_resume_checkpoint(latest_checkpoint, device="cuda")
        val_freq = 100
        # start_step = 0
        for step in range(start_step+1, max_steps):
                t0 = time.time()
                last_step = (step == max_steps - 1)

                # validation loop (once every 100 steps) ----------------------------------------------------
                if step % val_freq == 0 or last_step:
                        model.eval()
                        # val_loader.reset()
                        with torch.no_grad():
                            val_loss_accum = 0.0
                            val_loss_steps = 20
                            for _ in range(val_loss_steps):
                                x, y = val_loader.next_batch()
                                x, y = x.to(device), y.to(device)
                                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                    logits, loss = model(x, y)
                                loss = loss / val_loss_steps
                                val_loss_accum += loss.detach()

                        print(f"validation loss: {val_loss_accum.item():.4f}")
                        with open(log_file, "a") as f:
                                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

                        if step > 0 and (step % val_freq == 0 or last_step):
                                # Save complete training state
                                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                                checkpoint = {
                                    # Model and training state
                                    'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),

                                    # Configuration
                                    'config': model.config,
                                    'step': step,
                                    'val_loss': val_loss_accum.item(),

                                    # Dataloader states
                                    'train_loader_state': {
                                        'current_shard': train_loader.current_shard,
                                        'current_position': train_loader.current_position,
                                        'B': train_loader.B,
                                        'T': train_loader.T,
                                },
                                    'val_loader_state': {
                                        'current_shard': val_loader.current_shard,
                                        'current_position': val_loader.current_position,
                                        'B': val_loader.B,
                                        'T': val_loader.T,
                                },

                                    # Random states for reproducibility
                                    'torch_rng_state': torch.get_rng_state(),
                                    'cuda_rng_state': torch.cuda.get_rng_state(),
                                    'cuda_rng_state_all': torch.cuda.get_rng_state_all(),

                                    # Training parameters
                                    'learning_rate': optimizer.param_groups[0]['lr'],

                                    # Date and version info for tracking
                                    'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                                }
                                torch.save(checkpoint, checkpoint_path)

                # HellaSwag loop (once every 100 step) ---------------------------------------------------------
                if (step % val_freq == 0 or last_step):
                        evaluate_hellaswag(model, device, device, log_file, step)



                # sampling loop (once every 100 steps) ---------------------------------------------------------
                if (step > 0 and step % val_freq == 0) or last_step:
                        num_return_sequences = 4     
                        max_length = 32
                        generate_and_save_samples(model, enc, device, step, max_length=max_length, num_return_sequences=num_return_sequences, initial_text="Hello, I'm a language model,")




                # training loop -----------------------------------------------------------------------------------
                model.train()
                loss_accum = 0.0
                optimizer.zero_grad() # Initialize gradients to 0 at every iteration as `optimizer.backward()` accumulates the gradients. 
                for micro_step in range(grad_accum_steps):
                        x, y = train_loader.next_batch()
                        x, y = (x.to(device), y.to(device))
                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                logits, loss = model(x, y)
                        loss = loss/grad_accum_steps    # The gradients are normalised since we want the MEAN and not the SUM
                                                        # We normalize within the inner loop to reduce the magnitude and reduce the likelihood of precision issues .
                        loss_accum += loss.detach() 
                        loss.backward() # calculates the gradients via backpropagation
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # updating the learning rate
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                optimizer.step() # parameter update 
                torch.cuda.synchronize()
                t1 = time.time()
                dt = (t1-t0)*1000
                tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
                tokens_per_sec = tokens_processed/(t1-t0)
                print(f"step {step:4d} | loss: {loss_accum:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}") # ()` converts the single valued tensor stored in the GPU to a float stored in the CPU. 
                with open(log_file, "a") as f:
                        f.write(f"{step} train {loss_accum.item():.6f}\n")

        sys.exit(0)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# # generate! right now x is (B, T) where B = 5, T = 8
# # let the model do it's thing!
# max_length = T
# num_return_sequences = B
# while x.size(1) < max_length:
    
#     # forward the model to get the logits
#     # torch.no_grad() indicates to PyTorch that no backpropagation takes place. 
#     # This reduces the amount of space used by PyTorch in caching the intermediate tensors needed for backpropagation. 
#     with torch.no_grad():
#         logits, _ = model(x) # (B, T, vocab_size)

#         # take the logits corresponding to the last position (poistion closest to token being predicted)
#         logits = logits[:, -1, :] # (B, vocab_size)
        
#         # convert the logits to probabilities
#         probs = F.softmax(logits, dim=-1)

#         # `torch.topk`, where k=50, is used to filter the 50 tokens with the highest probability 
#         #and renormalize the probability distribution.
#         # Use of topk ensures that tokens with really small probabilities aren't chosen.
#         # probs here have shape (B, 24), tok_indices is (B, 24)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        
#         # model the distribution as multinomial and sample the next token.
#         # Sampling ensures that the same token isn't predicted all the time.
#         ix = torch.multinomial(topk_probs, 1)

#         # gather the corresponding tokens
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        
#         # concatenate the predicted tokens into the tensor, so the model can predict another token.
#         x = torch.cat((x, xcol), dim=1)

# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens) 
#     print(f">> {decoded}")

