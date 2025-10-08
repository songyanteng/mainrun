import utils
import math, random, time, os
from dataclasses import dataclass
import json
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog
import numpy as np
import itertools
import random as pyrand

# ----------------------------
# Hyperparameters
# ----------------------------
@dataclass
class Hyperparameters:
    # data / model
    block_size: int = 128
    batch_size: int = 64                  
    accumulation_steps: int = 4           
    vocab_size: int = 16_000
    n_layer: int = 6                      
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.1

    # optimisation
    lr: float = 6e-3                      
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.95)
    warmup_ratio: float = 0.10            
    max_grad_norm: float = 1.0

    # logging / eval
    evals_per_epoch: int = 3

    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"

# ---------- Sweep Constants and Helpers ----------
CONFIG_KEYS = [
    "dropout","weight_decay","lr","vocab_size",
    "n_layer","d_model","warmup_ratio",
    "batch_size","accumulation_steps"
]

def config_key(cfg: dict) -> tuple:
    # Stable unique key for a config row
    return tuple((k, cfg.get(k, None)) for k in CONFIG_KEYS)

def load_existing_result_keys(results_file: str) -> set[tuple]:
    keys = set()
    if os.path.exists(results_file):
        with open(results_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                norm = {
                    "dropout": float(row["dropout"]) if row["dropout"] else None,
                    "weight_decay": float(row["weight_decay"]) if row["weight_decay"] else None,
                    "lr": float(row["lr"]) if "lr" in row and row["lr"] else None,
                    "vocab_size": int(row["vocab_size"]) if row["vocab_size"] else None,
                    "n_layer": int(row["n_layer"]) if row["n_layer"] else None,
                    "d_model": int(row["d_model"]) if row["d_model"] else None,
                    "warmup_ratio": float(row["warmup_ratio"]) if row["warmup_ratio"] else None,
                    "batch_size": int(row["batch_size"]) if row["batch_size"] else None,
                    "accumulation_steps": int(row["accumulation_steps"]) if row["accumulation_steps"] else None,
                }
                keys.add(config_key(norm))
    return keys

def run_id_from_cfg(cfg: dict) -> str:
    # Short tag for filenames
    base = "_".join(f"{k}-{cfg[k]}" for k in CONFIG_KEYS if k in cfg)
    return base.replace("/", "_")

# ----------------------------
# Logging
# ----------------------------
def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = open(log_file, 'w')
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()
            
        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()
            
            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)
    
    return DualLogger(file_handler)

logger = None

# ----------------------------
# Data
# ----------------------------
def get_titles(num_titles: int, seed: int, val_frac: float) -> str:
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]

def get_batch_random(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    total_len = split_ids.size(0) - (block_size + 1)
    starts = torch.randint(0, total_len, (batch_size,), device=split_ids.device)
    x = torch.stack([split_ids[s:s+block_size]     for s in starts]).to(device)
    y = torch.stack([split_ids[s+1:s+block_size+1] for s in starts]).to(device)
    return x, y

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

# ----------------------------
# Tokeniser
# ----------------------------
def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token]
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer

class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer, eos: str):
        self.tk = tokenizer
        self.eos = eos # explicitly pass eos token
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}
        
    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()

# ----------------------------
# Model
# ----------------------------
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss

# ----------------------------
# Tokeniser Caching
# ----------------------------
def prepare_tokenizer(titles: list[str], vocab_size: int, save_dir: str = "./logs"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    tok_path = os.path.join(save_dir, f"tokenizer_{vocab_size}.json")
    if not os.path.exists(tok_path):
        tokenizer = train_tokenizer(titles, vocab_size)
        tokenizer.save(tok_path)
        print(f"[Tokenizer] Trained vocab={vocab_size} → {tok_path}")
    else:
        tokenizer = Tokenizer.from_file(tok_path)
        print(f"[Tokenizer] Loaded vocab={vocab_size} from {tok_path}")
    return tokenizer

def prepare_token_ids(train_titles, val_titles, tokenizer: Tokenizer, eos_token: str, cache_dir: str = "./logs"):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    vs = tokenizer.get_vocab_size()
    train_path = os.path.join(cache_dir, f"ids_train_vs{vs}.pt")
    val_path   = os.path.join(cache_dir, f"ids_val_vs{vs}.pt")

    if os.path.exists(train_path) and os.path.exists(val_path):
        train_ids = torch.load(train_path)
        val_ids   = torch.load(val_path)
        print(f"[IDs] Loaded token IDs from cache for vocab={vs}")
    else:
        tok = BPETokenizer(tokenizer, eos=eos_token)
        train_text = eos_token.join(train_titles) + eos_token
        val_text   = eos_token.join(val_titles) + eos_token
        train_ids  = torch.tensor(tok.encode(train_text), dtype=torch.long)
        val_ids    = torch.tensor(tok.encode(val_text), dtype=torch.long)
        torch.save(train_ids, train_path)
        torch.save(val_ids, val_path)
        print(f"[IDs] Saved token IDs cache for {train_path}, {val_path}")
    return train_ids, val_ids, tokenizer.get_vocab_size()

# ----------------------------
# Train
# ----------------------------
def main(overrides: dict | None = None):
    args = Hyperparameters()
    if overrides:
        for k, v in overrides.items():
            setattr(args, k, v)

    # Seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global logger
    logger = configure_logging(args.log_file)
    
    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log("device_info", device=str(device), cuda=torch.cuda.is_available(), cudnn=torch.backends.cudnn.version())

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)

    # Tokenizer + id caching
    eos_token = "<eos>"
    tokenizer = prepare_tokenizer(train_titles + val_titles, args.vocab_size)

    train_text = eos_token.join(train_titles) + eos_token
    val_text   = eos_token.join(val_titles) + eos_token

    train_ids, val_ids, vocab_size = prepare_token_ids(train_titles, val_titles, tokenizer, eos_token=eos_token)

    batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = max(1, batches // args.evals_per_epoch)
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=vocab_size)

    cfg = GPTConfig(
        vocab_size = vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            logger.log("compile_info", compiled=True)
        except Exception as e:
            logger.log("compile_info", compiled=False, error=str(e))

    use_fused = (device.type == "cuda")
    
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=args.betas, fused=use_fused
    )

    warmup_steps = int(args.warmup_ratio * max_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    logger.log("scheduler_info", warmup_steps=warmup_steps, max_steps=max_steps, base_lr=float(args.lr))

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    # ----- Training Loop -----
    step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}")
        opt.zero_grad(set_to_none=True)
        for _ in pbar:
            step += 1
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                xb, yb = get_batch_random(train_ids, args.block_size, args.batch_size, device)
                _, loss = model(xb, yb)
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if step % args.accumulation_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()

            elapsed = time.time() - t0
            logger.log("training_step",
                      step=step,
                      max_steps=max_steps,
                      loss=loss.item(),
                      elapsed_time=elapsed,
                      prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                logger.log("validation_step",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          elapsed_time=elapsed)

def run_experiments(configs: list[dict], results_file: str):
    Path("./logs").mkdir(parents=True, exist_ok=True)

    existed = load_existing_result_keys(results_file)
    new_file = not os.path.exists(results_file)

    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "dropout","weight_decay","lr","vocab_size",
                "n_layer","d_model","warmup_ratio",
                "batch_size","accumulation_steps",
                "val_loss","log_file","run_time_sec"
            ])

        for i, cfg in enumerate(configs, 1):
            # Default missing keys from Hyperparameters()
            defaulted = {k: getattr(Hyperparameters(), k) if hasattr(Hyperparameters(), k) else None
                         for k in CONFIG_KEYS}
            defaulted.update(cfg)
            ck = config_key(defaulted)
            if ck in existed:
                print(f"[SKIP] Already have results for config #{i}: {defaulted}")
                continue

            print(f"\n=== Experiment {i}/{len(configs)} ===")
            print(defaulted)

            suffix = run_id_from_cfg(defaulted)
            run_log = f"./logs/run_{suffix}.log"
            defaulted = defaulted.copy()
            defaulted["log_file"] = run_log

            # Run with walltime
            t0 = time.time()
            main(defaulted)
            runtime = time.time() - t0

            # Parse last val
            with open(run_log, "r") as lf:
                lines = lf.readlines()
                val_lines = [ln for ln in lines if "\"event\": \"validation_step\"" in ln]
                last_val = json.loads(val_lines[-1])["loss"] if val_lines else None

            # Write CSV row
            writer.writerow([
                defaulted.get("dropout"),
                defaulted.get("weight_decay"),
                defaulted.get("lr"),
                defaulted.get("vocab_size"),
                defaulted.get("n_layer"),
                defaulted.get("d_model"),
                defaulted.get("warmup_ratio"),
                defaulted.get("batch_size"),
                defaulted.get("accumulation_steps"),
                last_val,
                run_log,
                round(runtime, 2)
            ])
            f.flush()

def plan_broad_sample(n: int = 80, seed: int = 1337):
    # Random sample across full grid to see trends
    base = dict(
        vocab_size=16000,
        batch_size=64,
        accumulation_steps=4,
        warmup_ratio=0.10,
    )
    DROPOUT = [0.00, 0.05, 0.10]
    WEIGHT_DECAY = [0.00, 0.05, 0.10]
    LR = [1e-4, 3e-4, 6e-4, 1e-3]
    DEPTH = [5, 6, 7]
    WIDTH = [512, 640, 704, 768]

    # Full grid
    all_cfgs = []
    for d, wd, lr, nl, dm in itertools.product(DROPOUT, WEIGHT_DECAY, LR, DEPTH, WIDTH):
        c = dict(dropout=d, weight_decay=wd, lr=lr, n_layer=nl, d_model=dm)
        c.update(base)
        all_cfgs.append(c)

    pyrand.seed(seed)
    pyrand.shuffle(all_cfgs)

    # Initial baseline
    legacy_baseline = dict(
        dropout=0.10, weight_decay=0.0, lr=6e-3,
        n_layer=6, d_model=512, **base
    )

    # Updated baseline
    baseline = dict(dropout=0.05, weight_decay=0.05, lr=3e-4, n_layer=6, d_model=640, **base)

    # Include both baselines
    unique = [baseline, legacy_baseline]
    seen = {config_key(baseline), config_key(legacy_baseline)}

    # Fill with random sample from full grid until n configs reached
    for c in all_cfgs:
        ck = config_key(c)
        if ck not in seen:
            unique.append(c)
            seen.add(ck)
        if len(unique) >= n:
            break

    return unique

def plan_tier1():
    base = dict(vocab_size=16000, batch_size=64, accumulation_steps=4, warmup_ratio=0.10)
    DROPOUT = [0.00, 0.05, 0.10]
    WEIGHT_DECAY = [0.00, 0.02, 0.05, 0.10]
    LR = [1e-4, 2e-4, 3e-4, 6e-4, 1e-3]
    cfgs = []
    for lr in LR:
        for wd in WEIGHT_DECAY:
            for d in DROPOUT:
                cfgs.append(dict(lr=lr, weight_decay=wd, dropout=d, n_layer=6, d_model=640, **base))
    return cfgs

def plan_tier2(best_opt: dict):
    # best_opt must contain lr, weight_decay, dropout
    base = dict(vocab_size=16000, batch_size=64, accumulation_steps=4, warmup_ratio=0.10)
    DEPTH = [5, 6, 7]
    WIDTH = [512, 640, 704, 768]
    cfgs = []
    for nl in DEPTH:
        for dm in WIDTH:
            cfgs.append(dict(n_layer=nl, d_model=dm, **best_opt, **base))
    return cfgs

def plan_tier3(best_opt: dict, best_cap: dict):
    base = dict(vocab_size=16000, batch_size=64, accumulation_steps=4, warmup_ratio=0.10)
    lr0 = float(best_opt["lr"])

    # Default local band (includes lower and higher)
    band = [0.5, 0.75, 1.0, 1.25, 1.5]

    # If Tier-1 winner was at the top end, push higher to check for larger effects
    if lr0 >= 1e-3 * 0.95:
        band += [2.0, 3.0]

    fine_lrs = sorted({lr0 * f for f in band})
    cfgs = []
    for lr in fine_lrs:
        cfgs.append(dict(
            lr=lr,
            **best_cap,
            weight_decay=best_opt["weight_decay"],
            dropout=best_opt["dropout"],
            **base
        ))
    return cfgs

def pick_best_opt(csv_path: str) -> dict:
    # Returns {"lr":..., "weight_decay":..., "dropout":...} with the minimum val_loss
    best = None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                val = float(row["val_loss"])
            except (KeyError, ValueError, TypeError):
                continue
            cand = dict(
                lr=float(row["lr"]),
                weight_decay=float(row["weight_decay"]),
                dropout=float(row["dropout"]),
            )
            if best is None or val < best["val_loss"]:
                best = {"val_loss": val, **cand}
    if best is None:
        raise RuntimeError(f"No valid rows found in {csv_path}")
    return {k: best[k] for k in ["lr", "weight_decay", "dropout"]}

def pick_best_capacity(csv_path: str, fixed_opt: dict) -> dict:
    # Filter by the chosen optimizer triple and pick the best (n_layer, d_model)
    best = None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                if (float(row["lr"]) != float(fixed_opt["lr"]) or
                    float(row["weight_decay"]) != float(fixed_opt["weight_decay"]) or
                    float(row["dropout"]) != float(fixed_opt["dropout"])):
                    continue
                val = float(row["val_loss"])
                cand = dict(n_layer=int(row["n_layer"]), d_model=int(row["d_model"]))
            except (KeyError, ValueError, TypeError):
                continue
            if best is None or val < best["val_loss"]:
                best = {"val_loss": val, **cand}
    if best is None:
        raise RuntimeError(f"No matching rows for fixed optimizer in {csv_path}")
    return {k: best[k] for k in ["n_layer", "d_model"]}

def save_json(obj: dict, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    try:
        stage = os.environ.get("SWEEP_STAGE", "broad")  # "broad", "tier1", "tier2", or "tier3"

        if stage == "tier1":
            cfgs = plan_tier1()
            results_path = "./logs/sweep_tier1.csv"
            run_experiments(cfgs, results_file=results_path)
            best_opt = pick_best_opt(results_path)
            save_json(best_opt, "./logs/best_tier1_optimizer.json")
            print("[Tier1] Best optimizer:", best_opt)

        elif stage == "tier2":
            best_opt = load_json("./logs/best_tier1_optimizer.json")
            cfgs = plan_tier2(best_opt)
            results_path = "./logs/sweep_tier2.csv"
            run_experiments(cfgs, results_file=results_path)
            best_cap = pick_best_capacity(results_path, best_opt)
            save_json(best_cap, "./logs/best_tier2_capacity.json")
            print("[Tier2] Best capacity:", best_cap)

        elif stage == "tier3":
            best_opt = load_json("./logs/best_tier1_optimizer.json")
            best_cap = load_json("./logs/best_tier2_capacity.json")
            cfgs = plan_tier3(best_opt, best_cap)
            results_path = "./logs/sweep_tier3.csv"
            run_experiments(cfgs, results_file=results_path)
            print("[Tier3] Fine LR sweep complete.")

        elif stage == "broad":
            cfgs = plan_broad_sample(n=80)
            results_path = "./logs/sweep_broad.csv"
            run_experiments(cfgs, results_file=results_path)
            print("[Broad] Sweep complete. Check", results_path)

    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()