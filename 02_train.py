"""Step 2 - fine-tune JevLite on typed-decisions (single GPU or torchrun DDP).

Single GPU: python 02_train.py --epochs 20
Two GPUs:   torchrun --standalone --nproc_per_node=2 02_train.py --epochs 20 --bs 4 --accum 2

Trains against the consensus distribution (soft CE + Brier), not the argmax.
"""
import argparse
from contextlib import nullcontext
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

from td_data import load_split, TypedDecisions, collate, val_split
from model import JevLite, decision_loss, argmax_per_question, DEFAULT_ENCODER


def setup_device():
    """torchrun provides LOCAL_RANK/WORLD_SIZE; plain python stays single-process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return device, rank, world_size, distributed


def accuracy(model, dl, device, temp=False):
    model.eval()
    hit = tot = 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, probs = model(batch, apply_temperature=temp)
            pred, _ = argmax_per_question(probs, batch)
            gold, _ = argmax_per_question(batch["target"], batch)
            mask = batch["qmask"]
            hit += ((pred == gold) & mask).sum().item()
            tot += mask.sum().item()
    return hit / max(tot, 1)


def train(args, device, rank, world_size, distributed):
    main = rank == 0
    torch.manual_seed(42 + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42 + rank)

    base_model = JevLite(args.encoder).to(device)
    tok = base_model.tok

    # Fetch once, then send identical rows to both ranks. Avoid duplicate
    # Hugging Face network requests and preserve the deterministic val split.
    rows = [load_split("train", args.config, limit=args.limit) if main else None]
    if distributed:
        dist.broadcast_object_list(rows, src=0)
    tr_rows, va_rows = val_split(rows[0])
    del rows

    def make_dataset(samples):
        return TypedDecisions(samples, tok, args.max_len,
                              base_model.qid, base_model.lid)

    def collate_batch(batch):
        return collate(batch, tok.pad_token_id)

    train_ds = make_dataset(tr_rows)
    sampler = (DistributedSampler(train_ds, num_replicas=world_size,
                                  rank=rank, shuffle=True, seed=42)
               if distributed else None)
    train_loader = DataLoader(train_ds, batch_size=args.bs,
                              shuffle=sampler is None, sampler=sampler,
                              collate_fn=collate_batch)
    val_loader = (DataLoader(make_dataset(va_rows), batch_size=args.bs,
                             collate_fn=collate_batch) if main else None)

    # Keep parameter names and saved checkpoint keys unchanged.
    head = [p for name, p in base_model.named_parameters()
            if name.startswith("score.")]
    enc = [p for name, p in base_model.named_parameters()
           if not name.startswith("score.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": enc, "lr": args.lr},
                                   {"params": head, "lr": args.head_lr}],
                                  weight_decay=0.01)
    model = (DDP(base_model,
                 device_ids=[device.index] if device.type == "cuda" else None)
             if distributed else base_model)

    updates_per_epoch = math.ceil(len(train_loader) / args.accum)
    total_steps = max(1, updates_per_epoch * args.epochs)
    scheduler_fn = (get_cosine_schedule_with_warmup if args.schedule == "cosine"
                    else get_linear_schedule_with_warmup)
    scheduler = scheduler_fn(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if main:
        print(f"device {device} | GPUs {world_size if distributed else 1} "
              f"| train {len(tr_rows)} | val {len(va_rows)} "
              f"| effective batch {args.bs * args.accum * world_size}", flush=True)

    best, history, stale = -1.0, [], 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        started = time.time()
        every = max(1, len(train_loader) // 10)

        for i, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            # The last accumulation window may contain fewer than --accum batches.
            window_len = min(args.accum, len(train_loader) - (i // args.accum) * args.accum)
            do_step = (i + 1) % args.accum == 0 or i + 1 == len(train_loader)
            sync = model.no_sync() if distributed and not do_step else nullcontext()
            with sync:
                with (torch.autocast("cuda", dtype=torch.float16)
                      if device.type == "cuda" else nullcontext()):
                    _, probs = model(batch)
                    loss, _, _ = decision_loss(probs, batch, args.brier)
                scaler.scale(loss / window_len).backward()

            running += loss.item()
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if main and ((i + 1) % every == 0 or i + 1 == len(train_loader)):
                done = (i + 1) / len(train_loader)
                elapsed = time.time() - started
                print(f"  epoch {epoch + 1}  {done:4.0%}  "
                      f"loss {running / (i + 1):.4f}  "
                      f"{elapsed:.0f}s elapsed, ~{elapsed / done - elapsed:.0f}s left",
                      flush=True)

        # Average the training loss across ranks before writing history.
        epoch_loss = running / max(len(train_loader), 1)
        if distributed:
            mean_loss = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            epoch_loss = mean_loss.item() / world_size

        # Evaluate the complete val set only on rank 0 (using the unwrapped
        # module, so DDP does not require matching forward calls on other ranks).
        stop = False
        if main:
            val_acc = accuracy(base_model, val_loader, device)
            improved = val_acc > best
            print(f"epoch {epoch + 1}/{args.epochs}  "
                  f"loss {epoch_loss:.4f}  "
                  f"val acc {val_acc:.4f}{'  *' if improved else ''}", flush=True)
            history.append({"epoch": epoch + 1,
                            "loss": epoch_loss,
                            "val_acc": val_acc})
            with open("history.json", "w", encoding="utf-8") as file:
                json.dump(history, file, indent=1)
            if improved:
                stale, best = 0, val_acc
                torch.save({"state_dict": base_model.state_dict(),
                            "encoder": args.encoder, "max_len": args.max_len,
                            "val_acc": val_acc}, args.out)
            else:
                stale += 1
            stop = bool(args.patience and stale >= args.patience)
            if stop:
                print(f"\nno gain for {stale} epochs - stopping at epoch {epoch + 1}. "
                      f"The best checkpoint ({best:.4f}) is what was kept.", flush=True)

        if distributed:
            # Keep both processes at the same epoch boundary and stop together.
            control = torch.tensor([int(stop)], dtype=torch.int32, device=device)
            dist.broadcast(control, src=0)
            stop = bool(control.item())
        if stop:
            break

    if main:
        print(f"\nbest val accuracy {best:.4f} -> {args.out}")
        print("Next: python 03_calibrate.py   (fits per-type temperature on val)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6,
                        help="0 disables early stopping")
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--bs", type=int, default=4,
                        help="per-GPU batch size (also on single GPU)")
    parser.add_argument("--accum", type=int, default=4,
                        help="gradient accumulation steps")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--brier", type=float, default=1.0)
    parser.add_argument("--out", default="jevlite.pt")
    parser.add_argument("--config", default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.bs < 1 or args.accum < 1 or args.epochs < 1:
        parser.error("--bs, --accum and --epochs must be positive")

    device, rank, world_size, distributed = setup_device()
    try:
        train(args, device, rank, world_size, distributed)
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
