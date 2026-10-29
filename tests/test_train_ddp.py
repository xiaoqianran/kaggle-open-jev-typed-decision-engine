"""CPU-only regression tests for 02_train.py's single-process and DDP paths.

Run: python -m unittest discover -s tests -p 'test_train_ddp.py'
No Hugging Face download or GPU is required.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import torch


TRAIN = Path(__file__).resolve().parents[1] / "02_train.py"
STUBS = {
    "transformers.py": """
        from torch.optim.lr_scheduler import LambdaLR
        def get_cosine_schedule_with_warmup(opt, warmup, total):
            return LambdaLR(opt, lambda step: 1.0)
        get_linear_schedule_with_warmup = get_cosine_schedule_with_warmup
    """,
    "td_data.py": """
        import torch
        from torch.utils.data import Dataset
        def load_split(*args, **kwargs):
            return [{'id': i, 'x': float(i % 5), 'label': i % 2} for i in range(17)]
        def val_split(rows):
            return rows[:13], rows[13:]
        class TypedDecisions(Dataset):
            def __init__(self, rows, tok, max_len, qid, lid):
                self.rows = rows
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, index):
                row = self.rows[index]
                return {
                    'x': torch.tensor([row['x']]),
                    'target': torch.nn.functional.one_hot(
                        torch.tensor(row['label']), num_classes=2).float(),
                    'qmask': torch.ones(1, dtype=torch.bool),
                }
        def collate(batch, pad_id):
            return {k: torch.stack([row[k] for row in batch]) for k in batch[0]}
    """,
    "model.py": """
        import torch
        from torch import nn
        DEFAULT_ENCODER = 'fake'
        class JevLite(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = nn.Linear(1, 3)
                self.score = nn.Linear(3, 2)
                self.qid = self.lid = 0
                self.tok = type('Tok', (), {'pad_token_id': 0})()
            def forward(self, batch, apply_temperature=False):
                probs = self.score(self.encoder(batch['x'])).softmax(-1)
                return probs, probs
        def decision_loss(probs, batch, weight):
            loss = -(batch['target'] * probs.clamp_min(1e-8).log()).sum(-1).mean()
            return loss, loss.detach(), loss.detach()
        def argmax_per_question(probs, batch):
            pred = probs.argmax(-1).unsqueeze(1)
            return pred, torch.ones_like(pred, dtype=torch.float)
    """,
}


class TrainDDPTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stubs = self.root / "stubs"
        self.stubs.mkdir()
        for name, content in STUBS.items():
            (self.stubs / name).write_text(textwrap.dedent(content), encoding="utf-8")

    def run_training(self, *, distributed, epochs=1, lr="0.1", patience="0"):
        run_dir = self.root / ("ddp" if distributed else "single")
        run_dir.mkdir(exist_ok=True)
        cmd = [sys.executable]
        if distributed:
            cmd += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
        cmd += [str(TRAIN), "--epochs", str(epochs), "--patience", patience,
                "--bs", "64", "--accum", "3", "--lr", lr, "--head-lr", lr,
                "--out", "checkpoint.pt"]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.stubs) + os.pathsep + env.get("PYTHONPATH", "")
        env["OMP_NUM_THREADS"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run(cmd, cwd=run_dir, env=env,
                                   capture_output=True, text=True, timeout=90)
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + "\n" + completed.stderr)
        checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu",
                                weights_only=True)
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        self.assertTrue(checkpoint["state_dict"])
        self.assertTrue(all(not key.startswith("module.")
                            for key in checkpoint["state_dict"]))
        return checkpoint, history

    def test_single_process_and_ddp_update_short_final_window(self):
        torch.manual_seed(42)
        initial_encoder = torch.nn.Linear(1, 3).weight.detach().clone()
        for distributed in (False, True):
            with self.subTest(distributed=distributed):
                checkpoint, history = self.run_training(distributed=distributed)
                self.assertEqual(len(history), 1)
                # --bs=64 > data available; no full --accum=3 window occurs.
                # An optimizer update must still happen at epoch end.
                self.assertFalse(torch.equal(checkpoint["state_dict"]["encoder.weight"],
                                             initial_encoder))

    def test_ddp_early_stop_is_synchronized(self):
        # Zero learning rate guarantees equal validation accuracy each epoch.
        _, history = self.run_training(distributed=True, epochs=4,
                                       lr="0", patience="1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["val_acc"], history[1]["val_acc"])


if __name__ == "__main__":
    unittest.main()
