import os, json, glob
import numpy as np, torch

SOURCES = ["fineweb", "starcoder", "cosmopedia"]

class BlockData:
    """Fixed-length token blocks cut from the shards listed in manifest.json.
    Each item is (x, y) with y = x shifted by one token, both `block` tokens long.
      train:      order comes from a saved file -> identical batch order in every session
      validation: file order, no randomness      -> identical blocks at every evaluation"""

    def __init__(self, root, split="train", block=2048, sources=None, order_path=None):
        mp = os.path.join(root, "manifest.json")
        assert os.path.exists(mp), f"manifest.json not found in {root}"
        m = json.load(open(mp))
        self.block, self.files, self.src, parts = block, [], [], []
        for src, info in m["sources"].items():                 # only manifest sources (a backup folder is ignored)
            if sources and src not in sources: continue
            entries = info.get("shards", {}).get(split)
            if not entries:                                    # fall back to the files on disk
                entries = [dict(file=os.path.basename(p), tokens=int(np.load(p, mmap_mode="r").shape[0]))
                           for p in sorted(glob.glob(os.path.join(root, src, split, "shard_*.npy")))]
            for e in entries:
                fi = len(self.files)
                self.files.append(os.path.join(root, src, split, e["file"]))
                self.src.append(src)
                n = (e["tokens"] - 1) // block                 # need block+1 tokens per block (x and y)
                parts.append(np.stack([np.full(n, fi), np.arange(n)], 1))
        idx = np.concatenate(parts).astype(np.int64)           # rows: (file index, block number in that file)

        if order_path is not None:                             # saved order: must be a permutation of exactly these blocks
            saved = np.load(order_path).astype(np.int64)
            key = lambda a: np.sort(a[:, 0] * 1_000_000 + a[:, 1])
            assert saved.shape == idx.shape and np.array_equal(key(saved), key(idx)), \
                "order file does not match the manifest / block size"
            idx = saved
        self.index = idx
        self._mm = {}                                          # memory-mapped shards, opened on first use

    def __len__(self):
        return len(self.index)

    def get(self, i):
        fi, b = self.index[i]
        if fi not in self._mm:
            self._mm[fi] = np.load(self.files[fi], mmap_mode="r")
        s = int(b) * self.block
        chunk = torch.from_numpy(self._mm[fi][s:s + self.block + 1].astype(np.int64))
        assert chunk.numel() == self.block + 1, f"short block in {self.files[fi]}"
        return chunk[:-1], chunk[1:]

    def batch(self, i, bs):
        """micro-batch number i = blocks [i*bs, (i+1)*bs) of the order"""
        xs, ys = zip(*(self.get(i * bs + j) for j in range(bs)))
        return torch.stack(xs), torch.stack(ys)

def make_datasets(root, block, order_path, with_val=True):
    train = BlockData(root, "train", block, order_path=order_path)
    val = {s: BlockData(root, "validation", block, sources=[s]) for s in SOURCES} if with_val else {}
    return train, val