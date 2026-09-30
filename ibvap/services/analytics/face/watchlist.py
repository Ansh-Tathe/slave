"""
services/analytics/face/watchlist.py
======================================
IBVAP P6 -- Watchlist: in-memory face identity store with FAISS index.

The watchlist maps person names -> one or more ArcFace embeddings.
Matching uses FAISS IndexFlatIP (inner product = cosine similarity for
L2-normalised vectors) for sub-millisecond search even with 10k+ entries.

Persistence
-----------
save(path) / load(path):
  Saves the person_name list and embedding matrix as .npz.
  Reload at startup to avoid re-encoding the watchlist.

Usage
-----
    wl = Watchlist()
    wl.add_person("John Doe",  embed_a)
    wl.add_person("John Doe",  embed_b)   # second photo of same person
    wl.add_person("Jane Smith", embed_c)

    match = wl.search(query_embedding, threshold=0.45)
    if match:
        name, similarity = match
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


EMBED_DIM = 512


class Watchlist:
    """
    In-memory face watchlist backed by a FAISS inner-product index.

    Parameters
    ----------
    embed_dim   : embedding dimension (512 for ArcFace)
    use_gpu     : move FAISS index to GPU (requires faiss-gpu)
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        use_gpu:   bool = False,
    ) -> None:
        self._dim   = embed_dim
        self._names: List[str] = []     # index i -> person name
        self._index = self._build_index(use_gpu)
        self._use_gpu = use_gpu

        logger.info(
            f"Watchlist ready — dim={embed_dim}, "
            f"faiss={'gpu' if use_gpu else 'cpu'}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def add_person(self, name: str, embedding: np.ndarray) -> None:
        """
        Add one embedding for a person.

        Multiple embeddings per person are supported (different photos).
        The name need not be unique across calls.
        """
        emb = embedding.astype(np.float32).reshape(1, self._dim)
        self._index.add(emb)
        self._names.append(name)
        logger.debug(f"Watchlist: added '{name}' (total={len(self._names)})")

    def add_person_from_image(
        self,
        name:     str,
        image:    np.ndarray,
        embedder,                  # FaceEmbedder instance
        detector = None,           # Optional FaceDetector -- to auto-crop the face
    ) -> bool:
        """
        Convenience: detect face in image, embed it, add to watchlist.

        Returns True if a face was found and added, False otherwise.
        """
        import cv2
        if detector is not None:
            faces = detector.detect(image)
            if not faces:
                logger.warning(f"No face detected in image for '{name}'")
                return False
            crop = faces[0].crop
        else:
            crop = cv2.resize(image, (112, 112))

        emb = embedder.embed(crop)
        self.add_person(name, emb)
        return True

    def search(
        self,
        query:     np.ndarray,
        top_k:     int = 1,
        threshold: float = 0.45,
    ) -> Optional[Tuple[str, float]]:
        """
        Search for the closest match in the watchlist.

        Parameters
        ----------
        query     : L2-normalised embedding (512,) float32
        top_k     : number of FAISS candidates to retrieve
        threshold : minimum cosine similarity to accept as a match

        Returns
        -------
        (name, similarity) if best match >= threshold, else None.
        """
        if len(self._names) == 0:
            return None

        q = query.astype(np.float32).reshape(1, self._dim)
        sims, idxs = self._index.search(q, k=min(top_k, len(self._names)))

        best_sim = float(sims[0][0])
        best_idx = int(idxs[0][0])

        if best_idx < 0 or best_sim < threshold:
            return None

        return (self._names[best_idx], best_sim)

    def search_all(
        self,
        query:     np.ndarray,
        top_k:     int = 5,
        threshold: float = 0.45,
    ) -> List[Tuple[str, float]]:
        """
        Return up to top_k matches above threshold, sorted by similarity.
        """
        if len(self._names) == 0:
            return []

        q    = query.astype(np.float32).reshape(1, self._dim)
        k    = min(top_k, len(self._names))
        sims, idxs = self._index.search(q, k=k)

        results: List[Tuple[str, float]] = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            if float(sim) >= threshold:
                results.append((self._names[idx], float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def remove_person(self, name: str) -> int:
        """
        Remove all embeddings for a person by name.
        NOTE: FAISS IndexFlat does not support in-place removal.
        This rebuilds the index from scratch.

        Returns number of removed entries.
        """
        keep_idx = [i for i, n in enumerate(self._names) if n != name]
        removed  = len(self._names) - len(keep_idx)

        if removed == 0:
            return 0

        # Rebuild
        old_embs = self._get_all_embeddings()
        self._names = [self._names[i] for i in keep_idx]
        new_embs = old_embs[keep_idx] if len(keep_idx) else np.zeros((0, self._dim), np.float32)

        self._index = self._build_index(self._use_gpu)
        if len(new_embs):
            self._index.add(new_embs)

        logger.info(f"Watchlist: removed {removed} embedding(s) for '{name}'")
        return removed

    def save(self, path: str | Path) -> None:
        """Save watchlist to .npz file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        embs = self._get_all_embeddings()
        np.savez_compressed(
            str(p),
            embeddings = embs,
            names      = np.array(self._names, dtype=object),
        )
        logger.info(f"Watchlist saved: {p} ({len(self._names)} entries)")

    def load(self, path: str | Path) -> None:
        """Load watchlist from .npz file (merges into existing)."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"Watchlist file not found: {p}")
            return
        data = np.load(str(p), allow_pickle=True)
        embs  = data["embeddings"].astype(np.float32)
        names = data["names"].tolist()

        for name, emb in zip(names, embs):
            self.add_person(str(name), emb)

        logger.success(f"Watchlist loaded: {p} ({len(names)} entries)")

    @property
    def size(self) -> int:
        """Total number of embedding entries in the watchlist."""
        return len(self._names)

    @property
    def persons(self) -> List[str]:
        """Unique person names."""
        return list(dict.fromkeys(self._names))

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_index(self, use_gpu: bool):
        try:
            import faiss
            index = faiss.IndexFlatIP(self._dim)
            if use_gpu:
                try:
                    res   = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.debug("FAISS index on GPU")
                except Exception as exc:
                    logger.warning(f"FAISS GPU failed ({exc}), using CPU")
            return index
        except ImportError:
            logger.warning(
                "faiss not installed -- using NumpyIndex fallback. "
                "Install faiss-cpu or faiss-gpu for production."
            )
            return _NumpyIndex(self._dim)

    def _get_all_embeddings(self) -> np.ndarray:
        """Reconstruct full embedding matrix from FAISS index."""
        n = len(self._names)
        if n == 0:
            return np.zeros((0, self._dim), dtype=np.float32)
        try:
            import faiss
            # Reconstruct from flat index
            if hasattr(self._index, "reconstruct_n"):
                return self._index.reconstruct_n(0, n)
        except Exception:
            pass
        # NumpyIndex fallback
        if isinstance(self._index, _NumpyIndex):
            return self._index._embs.copy()
        return np.zeros((0, self._dim), dtype=np.float32)


# ── Pure-numpy FAISS fallback ─────────────────────────────────────────────────

class _NumpyIndex:
    """Minimal FAISS-compatible index using numpy dot product (no FAISS needed)."""

    def __init__(self, dim: int) -> None:
        self._dim  = dim
        self._embs = np.zeros((0, dim), dtype=np.float32)

    def add(self, x: np.ndarray) -> None:
        self._embs = np.vstack([self._embs, x]) if self._embs.shape[0] else x.copy()

    def search(self, q: np.ndarray, k: int):
        n = self._embs.shape[0]
        if n == 0:
            return np.array([[-1.0]]), np.array([[-1]])
        sims  = (self._embs @ q.T).flatten()    # (n,)
        top_k = min(k, n)
        idxs  = np.argsort(sims)[::-1][:top_k]
        return sims[idxs].reshape(1, -1), idxs.reshape(1, -1)

    def reconstruct_n(self, start: int, n: int) -> np.ndarray:
        return self._embs[start:start + n].copy()
