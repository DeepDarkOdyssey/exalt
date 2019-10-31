# REF: https://arxiv.org/abs/1909.03104
import numpy as np
from pydmd import HODMD


class EigenSent(object):
    def __init__(self, d: int = 1, n: int = 1, ensure_pos_def: bool = False):
        self.d = d
        self.n = n
        self.ensure_pos_def = ensure_pos_def

    def pool_embeds(self, token_embeddings: np.ndarray, debug: bool = False):
        embed_sequence = token_embeddings.T
        Stilde = np.vstack(
            [
                embed_sequence[:, i : embed_sequence.shape[1] - self.d + i + 1]
                for i in range(self.d)
            ]
        )
        U, s, Vh = np.linalg.svd(Stilde[:, :-1], full_matrices=False)
        if debug:
            print(f"Stilde shape: {Stilde.shape}")
            print(f"U shape: {U.shape}")
            print(f"num singular values: {s.size}")
            print(f"Vh shape: {Vh.shape}")
            assert np.allclose(Stilde[:, :-1], U @ np.diag(s) @ Vh)  # sanity check
            assert np.allclose(
                np.diag(np.reciprocal((s))) @ np.diag(s), np.identity(s.size)
            )
        # Atilde = U.T @ Stilde[:, 1:] @ Vh.T * np.reciprocal(s)
        Atilde = Stilde[:, 1:] @ Vh.T @ np.diag(np.reciprocal(s)) @ U.T
        if debug:
            print(f"Atilde shape: {Atilde.shape}")
            assert np.allclose(Atilde @ Stilde[:, :-1], Stilde[:, 1:])
        if self.ensure_pos_def:
            Atilde = Atilde.T @ Atilde
        eig_vals, eig_vecs = np.linalg.eig(Atilde)
        if debug:
            print(f"num eigen values: {eig_vals.size}")
            print(f"eigen vectors shape: {eig_vecs.shape}")
            assert np.allclose(Atilde @ eig_vecs, eig_vecs @ np.diag(eig_vals))
        # sorted by the magnitude of the eigenvalues
        orders = np.argsort(np.sqrt(np.abs(eig_vals * eig_vals.conj())))[::-1]
        pooled_embed = np.ravel(eig_vecs[:, orders[: self.n]], order="f")
        return pooled_embed