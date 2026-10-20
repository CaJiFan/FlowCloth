"""
trtm_gnn_model.py
─────────────────
Pure PyTorch implementation of the original TRTM (Tracking and Reconstruction of
Thin-shell Materials) model architecture.

Architecture Overview:
  - Backbone: ResNet-34 image encoder + AdaptiveAvgPool2d((1, 1)) -> 512-dim visual latent
  - Node Encoder: Linear(515 -> 256) -> ReLU -> Linear(256 -> 128) -> LayerNorm(128)
  - Edge Encoder: Linear(4 -> 256) -> ReLU -> Linear(256 -> 128) -> LayerNorm(128)
  - 15 GraphNetBlocks:
      * Edge Model: MLP(384 -> 128 -> 128 -> 128) -> LayerNorm(128)
      * Attention: Linear(128 -> 1) -> LeakyReLU(0.2) -> Softmax per receiver node
      * Node Model: MLP(256 -> 128 -> 128 -> 128) -> LayerNorm(128)
      * Residual skip connections on both edge and node latents
  - Node Decoder: Linear(128 -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU -> Linear(128 -> 3)

Eliminates external C++ wheel dependencies (torch_scatter) by implementing
group-wise scatter operations natively via torch.scatter_reduce_ and torch.scatter_add_.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class LazyMLP(nn.Module):
    """Sequential MLP mimicking TRTM's original LazyMLP layer structure."""

    def __init__(self, in_features: int, widths: list[int]):
        super().__init__()
        layers = OrderedDict()
        prev = in_features
        for i, w in enumerate(widths):
            layers[f"linear_{i}"] = nn.Linear(prev, w)
            if i < len(widths) - 1:
                layers[f"relu_{i}"] = nn.ReLU()
            prev = w
        self.layers = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Attention(nn.Module):
    """
    Graph Attention module with pure PyTorch scatter_softmax.
    Calculates attention coefficients across all incoming edges for each receiver node.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, edge_features: torch.Tensor, receivers: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Parameters
        ----------
        edge_features : torch.Tensor [B, E, 128]
        receivers     : torch.Tensor [E] long
        num_nodes     : int (number of nodes V, e.g. 2601)

        Returns
        -------
        attention : torch.Tensor [B, E, 1] normalized edge attention weights
        """
        B, E, _ = edge_features.shape
        x = self.activation(self.linear(edge_features))  # [B, E, 1]

        # 1. Receiver-wise Max for numerical stability
        idx = receivers.unsqueeze(0).unsqueeze(-1).expand(B, E, 1)
        max_val = torch.full((B, num_nodes, 1), float("-inf"), dtype=x.dtype, device=x.device)
        max_val.scatter_reduce_(1, idx, x, reduce="amax", include_self=False)
        max_val[max_val == float("-inf")] = 0.0

        # 2. Exponentiate shifted logits
        max_per_edge = max_val.gather(1, idx)
        exp_x = torch.exp(x - max_per_edge)

        # 3. Receiver-wise Sum
        sum_val = torch.zeros((B, num_nodes, 1), dtype=x.dtype, device=x.device)
        sum_val.scatter_add_(1, idx, exp_x)

        # 4. Normalize
        sum_per_edge = sum_val.gather(1, idx)
        attention = exp_x / (sum_per_edge + 1e-12)
        return attention


class GraphNetBlock(nn.Module):
    """Single GraphNet interaction layer with edge, attention, node update, and residuals."""

    def __init__(self):
        super().__init__()
        self.edge_model = nn.Sequential(
            LazyMLP(384, [128, 128, 128]),
            nn.LayerNorm(128),
        )
        self.node_model = nn.Sequential(
            LazyMLP(256, [128, 128, 128]),
            nn.LayerNorm(128),
        )
        self.attention_model = Attention()

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        node_feats : [B, V, 128]
        edge_feats : [B, E, 128]
        senders    : [E]
        receivers  : [E]

        Returns
        -------
        out_nodes : [B, V, 128]
        out_edges : [B, E, 128]
        """
        B, V, _ = node_feats.shape
        sender_feats = torch.index_select(node_feats, 1, senders)      # [B, E, 128]
        receiver_feats = torch.index_select(node_feats, 1, receivers)  # [B, E, 128]

        # 1. Update edges
        cat_edge = torch.cat([sender_feats, receiver_feats, edge_feats], dim=-1)  # [B, E, 384]
        updated_edges = self.edge_model(cat_edge)                                 # [B, E, 128]

        # 2. Compute attention & aggregate to receivers
        attention = self.attention_model(updated_edges, receivers, V)              # [B, E, 1]
        weighted_edges = updated_edges * attention                                # [B, E, 128]

        idx = receivers.unsqueeze(0).unsqueeze(-1).expand(B, -1, 128)
        agg_edges = torch.zeros((B, V, 128), dtype=node_feats.dtype, device=node_feats.device)
        agg_edges.scatter_add_(1, idx, weighted_edges)                            # [B, V, 128]

        # 3. Update nodes
        cat_node = torch.cat([node_feats, agg_edges], dim=-1)                     # [B, V, 256]
        updated_nodes = self.node_model(cat_node)                                 # [B, V, 128]

        # 4. Residual connections
        out_nodes = node_feats + updated_nodes
        out_edges = edge_feats + updated_edges
        return out_nodes, out_edges


class Encoder(nn.Module):
    """Encodes initial node [pos(3) + img_feat(512)] and edge [rel_pos(3) + dist(1)] latents."""

    def __init__(self):
        super().__init__()
        self.node_encoder = nn.Sequential(
            LazyMLP(515, [256, 128]),
            nn.LayerNorm(128),
        )
        self.edge_encoder = nn.Sequential(
            LazyMLP(4, [256, 128]),
            nn.LayerNorm(128),
        )


class Decoder(nn.Module):
    """Decodes latent node embeddings to 3D Cartesian coordinates (x, y, z)."""

    def __init__(self):
        super().__init__()
        self.node_decoder = LazyMLP(128, [128, 128, 3])


class Updater(nn.Module):
    """Chains 15 message passing GraphNetBlocks sequentially."""

    def __init__(self, steps: int = 15):
        super().__init__()
        submodules = OrderedDict()
        for i in range(steps):
            submodules[str(i)] = GraphNetBlock()
        self.submodules = nn.Sequential(submodules)


class LearnedModel(nn.Module):
    """TRTM Graph Neural Network processor."""

    def __init__(self, steps: int = 15):
        super().__init__()
        self.encoder = Encoder()
        self.updater = Updater(steps=steps)
        self.decoder = Decoder()


# ─────────────────────────────────────────────────────────────────────────────
# Top-Level TRTM GNN Model
# ─────────────────────────────────────────────────────────────────────────────

class TRTMGNN(nn.Module):
    """
    End-to-End TRTM Cloth Mesh Estimation Model.
    Input:  Image [B, 3, 224, 224] (RGB/BGR float32 in [0, 1])
    Output: Reconstructed 3D mesh coordinates [B, V, 3] (V=2601 for large square template)
    """

    def __init__(
        self,
        template_path: str | Path,
        edge_path: str | Path,
        steps: int = 15,
    ):
        super().__init__()
        # Backbone: torchvision ResNet-34 encoder
        b_model = models.resnet34()
        self.backbone_model = nn.Module()
        self.backbone_model.encoder = nn.Sequential(
            b_model.conv1,
            b_model.bn1,
            b_model.relu,
            b_model.maxpool,
            b_model.layer1,
            b_model.layer2,
            b_model.layer3,
            b_model.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.learned_model = LearnedModel(steps=steps)

        # Load graph template
        verts = np.loadtxt(str(template_path)).astype(np.float32)
        edges = np.loadtxt(str(edge_path)).astype(np.int64)

        # Construct bidirectional edges
        senders = edges[:, 0]
        receivers = edges[:, 1]
        sender = np.concatenate([senders, receivers], 0)
        receiver = np.concatenate([receivers, senders], 0)

        # Register persistent=False buffers so load_state_dict doesn't require them
        self.register_buffer("template_verts", torch.from_numpy(verts), persistent=False)
        self.register_buffer("senders", torch.from_numpy(sender), persistent=False)
        self.register_buffer("receivers", torch.from_numpy(receiver), persistent=False)

        # Precompute initial edge displacement and distance: [dx, dy, dz, dist]
        v_send = verts[sender]
        v_recv = verts[receiver]
        rel_pos = v_send - v_recv
        rel_dist = np.linalg.norm(rel_pos, axis=-1, keepdims=True)
        edge_feats = np.concatenate([rel_pos, rel_dist], axis=-1).astype(np.float32)
        self.register_buffer("template_edge_feats", torch.from_numpy(edge_feats), persistent=False)

        self.num_vertices = verts.shape[0]
        self.num_edges = sender.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor [B, 3, 224, 224]

        Returns
        -------
        pred_coords : torch.Tensor [B, V, 3]
        """
        B = x.shape[0]
        V = self.template_verts.shape[0]
        E = self.senders.shape[0]

        # 1. ResNet backbone visual feature extraction
        img_feat = self.backbone_model.encoder(x).flatten(1)  # [B, 512]

        # 2. Initial node latents
        img_feat_expanded = img_feat.unsqueeze(1).expand(B, V, -1)             # [B, V, 512]
        node_pos_expanded = self.template_verts.unsqueeze(0).expand(B, V, -1)   # [B, V, 3]
        cat_node = torch.cat([node_pos_expanded, img_feat_expanded], dim=-1)    # [B, V, 515]
        node_latent = self.learned_model.encoder.node_encoder(cat_node)        # [B, V, 128]

        # 3. Initial edge latents
        edge_feat_expanded = self.template_edge_feats.unsqueeze(0).expand(B, E, -1)  # [B, E, 4]
        edge_latent = self.learned_model.encoder.edge_encoder(edge_feat_expanded)    # [B, E, 128]

        # 4. Message passing through 15 GraphNetBlocks
        for block in self.learned_model.updater.submodules:
            node_latent, edge_latent = block(node_latent, edge_latent, self.senders, self.receivers)

        # 5. Decode final 3D node coordinates
        pred_coords = self.learned_model.decoder.node_decoder(node_latent)  # [B, V, 3]
        return pred_coords


# ─────────────────────────────────────────────────────────────────────────────
# Factory / Checkpoint Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_trtm_model(
    ckpt_path: str | Path,
    template_path: Optional[str | Path] = None,
    edge_path: Optional[str | Path] = None,
    device: str | torch.device = "cuda",
) -> TRTMGNN:
    """
    Factory function to instantiate TRTMGNN and load checkpoint_large_size.pt.
    """
    if template_path is None:
        template_path = Path(__file__).resolve().parent.parent / "assets" / "trtm_large" / "flat_state_050.txt"
    if edge_path is None:
        edge_path = Path(__file__).resolve().parent.parent / "assets" / "trtm_large" / "mesh_edge_idx.txt"

    if not Path(template_path).exists():
        raise FileNotFoundError(f"TRTM template file not found at: {template_path}")
    if not Path(edge_path).exists():
        raise FileNotFoundError(f"TRTM edge file not found at: {edge_path}")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"TRTM checkpoint not found at: {ckpt_path}")

    model = TRTMGNN(template_path=template_path, edge_path=edge_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"[TRTM] Successfully loaded {ckpt_path} with strict=True ({len(state_dict)} weights).")

    device = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    import time

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_default = "/home/cjimenez/Downloads/checkpoint_large_size.pt"
    print(f"Testing TRTMGNN on {dev}...")

    model = load_trtm_model(ckpt_default, device=dev)
    dummy = torch.randn(1, 3, 224, 224, device=dev)

    # Warmup
    with torch.no_grad():
        _ = model(dummy)
        t0 = time.perf_counter()
        out = model(dummy)
        dt = (time.perf_counter() - t0) * 1000.0

    print(f"Prediction output shape: {out.shape} (expected [1, 2601, 3])")
    print(f"Inference latency: {dt:.2f} ms ({1000/dt:.1f} FPS)")
