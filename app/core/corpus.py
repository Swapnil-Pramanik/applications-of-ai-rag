"""The 18-paper demo/eval corpus: arXiv ID -> paper title.

One source of truth, because three places need the same mapping and they must
not drift: `scripts/fetch_corpus.py` (what to download), the golden set's
validator (turning `1706.03762.pdf` back into a human-readable title so the
owner can verify a span against the right paper), and the demo tenant (Day 20).

Files are stored as `<arxiv_id>.pdf`, not `<Title>.pdf`, so the filename is
stable, ASCII, and unambiguous — a golden-set label that points at
`1810.04805.pdf` cannot silently start meaning a different document.
"""

from __future__ import annotations

CORPUS: dict[str, str] = {
    "1706.03762": "Attention Is All You Need",
    "1810.04805": "BERT",
    "2005.14165": "Language Models are Few-Shot Learners (GPT-3)",
    "1512.03385": "Deep Residual Learning (ResNet)",
    "2010.11929": "An Image is Worth 16x16 Words (ViT)",
    "1506.02640": "You Only Look Once (YOLO)",
    "1406.2661": "Generative Adversarial Networks",
    "2006.11239": "Denoising Diffusion Probabilistic Models",
    "1412.6980": "Adam Optimizer",
    "1502.03167": "Batch Normalization",
    "2106.09685": "LoRA",
    "2004.04906": "Dense Passage Retrieval (DPR)",
    "2002.08909": "REALM",
    "2005.11401": "Retrieval-Augmented Generation (RAG)",
    "1702.08734": "Billion-scale similarity search (FAISS)",
    "1603.02754": "XGBoost",
    "1312.5602": "Playing Atari with Deep RL (DQN)",
    "1707.06347": "Proximal Policy Optimization (PPO)",
}


def title_for(filename: str) -> str:
    """`1706.03762.pdf` -> `Attention Is All You Need`; unknown files pass through."""
    return CORPUS.get(filename.removesuffix(".pdf"), filename)
