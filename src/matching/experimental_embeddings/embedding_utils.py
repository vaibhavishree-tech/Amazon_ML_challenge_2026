"""
embedding_utils.py

Shared utilities for the experimental pretrained semantic embedding
feature. Used identically across the sanity check, the train/test
embedding builds, and inference -- one implementation, so there is zero
risk of the tagging format or prefix handling drifting between where
embeddings are built and where they are used.

Design decision, worth restating here since it governs every caller of
build_tagged_text(): always pass RAW, uncleaned business_name and
business_address text, never the cleaned/unidecoded versions from the
cleaning phase. The purpose of this feature is to let a multilingual
model recognise semantic equivalence directly from native script (e.g.
Devanagari) -- unidecode's phonetic transliteration already destroys
that information before a lexical similarity metric ever sees it
(confirmed during blocking validation: token-based name similarity
collapsed sharply for names originally written in non-Latin scripts).
Feeding the model the already-transliterated cleaned text would defeat
the purpose entirely.
"""

import torch
from sentence_transformers import SentenceTransformer, util

from config import EMBEDDING_MODEL_NAME, EMBEDDING_QUERY_PREFIX


def load_embedding_model() -> SentenceTransformer:
    """
    Loads the embedding model onto GPU if available, otherwise CPU.
    The device is printed explicitly so it's always clear which one is
    actually in use -- a silent CPU fallback would make later steps
    far slower than expected, with no obvious indication of why.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' on device={device}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    print(f"  model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def build_tagged_text(name: str, address: str, country: str) -> str:
    """
    Builds an explicit-tag input format (field labels included directly
    in the text), rather than an unlabelled concatenation of raw
    strings. Explicit field labels give the embedding model clearer
    structure to attend to when encoding each record.

    Always pass RAW (uncleaned) name/address here -- see module
    docstring for why.
    """
    name = name or ""
    address = address or ""
    country = country or ""
    return f"NAME: {name} | ADDRESS: {address} | COUNTRY: {country}"


def encode_texts(model: SentenceTransformer, texts: list, batch_size: int = 256):
    """
    Encodes a list of already-tagged text strings into normalized
    embeddings, applying the required E5-family "query: " prefix here
    so every caller gets it automatically rather than needing to add it
    individually.

    normalize_embeddings=True means each output vector has unit length,
    so cosine similarity between two embeddings reduces to a plain dot
    product -- cheaper to compute at scale than re-normalizing on every
    comparison.
    """
    prefixed = [EMBEDDING_QUERY_PREFIX + t for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings


def cosine_sim(vec_a, vec_b) -> float:
    """Cosine similarity between two embedding vectors."""
    return float(util.cos_sim(vec_a, vec_b))