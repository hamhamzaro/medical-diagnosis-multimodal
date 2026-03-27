"""
preprocess/text_preprocess.py
------------------------------
Clinical report preprocessing and BERT tokenization pipeline.

Steps:
    1. Load clinical reports (CSV / JSON / TXT)
    2. De-identification (patient names, dates, IDs)
    3. Section segmentation (findings, impression, history)
    4. Text cleaning (abbreviation expansion, noise removal)
    5. BERT tokenization (bert-base-uncased, max 512 tokens)
    6. Export tokenized tensors for fast loading

Usage:
    python src/preprocess/text_preprocess.py
"""

import re
import json
import pandas as pd
import numpy as np
import torch
import os
from transformers import BertTokenizer
from pathlib import Path
from tqdm import tqdm
from typing import Optional


# ─── De-identification Patterns ───────────────────────────────────────────────

DEIDENT_PATTERNS = [
    # Patient names (e.g., "Patient: John Doe" or "M. Dupont")
    (r"\b(patient|mr|mrs|ms|dr|prof)\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+", "[NAME]"),
    # Dates (various formats)
    (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]"),
    (r"\b(january|february|march|april|may|june|july|august|september|"
     r"october|november|december)\s+\d{1,2},?\s+\d{4}", "[DATE]"),
    # Ages
    (r"\b\d{1,3}[-\s]?(year|yr|y\.o\.?|ans?)[-\s]?(old)?\b", "[AGE]"),
    # IDs and accession numbers
    (r"\b[A-Z]{2,4}\d{6,}\b", "[ID]"),
    (r"\bDOB:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", "[DOB]"),
    # Phone numbers
    (r"\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
]

# Medical abbreviation expansions
ABBREVIATION_MAP = {
    r"\bpt\b": "patient",
    r"\bhx\b": "history",
    r"\bdx\b": "diagnosis",
    r"\brx\b": "treatment",
    r"\bw/o\b": "without",
    r"\bw/\b": "with",
    r"\bs/p\b": "status post",
    r"\bc/o\b": "complains of",
    r"\bSOB\b": "shortness of breath",
    r"\bCP\b": "chest pain",
    r"\bHTN\b": "hypertension",
    r"\bDM\b": "diabetes mellitus",
    r"\bCAD\b": "coronary artery disease",
    r"\bCOPD\b": "chronic obstructive pulmonary disease",
}

# Clinical section headers
SECTION_HEADERS = {
    "findings":   r"(findings?|observations?|résultats?)\s*:?",
    "impression": r"(impression|conclusion|assessment|assessment and plan)\s*:?",
    "history":    r"(clinical\s+history|history|antécédents?)\s*:?",
    "technique":  r"(technique|protocol|acquisition)\s*:?",
}


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def deidentify(text: str) -> str:
    """Replace PHI (Protected Health Information) with placeholders."""
    for pattern, replacement in DEIDENT_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def expand_abbreviations(text: str) -> str:
    """Expand common medical abbreviations for better tokenization."""
    for pattern, expansion in ABBREVIATION_MAP.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


def clean_text(text: str) -> str:
    """
    General text cleaning for clinical reports.

    - Remove excessive whitespace
    - Normalize punctuation
    - Remove non-ASCII characters (keep accented chars for French)
    - Lowercase

    Args:
        text: Raw clinical report string.

    Returns:
        Cleaned string.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize dashes and quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Lowercase
    text = text.lower()
    return text


def segment_report(text: str) -> dict[str, str]:
    """
    Segment a clinical report into semantic sections.

    Args:
        text: Full cleaned report string.

    Returns:
        Dict with section names as keys and section text as values.
    """
    sections = {k: "" for k in SECTION_HEADERS}
    sections["full"] = text

    for section_name, pattern in SECTION_HEADERS.items():
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            start = matches[0].end()
            # Find next section header
            next_starts = []
            for other_name, other_pattern in SECTION_HEADERS.items():
                if other_name == section_name:
                    continue
                other_matches = list(re.finditer(other_pattern, text[start:], re.IGNORECASE))
                if other_matches:
                    next_starts.append(start + other_matches[0].start())

            end = min(next_starts) if next_starts else len(text)
            sections[section_name] = text[start:end].strip()

    return sections


def full_preprocess(text: str) -> str:
    """Apply full preprocessing pipeline to a single report."""
    text = deidentify(text)
    text = expand_abbreviations(text)
    text = clean_text(text)
    return text


# ─── BERT Tokenization ────────────────────────────────────────────────────────

class ClinicalTokenizer:
    """
    BERT tokenizer wrapper for clinical reports.

    Args:
        model_name: HuggingFace tokenizer identifier.
        max_length: Maximum sequence length (512 for BERT).
        use_sections: If True, prepend section tags to improve context.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        use_sections: bool = True
    ):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.use_sections = use_sections

    def tokenize_report(self, text: str) -> dict[str, torch.Tensor]:
        """
        Tokenize a single clinical report.

        Args:
            text: Preprocessed report string.

        Returns:
            Dict with 'input_ids', 'attention_mask', 'token_type_ids'.
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def tokenize_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a list of reports (batch)."""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def build_text_dataset(
    reports_path: str,
    output_dir: str,
    text_col: str = "report",
    label_cols: Optional[list[str]] = None,
    max_length: int = 512
) -> pd.DataFrame:
    """
    Load, preprocess, and tokenize clinical reports from a CSV file.

    Expected CSV format:
        report_id | report | label_1 | label_2 | ...

    Args:
        reports_path: Path to CSV file.
        output_dir:   Directory to save tokenized tensors.
        text_col:     Column name containing report text.
        label_cols:   List of label column names.
        max_length:   Max BERT token length.

    Returns:
        DataFrame with preprocessed text and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(reports_path)
    print(f"Loaded {len(df)} reports from {reports_path}")

    tokenizer = ClinicalTokenizer(max_length=max_length)

    # Preprocess text
    print("Preprocessing reports...")
    df["text_clean"] = df[text_col].apply(full_preprocess)
    df["n_tokens"] = df["text_clean"].apply(
        lambda t: len(tokenizer.tokenizer.tokenize(t))
    )

    print(f"  Token stats: mean={df['n_tokens'].mean():.0f}, "
          f"max={df['n_tokens'].max()}, "
          f"truncated={( df['n_tokens'] > max_length).sum()}")

    # Tokenize all reports
    print("Tokenizing...")
    all_input_ids, all_attention_masks = [], []

    for text in tqdm(df["text_clean"].tolist()):
        encoded = tokenizer.tokenize_report(text)
        all_input_ids.append(encoded["input_ids"])
        all_attention_masks.append(encoded["attention_mask"])

    input_ids = torch.stack(all_input_ids)           # (N, max_length)
    attention_masks = torch.stack(all_attention_masks)

    # Save tensors
    torch.save(input_ids, os.path.join(output_dir, "input_ids.pt"))
    torch.save(attention_masks, os.path.join(output_dir, "attention_masks.pt"))
    print(f"Tokenized tensors saved → {output_dir}")

    # Save labels if provided
    if label_cols:
        labels = torch.tensor(df[label_cols].values, dtype=torch.float32)
        torch.save(labels, os.path.join(output_dir, "labels.pt"))
        print(f"Labels saved: {labels.shape}")

    # Save metadata
    df.drop(columns=[text_col], errors="ignore").to_csv(
        os.path.join(output_dir, "metadata.csv"), index=False
    )
    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/reports.csv")
    parser.add_argument("--output", type=str, default="data/processed/text")
    parser.add_argument("--text-col", type=str, default="report")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    df = build_text_dataset(
        args.input, args.output,
        text_col=args.text_col,
        max_length=args.max_length
    )
    print(f"\nDone. {len(df)} reports preprocessed.")
    print("Next step: run train.py")
