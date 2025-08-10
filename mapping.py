# mapping.py
import re
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def convert_for_output(v):
    try:
        import json
        json.dumps(v)
        return v
    except:
        if isinstance(v, pd.DataFrame):
            return v.to_dict(orient="records")
        if isinstance(v, pd.Series):
            return v.tolist()
        try:
            if isinstance(v, np.generic):
                return v.item()
        except:
            pass
        return v if not (isinstance(v, bytes)) else v.decode(errors='ignore')

def map_questions_to_schema(answers_dict, schema_hints):
    """
    Map freeform Q->A dict into ordered schema_hints.
    """
    q_keys = list(answers_dict.keys())
    q_norms = [normalize_text(k) for k in q_keys]

    mapped = {}
    for hint in schema_hints:
        hint_norm = normalize_text(hint.replace('_', ' '))
        value = None

        if "chart" in hint_norm or "plot" in hint_norm:
            # prefer data:image, then long base64, then question keyword match
            for k in q_keys:
                v = answers_dict[k]
                if isinstance(v, str) and v.startswith("data:image"):
                    value = v
                    break
            if value is None:
                for k in q_keys:
                    v = answers_dict[k]
                    if isinstance(v, str) and len(v) > 200:
                        sample = v.strip().replace("\n","")
                        if re.fullmatch(r"[A-Za-z0-9+/=\s]+", sample[:300]):
                            value = v
                            break
            if value is None:
                for k, kn in zip(q_keys, q_norms):
                    if any(tok in kn for tok in ("plot", "chart", "graph", "visualize", "encode")):
                        value = answers_dict[k]
                        break
        else:
            # fuzzy match
            best_idx, best_score = None, 0.0
            for i, kn in enumerate(q_norms):
                score = SequenceMatcher(None, hint_norm, kn).ratio()
                if score > best_score:
                    best_score, best_idx = score, i
            if best_idx is not None and best_score >= 0.25:
                value = answers_dict[q_keys[best_idx]]
            else:
                # token overlap fallback
                for i, kn in enumerate(q_norms):
                    for token in hint_norm.split():
                        if token and token in kn:
                            value = answers_dict[q_keys[i]]
                            break
                    if value is not None:
                        break

        mapped[hint] = convert_for_output(value)
    return {k: mapped.get(k) for k in schema_hints}
