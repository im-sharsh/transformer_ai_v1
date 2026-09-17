"""
Tiny Transformer - a genuine (if small) transformer, trained entirely from
scratch, implemented in plain NumPy (no PyTorch/TensorFlow dependency).

Purpose: prove the shaped pipeline output has learnable signal, not just
plausible-looking text. This is a SANITY CHECK, not a production training
system - a real fine-tune of a real LLM is a separate, much larger effort.

Architecture (single-head, single-layer, intentionally small):
  token embedding + positional embedding
  -> self-attention (Q/K/V, scaled dot-product)
  -> residual
  -> feed-forward (ReLU)
  -> residual
  -> mean-pool over sequence
  -> linear classifier head + softmax

All forward and backward passes are implemented manually (standard
analytic gradients for matmul/softmax/ReLU/embedding lookup) - there is
no autograd library involved.
"""
import re
import numpy as np


class SimpleTokenizer:
    """Word-level tokenizer built from the training corpus itself."""

    def __init__(self, max_vocab=500):
        self.max_vocab = max_vocab
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, texts):
        from collections import Counter
        counts = Counter()
        for t in texts:
            counts.update(self._split(t))
        for tok, _ in counts.most_common(self.max_vocab - len(self.token_to_id)):
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)
        return self

    def _split(self, text):
        # match long hex-like IDs (e.g. hashed entity/counterparty values) as a
        # SINGLE token first - otherwise alpha/digit splitting fragments a
        # hash like 'd0105305d854766b' into inconsistent pieces, diluting
        # exactly the signal a categorical ID is supposed to provide.
        return re.findall(r"[0-9a-f]{8,}|[a-zA-Z]+|\d+|[^\sa-zA-Z\d]", text.lower())

    def encode(self, text, max_len):
        ids = [self.token_to_id.get(tok, 1) for tok in self._split(text)][:max_len]
        ids += [0] * (max_len - len(ids))
        return np.array(ids, dtype=np.int64)

    @property
    def vocab_size(self):
        return len(self.token_to_id)


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class TinyTransformer:
    def __init__(self, vocab_size, num_classes, max_len, d_model=32, d_ff=64, seed=0):
        rng = np.random.default_rng(seed)
        self.d = d_model
        self.max_len = max_len

        def init(*shape):
            return rng.normal(0, 0.1, size=shape)

        self.E = init(vocab_size, d_model)
        self.P = init(max_len, d_model)
        self.Wq = init(d_model, d_model)
        self.Wk = init(d_model, d_model)
        self.Wv = init(d_model, d_model)
        self.Wo = init(d_model, d_model)
        self.W1 = init(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = init(d_ff, d_model)
        self.b2 = np.zeros(d_model)
        self.Wc = init(d_model, num_classes)
        self.bc = np.zeros(num_classes)

    def _forward(self, token_ids):
        L = len(token_ids)
        d = self.d
        emb = self.E[token_ids] + self.P[:L]

        Q, K, V = emb @ self.Wq, emb @ self.Wk, emb @ self.Wv
        raw = (Q @ K.T) / np.sqrt(d)
        attn = _softmax(raw, axis=-1)
        context = attn @ V
        attn_out = context @ self.Wo
        res1 = emb + attn_out

        z1 = res1 @ self.W1 + self.b1
        ff1 = np.maximum(z1, 0)
        ff2 = ff1 @ self.W2 + self.b2
        res2 = res1 + ff2

        pooled = res2.mean(axis=0)
        logits = pooled @ self.Wc + self.bc
        probs = _softmax(logits)

        cache = dict(token_ids=token_ids, emb=emb, Q=Q, K=K, V=V, attn=attn,
                     context=context, res1=res1, z1=z1, ff1=ff1, res2=res2,
                     pooled=pooled, L=L)
        return probs, cache

    def _backward(self, probs, true_label, cache):
        d = self.d
        L = cache["L"]
        grads = {}

        dlogits = probs.copy()
        dlogits[true_label] -= 1  # softmax + cross-entropy gradient

        grads["Wc"] = np.outer(cache["pooled"], dlogits)
        grads["bc"] = dlogits
        dpooled = self.Wc @ dlogits

        dres2 = np.tile(dpooled / L, (L, 1))  # mean-pool gradient

        dff2 = dres2  # residual: res2 = res1 + ff2
        grads["W2"] = cache["ff1"].T @ dff2
        grads["b2"] = dff2.sum(axis=0)
        dff1 = dff2 @ self.W2.T
        dz1 = dff1 * (cache["z1"] > 0)
        grads["W1"] = cache["res1"].T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        dres1_from_ff = dz1 @ self.W1.T

        dres1 = dres2 + dres1_from_ff  # residual passthrough + ff branch
        dattn_out = dres1  # residual: res1 = emb + attn_out
        demb = dres1.copy()

        grads["Wo"] = cache["context"].T @ dattn_out
        dcontext = dattn_out @ self.Wo.T
        dattn = dcontext @ cache["V"].T

        # softmax jacobian for attention weights (row-wise)
        attn = cache["attn"]
        dscaled = attn * (dattn - np.sum(attn * dattn, axis=-1, keepdims=True))
        draw = dscaled / np.sqrt(d)

        dQ = draw @ cache["K"]
        dK = draw.T @ cache["Q"]
        dV = cache["attn"].T @ dcontext

        grads["Wq"] = cache["emb"].T @ dQ
        grads["Wk"] = cache["emb"].T @ dK
        grads["Wv"] = cache["emb"].T @ dV

        demb += dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T

        grads["E_rows"] = (cache["token_ids"], demb)
        grads["P_rows"] = demb[:L]

        return grads

    def _apply_grads(self, grads, lr, clip_norm=5.0):
        # gradient clipping: without this, occasional large updates on
        # unfamiliar/high-variance real-world data can cause the loss to
        # explode to NaN - clip each gradient array by its own norm.
        def _clip(g):
            norm = np.linalg.norm(g)
            return g if norm <= clip_norm else g * (clip_norm / (norm + 1e-8))

        self.Wc -= lr * _clip(grads["Wc"])
        self.bc -= lr * _clip(grads["bc"])
        self.W2 -= lr * _clip(grads["W2"])
        self.b2 -= lr * _clip(grads["b2"])
        self.W1 -= lr * _clip(grads["W1"])
        self.b1 -= lr * _clip(grads["b1"])
        self.Wo -= lr * _clip(grads["Wo"])
        self.Wq -= lr * _clip(grads["Wq"])
        self.Wk -= lr * _clip(grads["Wk"])
        self.Wv -= lr * _clip(grads["Wv"])

        token_ids, demb = grads["E_rows"]
        np.add.at(self.E, token_ids, -lr * _clip(demb))
        L = len(token_ids)
        self.P[:L] -= lr * _clip(grads["P_rows"])

    def predict(self, token_ids):
        probs, _ = self._forward(token_ids)
        return int(np.argmax(probs)), probs

    def train_step(self, token_ids, true_label, lr):
        probs, cache = self._forward(token_ids)
        loss = -np.log(max(probs[true_label], 1e-9))
        grads = self._backward(probs, true_label, cache)
        self._apply_grads(grads, lr)
        return loss


def run_training_sanity_check(train_examples, val_examples, epochs=25, lr=0.08,
                               max_len=48, d_model=32, max_examples=400, seed=0):
    """
    train_examples / val_examples: lists of {"context": [str], "output": str}
    (as produced by the categorization task shaper).

    Returns a dict with per-epoch history (train_loss, train_accuracy,
    val_accuracy), before/after sample predictions, baselines (random-guess
    AND majority-class), a per-class accuracy breakdown, a small-sample
    warning where relevant, and an explicit verdict - everything needed to
    judge "did this actually work," not just raw numbers.
    """
    rng = np.random.default_rng(seed)
    train_examples = list(train_examples)
    val_examples = list(val_examples)
    if len(train_examples) > max_examples:
        idx = rng.choice(len(train_examples), max_examples, replace=False)
        train_examples = [train_examples[i] for i in idx]
    if len(val_examples) > max_examples // 2:
        idx = rng.choice(len(val_examples), max_examples // 2, replace=False)
        val_examples = [val_examples[i] for i in idx]

    labels = sorted({e["output"] for e in train_examples + val_examples})
    label_to_id = {l: i for i, l in enumerate(labels)}

    tokenizer = SimpleTokenizer().fit([e["context"][0] for e in train_examples])
    model = TinyTransformer(tokenizer.vocab_size, len(labels), max_len, d_model=d_model, seed=seed)

    def _encode_set(examples):
        return [(tokenizer.encode(e["context"][0], max_len), label_to_id[e["output"]], e)
                for e in examples]

    train_enc = _encode_set(train_examples)
    val_enc = _encode_set(val_examples)

    # majority-class baseline: what a model that always predicts the most
    # common label would score - a much more honest bar than random chance
    # on imbalanced data, where "always guess the common one" can look
    # deceptively good against a random baseline alone.
    train_label_counts = {}
    for _, y, _ in train_enc:
        train_label_counts[y] = train_label_counts.get(y, 0) + 1
    majority_label_id = max(train_label_counts, key=train_label_counts.get) if train_label_counts else None
    majority_baseline_accuracy = None
    if val_enc and majority_label_id is not None:
        majority_baseline_accuracy = sum(1 for _, y, _ in val_enc if y == majority_label_id) / len(val_enc)

    def _accuracy(dataset):
        if not dataset:
            return None
        correct = sum(1 for ids, y, _ in dataset if model.predict(ids)[0] == y)
        return correct / len(dataset)

    before_samples = []
    for ids, y, e in val_enc[:5]:
        pred_id, probs = model.predict(ids)
        before_samples.append({
            "text": e["context"][0], "true": e["output"],
            "predicted": labels[pred_id], "confidence": round(float(probs[pred_id]), 3),
        })

    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(train_enc))
        epoch_loss = 0.0
        for i in order:
            ids, y, _ = train_enc[i]
            epoch_loss += model.train_step(ids, y, lr)
        avg_loss = epoch_loss / max(len(train_enc), 1)
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": _accuracy(train_enc),
            "val_accuracy": _accuracy(val_enc),
        })

    after_samples = []
    for ids, y, e in val_enc[:5]:
        pred_id, probs = model.predict(ids)
        after_samples.append({
            "text": e["context"][0], "true": e["output"],
            "predicted": labels[pred_id], "confidence": round(float(probs[pred_id]), 3),
        })

    # per-class accuracy on validation - catches a model that looks good
    # overall but has quietly failed on rarer categories
    per_class_correct, per_class_total = {}, {}
    for ids, y, _ in val_enc:
        pred_id, _ = model.predict(ids)
        label_name = labels[y]
        per_class_total[label_name] = per_class_total.get(label_name, 0) + 1
        if pred_id == y:
            per_class_correct[label_name] = per_class_correct.get(label_name, 0) + 1
    per_class_accuracy = {
        l: round(per_class_correct.get(l, 0) / per_class_total[l], 3)
        for l in per_class_total
    }

    small_sample_warning = None
    if len(train_enc) < 50 or len(val_enc) < 20:
        small_sample_warning = (
            f"Small sample size (train={len(train_enc)}, val={len(val_enc)}) - results here "
            f"are noisy and could look good or bad partly by chance, not just data quality."
        )

    final_val_accuracy = history[-1]["val_accuracy"] if history else None
    verdict, verdict_level = _make_verdict(
        final_val_accuracy, majority_baseline_accuracy,
        1.0 / len(labels) if labels else None, small_sample_warning,
    )

    return {
        "history": history,
        "before_samples": before_samples,
        "after_samples": after_samples,
        "num_classes": len(labels),
        "random_baseline_accuracy": 1.0 / len(labels) if labels else None,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "final_val_accuracy": final_val_accuracy,
        "final_train_accuracy": history[-1]["train_accuracy"] if history else None,
        "per_class_accuracy": per_class_accuracy,
        "small_sample_warning": small_sample_warning,
        "verdict": verdict,
        "verdict_level": verdict_level,  # "good" | "weak" | "none"
        "train_examples_used": len(train_enc),
        "val_examples_used": len(val_enc),
    }


def _make_verdict(final_val_accuracy, majority_baseline, random_baseline, small_sample_warning):
    """Turn the raw numbers into an explicit, honest answer to
    'did this actually work' - rather than leaving the user to eyeball it."""
    if final_val_accuracy is None or majority_baseline is None:
        return "Not enough data to evaluate.", "none"

    margin_over_majority = final_val_accuracy - majority_baseline

    if margin_over_majority > 0.15:
        verdict = (
            f"✅ Learned real signal — validation accuracy ({final_val_accuracy:.0%}) clears "
            f"both the random-guess baseline ({random_baseline:.0%}) and the majority-class "
            f"baseline ({majority_baseline:.0%}) by a solid margin."
        )
        level = "good"
    elif margin_over_majority > 0.03:
        verdict = (
            f"⚠️ Learned something, but weakly — validation accuracy ({final_val_accuracy:.0%}) "
            f"only edges past the majority-class baseline ({majority_baseline:.0%}). More data, "
            f"more epochs, or stronger signal in the source columns may help."
        )
        level = "weak"
    else:
        verdict = (
            f"❌ No real signal detected — validation accuracy ({final_val_accuracy:.0%}) doesn't "
            f"clear the majority-class baseline ({majority_baseline:.0%}), meaning a model that "
            f"always guessed the most common category would do about as well."
        )
        level = "none"

    if small_sample_warning:
        verdict += f" ({small_sample_warning})"

    return verdict, level
