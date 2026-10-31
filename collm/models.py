"""Core architecture: frozen MF -> shared MLP -> LLM input embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

HISTORY_LIMIT = 10
TITLE_TOKEN_LIMIT = 48
MAPPER_HIDDEN_MULTIPLIER = 10
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("q_proj", "v_proj")


class MatrixFactorization(nn.Module):
    """Learn user/item factors and score their dot product without bias terms."""

    def __init__(self, num_users, num_items, dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, dim, padding_idx=0)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        self.register_buffer("seen_users", torch.zeros(num_users, dtype=torch.bool))
        self.register_buffer("seen_items", torch.zeros(num_items, dtype=torch.bool))

    def set_seen(self, rows):
        self.seen_users[[r["uid"] for r in rows]] = True
        self.seen_items[[r["iid"] for r in rows]] = True

    def features(self, users, items):
        # A truly unseen ID gets zero information, not an arbitrary random vector.
        u = self.user_embedding(users) * self.seen_users[users].unsqueeze(-1)
        i = self.item_embedding(items) * self.seen_items[items].unsqueeze(-1)
        return u, i

    def forward(self, users, items):
        u, i = self.features(users, items)
        return (u * i).sum(-1)


class PromptDataset(Dataset):
    """Encode prompts once, keeping target/features intact when history is long."""

    def __init__(self, rows, tokenizer, max_length=256, collaborative=False):
        def encode(text):
            return tokenizer.encode(text, add_special_tokens=False)

        prefix = encode("#Question: A user has given high ratings to the following movies: ")
        suffix = encode('? Answer with "Yes" or "No".\n#Answer:')
        # These fragments are identical for every record; tokenize them only once.
        if collaborative:
            before_user = encode(
                ". Additionally, we have information about the user's preferences "
                "encoded in the feature "
            )
            before_title = encode(
                ". Using all available information, make a prediction about whether "
                "the user would enjoy the movie titled "
            )
            before_item = encode(" with the feature ")
        else:
            before_title = encode(
                ". Leverage the information to predict whether the user would enjoy "
                "the movie titled "
            )

        self.examples = []
        for row in rows:
            history = encode("; ".join(row["history_titles"][-HISTORY_LIMIT:]) or "(none)")
            title = encode(row["title"])[:TITLE_TOKEN_LIMIT]
            if collaborative:
                placeholder = [tokenizer.eos_token_id]
                tail = (
                    before_user
                    + placeholder
                    + before_title
                    + title
                    + before_item
                    + placeholder
                    + suffix
                )
                user_in_tail = len(before_user)
                item_in_tail = (
                    len(before_user) + 1 + len(before_title) + len(title) + len(before_item)
                )
            else:
                tail = before_title + title + suffix
            room = max_length - len(prefix) - len(tail)
            if room < 1:
                raise ValueError("max_length too short for the fixed prompt and target title")
            history = history[-room:]
            ids = prefix + history + tail
            offset = len(prefix) + len(history)
            self.examples.append(
                {
                    "input_ids": ids,
                    "uid": row["uid"],
                    "iid": row["iid"],
                    "label": row["label"],
                    "user_pos": offset + user_in_tail if collaborative else -1,
                    "item_pos": offset + item_in_tail if collaborative else -1,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class Collator:
    """Left-pad prompts and shift collaborative feature positions consistently."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        length = max(len(e["input_ids"]) for e in examples)
        fields = (
            "input_ids",
            "attention_mask",
            "uid",
            "iid",
            "label",
            "user_pos",
            "item_pos",
        )
        result = {key: [] for key in fields}
        for e in examples:
            pad = length - len(e["input_ids"])
            result["input_ids"].append([self.pad_id] * pad + e["input_ids"])
            result["attention_mask"].append([0] * pad + [1] * len(e["input_ids"]))
            for key in ("uid", "iid", "label"):
                result[key].append(e[key])
            for key in ("user_pos", "item_pos"):
                result[key].append(e[key] + pad if e[key] >= 0 else -1)
        return {
            key: torch.tensor(values, dtype=torch.float32 if key == "label" else torch.long)
            for key, values in result.items()
        }


class CoLLM(nn.Module):
    """Score the Yes logit, tuning LoRA first and only the mapper second."""

    def __init__(self, lm, mf, yes_id, stage="stage1"):
        super().__init__()
        self.lm = lm
        self.mf = mf
        dim = mf.user_embedding.embedding_dim
        hidden_dim = MAPPER_HIDDEN_MULTIPLIER * dim
        self.mapper = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, lm.config.hidden_size),
        )
        self.yes_id = int(yes_id)
        self.set_stage(stage)

    def set_stage(self, stage):
        if stage not in ("stage1", "stage2"):
            raise ValueError(stage)
        self.stage = stage
        for name, p in self.lm.named_parameters():
            p.requires_grad_(stage == "stage1" and "lora_" in name)
        for p in self.mf.parameters():
            p.requires_grad_(False)
        for p in self.mapper.parameters():
            p.requires_grad_(stage == "stage2")

    def train(self, mode=True):
        super().train(mode)
        self.mf.eval()
        if self.stage == "stage2":
            self.lm.eval()  # Disable frozen LoRA dropout; keep autograd through the LLM.
        return self

    def forward(self, input_ids, attention_mask, uid, iid, user_pos, item_pos):
        embeddings = self.lm.get_input_embeddings()(input_ids)
        if self.stage == "stage2":
            if bool((user_pos < 0).any() or (item_pos < 0).any()):
                raise ValueError("CoLLM stage2 requires both collaborative placeholders")
            u, i = self.mf.features(uid, iid)
            u = self.mapper(u).to(embeddings.dtype)
            i = self.mapper(i).to(embeddings.dtype)
            batch_index = torch.arange(input_ids.shape[0], device=input_ids.device)
            embeddings = embeddings.clone()
            embeddings[batch_index, user_pos] = u
            embeddings[batch_index, item_pos] = i
        positions = attention_mask.cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 0)
        # LoRA layers are installed inside the base decoder. Computing just its final
        # hidden state and one output row avoids a [batch, length, vocabulary] tensor.
        decoder = self.lm.get_base_model().model
        hidden = decoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, -1]
        head = self.lm.get_output_embeddings()
        weight = head.weight[self.yes_id : self.yes_id + 1]
        bias = (
            head.bias[self.yes_id : self.yes_id + 1]
            if getattr(head, "bias", None) is not None
            else None
        )
        # Follow the upstream Qwen implementation: BCE on the raw Yes logit,
        # rather than softmax-normalizing Yes against No or the whole vocabulary.
        return F.linear(hidden, weight, bias).squeeze(-1).float()


def build_model(cfg, mf, stage, adapter_path: Path | None = None):
    """Verify pinned local assets, then attach a new or selected LoRA adapter."""

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .assets import verify_local_model
    from .common import ROOT

    path = ROOT / cfg["model"]["path"]
    verify_local_model(path, cfg["model"]["id"], cfg["model"]["revision"])
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    if len(yes_ids) != 1:
        raise ValueError("Yes must be one token for binary next-token scoring")
    dtype = torch.bfloat16 if cfg["device"] == "xpu" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=dtype, attn_implementation="sdpa"
    )
    base.config.use_cache = False
    if adapter_path is not None:
        lm = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    else:
        lm = get_peft_model(
            base,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=list(LORA_TARGET_MODULES),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    model = CoLLM(lm, mf, yes_ids[0], stage)
    return model.to(cfg["device"]), tokenizer
