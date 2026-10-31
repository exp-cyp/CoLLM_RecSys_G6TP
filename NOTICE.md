# References and attribution

This educational implementation follows:

Yang Zhang, Fuli Feng, Jizhi Zhang, Keqin Bao, Qifan Wang, and Xiangnan He.
**CoLLM: Integrating Collaborative Embeddings Into Large Language Models for Recommendation.**
IEEE Transactions on Knowledge and Data Engineering, 37(5), 2025.
DOI: https://doi.org/10.1109/TKDE.2025.3540912
Preprint: https://arxiv.org/abs/2310.19488

Reference repository: https://github.com/zyang1580/CoLLM
Inspected commit: `df1a24dcc083ce9568565b72def58c7af80cd8c9`.
The reference repository is BSD-3-Clause licensed, with MiniGPT-4/LAVIS notices.
Its prompt templates, preprocessing notebook, stage-freezing logic, and MLP
architecture informed this smaller implementation. The accompanying upstream
license is retained in `docs/licenses/CoLLM-BSD.txt`.

MovieLens-1M is provided by GroupLens: https://grouplens.org/datasets/movielens/1m/.
Observe its README and data-use conditions. Data is downloaded separately and is
not included in this repository.

Qwen2-0.5B is provided by the Qwen team under Apache-2.0:
https://huggingface.co/Qwen/Qwen2-0.5B.
Weights and their license are downloaded separately; provenance records the exact
model revision. Hugging Face Transformers and PEFT provide the model and LoRA implementation.

LLM assistance was used in implementation and documentation.
