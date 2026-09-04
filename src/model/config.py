from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Architecture configuration for D32-1216-R12.

    The Q/K RMSNorm parameters are per projected channel (384 Q channels and
    128 K/V channels), while each xIELU owns two trainable scalar parameters.
    This convention is part of the frozen parameter-count contract.
    """

    model_name: str = "D32-1216-R12"
    vocab_size: int = 16_384
    d_model: int = 384
    depth: int = 32
    num_q_heads: int = 6
    num_kv_heads: int = 2
    head_dim: int = 64
    mlp_hidden_dim: int = 1_216
    output_residual_rank: int = 12
    max_seq_len: int = 1_024
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    xielu_alpha_p_init: float = 0.5
    xielu_alpha_n_init: float = 1.0
    tie_embeddings: bool = True
    use_qk_norm: bool = True
    use_bias: bool = False

    def __post_init__(self) -> None:
        if self.d_model != self.num_q_heads * self.head_dim:
            raise ValueError("d_model must equal num_q_heads * head_dim")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if self.vocab_size >= 65_536:
            raise ValueError("vocab_size must fit uint16 token storage")
        if self.depth <= 0 or self.mlp_hidden_dim <= 0:
            raise ValueError("depth and mlp_hidden_dim must be positive")
        if self.output_residual_rank <= 0:
            raise ValueError("output_residual_rank must be positive")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if not self.tie_embeddings:
            raise ValueError("the frozen D32-1216-R12 architecture requires tied embeddings")
        if not self.use_qk_norm:
            raise ValueError("the frozen D32-1216-R12 architecture requires Q/K RMSNorm")
        if self.use_bias:
            raise ValueError("the frozen D32-1216-R12 architecture is bias-free")

    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def parameter_count(self) -> int:
        """Return the exact count of trainable parameters in the model.

        The two scalar xIELU parameters per block are intentional. With the
        frozen configuration this evaluates to exactly 49,001,408.
        """

        token_embedding = self.vocab_size * self.d_model
        attention = self.q_dim * self.d_model + 2 * self.kv_dim * self.d_model
        attention += self.q_dim * self.d_model
        attention += self.q_dim + self.kv_dim  # Q/K RMSNorm
        mlp = 2 * self.d_model * self.mlp_hidden_dim
        block_norms = 2 * self.d_model
        xielu = 2
        blocks = self.depth * (attention + mlp + block_norms + xielu)
        final_norm = self.d_model
        residual_head = self.output_residual_rank * (self.d_model + self.vocab_size)
        return token_embedding + blocks + final_norm + residual_head

    def parameter_breakdown(self) -> dict[str, int]:
        """Return an auditable trainable-parameter breakdown."""

        per_layer = {
            "q_projection": self.q_dim * self.d_model,
            "k_projection": self.kv_dim * self.d_model,
            "v_projection": self.kv_dim * self.d_model,
            "o_projection": self.q_dim * self.d_model,
            "qk_rmsnorm": self.q_dim + self.kv_dim,
            "attention_rmsnorm": 2 * self.d_model,
            "mlp_in_projection": self.d_model * self.mlp_hidden_dim,
            "mlp_out_projection": self.mlp_hidden_dim * self.d_model,
            "xielu_scalars": 2,
        }
        breakdown = {
            "token_embedding_and_tied_lm_head": self.vocab_size * self.d_model,
            "blocks": self.depth * sum(per_layer.values()),
            "final_rmsnorm": self.d_model,
            "rank_12_residual_output_head": self.output_residual_rank
            * (self.d_model + self.vocab_size),
        }
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "ModelConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})
