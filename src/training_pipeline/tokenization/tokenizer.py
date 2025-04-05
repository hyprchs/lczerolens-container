"""
This module defines the tokenization strategy for HyprGM, a multimodal LLM designed for chess instruction.

It introduces custom tokens for chess-related **entities** and **actions**:

Entities:
- Chess squares (e.g., "a1", "h8").
- Piece types (e.g., "knight", "bishop") and symbols (e.g., "N" in "Nf3").
- SAN-specific symbols (e.g., "+", "x", "=") and NAGs (e.g., "!!", "?").

Actions:
- Move tokens for updating the board state (e.g., ["<|move_start|>", "g1", "f3", "<|move_end|>"]).
- Undo tokens (e.g., "<|move_undo|>") to revert moves.
- Tokens for manually resetting the FEN or navigating to a new position.

These tokens enable HyprGM to align chessboard states with video transcripts, integrate contextual
chess knowledge via Leela's feature vector, and dynamically generate text/commentary with board awareness.

Note that if we use causal (as opposed to masked) language modelling, we must avoid converting "future"
Action tokens that correspond to a change in board state to a Chess Token with Leela. This is because
the model should generate the Action Tokens during inference, only converting them to a Chess Token
when the group of Action Tokens is completed (i.e. the model has successfully specified a full move).
We do not want the model to predict the Chess Tokens directly.

## Terminology
- **Board-change token(s) (BTs)**: A token or group of tokens that correspond to a change in the board state.
- **Chess Token (CT)**: A "token" from the perspective of the LLM, but really it's a semantic vector
  representing a new board state produced by Leela Chess Zero and then projected onto the LLM's embedding space
  using an MLP. See the [Fuyu model](https://www.adept.ai/blog/fuyu-8b) for more details. (I'll have to find the
  source, but I believe a recent LLaVA model saw significantly better multimodal benchmarks by using a slightly
  larger 2-layer MLP for the projection, as opposed to Fuyu's linear projection.)
"""

from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding
from typing import List, Tuple, Union


class HyprGMTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
