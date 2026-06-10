# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import logging
import torch
from dataclasses import dataclass
from types import MethodType


logger = logging.getLogger(__name__)


LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResult:
    # was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    false_start: bool
    # was this frame detected as being part of a long tail with potential hallucinations?
    long_tail: bool
    # was this frame detected as repeating existing text content?
    repetition: bool
    # was the alignment position of this frame too far from the previous frame?
    discontinuity: bool
    # has inference reached the end of the text tokens? eg, this remains false if inference stops early
    complete: bool
    # approximate position in the text token sequence. Can be used for generating online timestamps.
    position: int


class AlignmentStreamAnalyzer:
    def __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0, lang="en"):
        """
        Some transformer TTS models implicitly solve text-speech alignment in one or more of their self-attention
        activation maps. This module exploits this to perform online integrity checks which streaming.
        A hook is injected into the specified attention layer, and heuristics are used to determine alignment
        position, repetition, etc.

        NOTE: currently requires no queues.
        """
        print(f"📊 [AlignmentStreamAnalyzer] Initialized for text slice: {text_tokens_slice}")
        # self.queue = queue
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.lang = lang
        self.alignment = torch.zeros(0, j-i).to(tfmr.device)
        # self.alignment_bin = torch.zeros(0, j-i)
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None
        self.emergency_brake = False
        
        # Track generated tokens for repetition detection
        self.generated_tokens = []
        self.last_tokens = []
        
        # Stability counters
        self.discontinuity_streak = 0
        self.stagnation_streak = 0
        self.last_text_posn = -1

        # Using `output_attentions=True` is incompatible with optimized attention kernels, so
        # using it for all layers slows things down too much. We can apply it to just one layer
        # by intercepting the kwargs and adding a forward hook (credit: jrm)
        self.last_aligned_attns = []
        for i, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
            self.last_aligned_attns += [None]
            self._add_attention_spy(tfmr, i, layer_idx, head_idx)

    def _add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):
        """
        Adds a forward hook to a specific attention layer to collect outputs.
        """
        def attention_forward_hook(module, input, output):
            """
            See `LlamaAttention.forward`; the output is a 3-tuple: `attn_output, attn_weights, past_key_value`.
            NOTE:
            - When `output_attentions=True`, `LlamaSdpaAttention.forward` calls `LlamaAttention.forward`.
            - `attn_output` has shape [B, H, T0, T0] for the 0th entry, and [B, H, 1, T0+i] for the rest i-th.
            """
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                step_attention = output[1]  # Keep on GPU: (B, n_heads, T0, Ti)
                self.last_aligned_attns[buffer_idx] = step_attention[0, head_idx]  # (T0, Ti)

        target_layer = tfmr.layers[layer_idx].self_attn
        # Register hook and store the handle
        target_layer.register_forward_hook(attention_forward_hook)
        if hasattr(tfmr, 'config') and hasattr(tfmr.config, 'output_attentions'):
            self.original_output_attentions = tfmr.config.output_attentions
            self.original_attn_implementation = getattr(tfmr.config, '_attn_implementation', None)
            if getattr(tfmr.config, '_attn_implementation', None) == 'sdpa':
                tfmr.config._attn_implementation = 'eager'
            tfmr.config.output_attentions = True

    def step(self, logits, next_token=None):
        """
        Emits an AlignmentAnalysisResult into the output queue, and potentially modifies the logits to force an EOS.
        """
        # extract approximate alignment matrix chunk (1 frame at a time after the first chunk)
        aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0) # (N, N)
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            # first chunk has conditioning info, text tokens, and BOS token
            A_chunk = aligned_attn[j:, i:j].clone() # (T, S)
        else:
            # subsequent chunks have 1 frame due to KV-caching
            A_chunk = aligned_attn[:, i:j].clone() # (1, S)

        # Tightened monotonic mask (12 tokens) to force alignment on every word.
        T, S = A_chunk.shape
        mask_threshold = self.text_position + 12
        if mask_threshold < S:
            A_chunk[:, mask_threshold:] = 0


        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

        A = self.alignment
        T, S = A.shape

        # update position
        cur_text_posn = A_chunk[-1].argmax()
        # Widened window to allow for more natural look-ahead context (-5 to +20)
        discontinuity = not(-5 < cur_text_posn - self.text_position < 20) 
        if not discontinuity:
            self.text_position = cur_text_posn

        # Hallucinations at the start of speech show up as activations at the bottom of the attention maps!
        false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = self.curr_frame_pos

        # 3. Check for sentence completion
        # S-2 is slightly more conservative than S-3 to prevent premature completion
        self.complete = self.complete or (self.text_position >= S - 2)
        
        # Stagnation check: if we are near the end and attention is sticking
        if not self.complete and self.text_position >= S - 2:
            # sum attention on the last 2 tokens over the last 2 frames
            if A[-2:, -2:].sum() > 1.5:
                self.complete = True

        if self.complete and self.completed_at is None:
            self.completed_at = self.curr_frame_pos
        
        # 4. Long tail check (model keeps going after sentence)
        long_tail = False
        if self.complete and self.completed_at is not None:
            # Check if attention is sticking to the last few tokens (punctuation/end)
            long_tail = (A[self.completed_at:, -6:].sum(dim=0).max() >= 0.8)
        self.long_tail_triggered = long_tail.item() if isinstance(long_tail, torch.Tensor) else long_tail
        
        # 5. Token repetition check
        token_repetition = False
        if next_token is not None:
            # Convert tensor to scalar if needed
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = next_token
            
            self.last_tokens.append(token_id)
            if len(self.last_tokens) > 15:
                self.last_tokens.pop(0)
            
            if len(self.last_tokens) >= 10:
                most_common = max(set(self.last_tokens), key=self.last_tokens.count)
                if self.last_tokens.count(most_common) >= 8:
                    token_repetition = True
            
            self.generated_tokens.append(token_id)

        if self.complete and self.completed_at is not None:
            # Language-aware delay: English needs more time to finish syllables (~320ms)
            # Nepali/Maithili are crisper and need less (~80ms)
            delay = 8 if self.lang == 'en' else 2
            if (self.curr_frame_pos - self.completed_at) > delay:
                self.emergency_brake = True
        
        # 6b. Stability Checks (Discontinuity & Stagnation)
        if discontinuity:
            self.discontinuity_streak += 1
        else:
            self.discontinuity_streak = 0
            
        if cur_text_posn == self.last_text_posn:
            self.stagnation_streak += 1
        else:
            self.stagnation_streak = 0
        self.last_text_posn = cur_text_posn

        # If model is lost (discontinuous) for 30 frames (~1.2s), or stuck for 40 frames (~1.6s)
        # We ignore discontinuity during the first 50 steps (warm-up period)
        is_lost = self.discontinuity_streak > 30 and self.curr_frame_pos > 50
        is_stuck = self.stagnation_streak > 40
        
        if is_lost or is_stuck:
            import logging
            logging.getLogger(__name__).warning(f"Stability brake triggered: discontinuity={self.discontinuity_streak}, stagnation={self.stagnation_streak}. Forcing EOS.")
            self.emergency_brake = True

        self.emergency_brake_triggered = self.emergency_brake
        
        # 7. Final Force EOS check
        force_eos = self.emergency_brake or self.long_tail_triggered or token_repetition
        
        if force_eos:
            import logging
            logging.getLogger(__name__).warning(f"forcing EOS token, long_tail={self.long_tail_triggered}, emergency_brake={self.emergency_brake}, token_repetition={token_repetition}")
            logits = torch.full_like(logits, -100.0)
            logits[0, self.eos_idx] = 100.0

        self.curr_frame_pos += 1
        return logits