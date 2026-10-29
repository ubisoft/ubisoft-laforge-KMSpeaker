"""
Utility functions for handling long sequences with overlapping chunks.
"""

import logging
import torch
import numpy as np
from typing import Dict, Optional, Callable, Any
from kmspeaker.utils.flow_matching_utils import ModelWrapper

def get_keypoint_style_embedding(conditioning_full, keypoint_chunks, keypoint_style_encoder):
    # Fixed style embedding, shared by all chunks
    if "keypoint_style" in conditioning_full:
        return conditioning_full["keypoint_style"]

    # Compute style embedding from keypoints chunk if encoder is provided
    elif keypoint_style_encoder is not None and "keypoint_style_to_proc" in conditioning_full:
        return keypoint_style_encoder(keypoint_chunks, do_sample=True)

    # No style embedding to process
    return None

def compute_chunked_velocity(
    model: torch.nn.Module,
    x_current: torch.Tensor,
    t_tensor: torch.Tensor,
    audio_features: torch.Tensor,
    cfg: Dict[str, Any],
    conditioning_full: Optional[Dict[str, torch.Tensor]] = None,
    keypoint_style_encoder: Optional[torch.nn.Module] = None,
    cfg_scale: float = 1.0,
    keep_global_style_conditioning_no_matter_cfg_guidance_scale: bool = False,
    previous_window_style: bool = False,
) -> torch.Tensor:
    """
    Computes velocity for overlapping chunks and combines them.
    This is a standalone function that can be used by any sampler (Euler, ODE solver, etc.).
    
    Args:
        model: The velocity model to call
        x_current: Current animation state tensor (batch_size, total_anim_len, input_dim)
        t_tensor: Current time tensor
        audio_features: Audio features tensor (batch_size, total_audio_len, feature_dim)
        cfg: Configuration dictionary containing data and model parameters
        previous_window_style: "previous_window" strategy. Each chunk's style comes from the
            expected_anim_len frames before it, or the first window if there are none.

    Returns:
        Combined velocity tensor for the full sequence
    """
    batch_size, total_anim_len, input_dim = x_current.shape
    total_audio_len = audio_features.shape[1]
    
    # Expected lengths from config
    expected_anim_len = cfg['data']['sequence_length']
    audio_fps = cfg['audio_encoder']['audio_model_fps']
    anim_fps = cfg['data']['animation_fps']

    # Calculate overlapping chunks (50% overlap)
    def anim_to_audio_coords(anim_start, anim_end):
        """Convert animation coordinates to audio coordinates."""
        audio_start = int(np.round(anim_start * audio_fps / anim_fps))
        audio_end = int(np.round(anim_end * audio_fps / anim_fps))
        return audio_start, audio_end
    
    anim_step = expected_anim_len // 2  # 50% overlap
    anim_chunks = []
    audio_chunks = []
    keypoint_chunks = []

    # Generate regular overlapping chunks
    anim_start = 0
    while anim_start + expected_anim_len <= total_anim_len:
        anim_end = anim_start + expected_anim_len
        anim_chunks.append((anim_start, anim_end))
        audio_chunks.append(anim_to_audio_coords(anim_start, anim_end))
        anim_start += anim_step
    
    # Add final chunk if there's remaining sequence
    if anim_chunks and anim_chunks[-1][1] < total_anim_len:
        final_anim_start = total_anim_len - expected_anim_len
        final_anim_end = total_anim_len
        anim_chunks.append((final_anim_start, final_anim_end))
        audio_chunks.append(anim_to_audio_coords(final_anim_start, final_anim_end))
    
    # Initialize combined velocity tensor
    combined_velocity = torch.zeros_like(x_current)
    overlap_counts = torch.zeros((batch_size, total_anim_len, 1), device=x_current.device)
    
    extras = {
        'cfg_scale': cfg_scale,
        'keep_global_style_conditioning_no_matter_cfg_guidance_scale': keep_global_style_conditioning_no_matter_cfg_guidance_scale
    }
    # Process each chunk
    for j, ((anim_start, anim_end), (audio_start, audio_end)) in enumerate(zip(anim_chunks, audio_chunks)):

        # Extract chunks and prepare style
        x_chunk = x_current[:, anim_start:anim_end, :]
        audio_chunk = audio_features[:, audio_start:audio_end, :]
        keypoint_chunks = conditioning_full["keypoint"][:, anim_start:anim_end, :] if "keypoint" in conditioning_full.keys() else None

        if "keypoint_style_to_proc" in conditioning_full.keys():
            if previous_window_style:
                # Previous window, or the first window for chunks starting before expected_anim_len
                style_start = max(0, anim_start - expected_anim_len)
                style_end = style_start + expected_anim_len
                style_ref_len = conditioning_full["keypoint_style_to_proc"].shape[1]
                assert style_end <= style_ref_len, (
                    f"previous_window_style needs style reference frames [{style_start}:{style_end}) "
                    f"for output chunk [{anim_start}:{anim_end}), but the style reference only has "
                    f"{style_ref_len} frames - provide a longer style reference."
                )
            else:
                style_start, style_end = anim_start, anim_end
            keypoint_style_chunks = conditioning_full["keypoint_style_to_proc"][:, style_start:style_end, :]
        else:
            keypoint_style_chunks = None
        keypoint_style_chunk = get_keypoint_style_embedding(conditioning_full, keypoint_style_chunks, keypoint_style_encoder)
        if keypoint_style_encoder is not None: 
            assert keypoint_style_chunk is not None, "Keypoint style chunk embedding should have been computed but is None."

        # Set up conditioning for this chunk
        conditioning = {'audio': audio_chunk}
        if keypoint_chunks is not None:
            conditioning["keypoint"] = keypoint_chunks
            if keypoint_style_chunk is not None:
                conditioning["keypoint_style"] = keypoint_style_chunk
                if "keypoint_style_unconditional" in conditioning_full:
                    conditioning["keypoint_style_unconditional"] = conditioning_full["keypoint_style_unconditional"]

        # Compute velocity for this chunk
        velocity_chunk = model(x=x_chunk, t=t_tensor, conditioning=conditioning, **extras)


        # Add to combined velocity
        combined_velocity[:, anim_start:anim_end, :] += velocity_chunk
        overlap_counts[:, anim_start:anim_end, :] += 1

    
    # Average overlapping regions
    combined_velocity = combined_velocity / overlap_counts
    
    return combined_velocity


class ChunkedModelWrapper(ModelWrapper):
    """
    A wrapper that makes any model work with long sequences by automatically
    chunking the input and stitching back the results.
    """

    def __init__(self, model: torch.nn.Module, cfg: Dict[str, Any], keypoint_style_encoder: Optional[torch.nn.Module] = None):
        super().__init__(model=model)
        self.cfg = cfg
        self.keypoint_style_encoder = keypoint_style_encoder
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        """
        Forward pass that automatically handles chunking if needed.

        Args:
            x: Animation state tensor (batch_size, seq_len, input_dim)
            t: Time tensor
            **extras: Additional conditioning including 'conditioning' dict

        Returns:
            Velocity tensor for the full sequence
        """
        cfg_scale = extras.get('cfg_scale', 1.0)
        keep_global_style_conditioning_no_matter_cfg_guidance_scale = extras.get('keep_global_style_conditioning_no_matter_cfg_guidance_scale', False)
        previous_window_style = extras.get('previous_window_style', False)
        conditioning = extras.get('conditioning', {})
        audio_features = conditioning['audio']
        
        # Extract other than audio conditioning if present
        conditioning_no_audio = {}
        for key, value in conditioning.items():
            if key != 'audio':
                conditioning_no_audio[key] = value

        conditioning_no_audio = conditioning_no_audio if conditioning_no_audio else None
        
        return compute_chunked_velocity(
            model=self.model,
            x_current=x,
            t_tensor=t,
            audio_features=audio_features,
            cfg=self.cfg,
            conditioning_full=conditioning_no_audio,
            keypoint_style_encoder=self.keypoint_style_encoder,
            cfg_scale=cfg_scale,
            keep_global_style_conditioning_no_matter_cfg_guidance_scale=keep_global_style_conditioning_no_matter_cfg_guidance_scale,
            previous_window_style=previous_window_style,
        )
