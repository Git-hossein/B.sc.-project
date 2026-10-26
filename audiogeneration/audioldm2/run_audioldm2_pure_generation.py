import argparse
import os
import json
import csv
import shutil
import soundfile as sf
import torch
from diffusers import AudioLDM2Pipeline
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

# --- MONKEY PATCH FOR TRANSFORMERS / AUDIOLDM2 ---
def _patch_update_model_kwargs_for_generation(self, outputs, model_kwargs, *args, **kwargs):
    model_kwargs["past_key_values"] = getattr(outputs, "past_key_values", None)
    if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
        mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat(
            [mask, mask.new_ones((mask.shape[0], 1))], dim=-1
        )
    return model_kwargs

GPT2Model._update_model_kwargs_for_generation = _patch_update_model_kwargs_for_generation


def run_audioldm2_text2audio(
    pipe,
    video_list,
    scratch_output_folder,
    descriptions=None,
    negative_prompt="low quality, distorted, noisy, glitch",
    num_inference_steps=100,
    guidance_scale=3.5,
    audio_length_in_s=10.0,
    seed=None,
    batch_size=2,
    sr=16000,
    device="cuda"
):
    """
    Batched Text-to-Audio generation for a list of video identifiers or filenames.
    """
    inference_queue = []

    for video_item in video_list:
        # Extract stem/vidID (e.g., 'abc.mp4' -> 'abc', or raw 'abc' -> 'abc')
        vid_id = os.path.splitext(os.path.basename(video_item))[0]
        
        prompt = None
        if descriptions and vid_id in descriptions:
            prompt = descriptions[vid_id]
        
        if prompt:
            inference_queue.append((vid_id, prompt))
        else:
            print(f"⚠️ Skipping item '{vid_id}': No prompt found in descriptions.")

    total_items = len(inference_queue)
    print(f"🚀 Processing {total_items} items in batches of {batch_size}...")

    for i in range(0, total_items, batch_size):
        batch = inference_queue[i : i + batch_size]
        batch_ids = [item[0] for item in batch]
        batch_prompts = [item[1] for item in batch]
        batch_neg = [negative_prompt] * len(batch)

        # Set deterministic per-item generator if seed is supplied
        generator = None
        if seed is not None:
            generator = [
                torch.Generator(device=device).manual_seed(seed + i + idx)
                for idx in range(len(batch))
            ]

        print(f"[{i + 1}/{total_items}] Generating audio for: {batch_ids}")

        with torch.inference_mode():
            output = pipe(
                prompt=batch_prompts,
                negative_prompt=batch_neg,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                audio_length_in_s=audio_length_in_s,
                generator=generator
            )

        audios = output.audios

        # Save generated audio as <vidID>_GEN.wav
        for vid_id, audio_waveform in zip(batch_ids, audios):
            out_filename = f"{vid_id}_GEN.wav"
            out_path = os.path.join(scratch_output_folder, out_filename)
            sf.write(out_path, audio_waveform, samplerate=sr)
            print(f"💾 Saved: {out_path}")


if __name__ == "__main__":
    pass