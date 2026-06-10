import sys
from unittest.mock import MagicMock

# --- KAGGLE/ENVIRONMENT FIX: Mock torchvision if it's broken ---
# This prevents 'RuntimeError: operator torchvision::nms does not exist' which blocks transformers imports.
try:
    import torchvision
except (ImportError, RuntimeError) as e:
    # If torchvision is missing or throws a registration error (common on Kaggle with torch mismatch)
    # we mock it because Chatterbox doesn't actually use torchvision.
    if isinstance(e, ImportError) or "torchvision::nms" in str(e):
        mock_tv = MagicMock()
        mock_tv.__version__ = "0.0.0"
        sys.modules["torchvision"] = mock_tv
        sys.modules["torchvision.transforms"] = MagicMock()
        sys.modules["torchvision.ops"] = MagicMock()
        print("⚠️ Warning: torchvision is missing or broken. Mocked to allow transformers import.")

import argparse
import os
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import librosa
import numpy as np
from safetensors.torch import save_file, load_file as load_safetensors
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, PeftModel
import gc

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import S3_SR, S3Tokenizer
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import MTLTokenizer

from datasets import load_dataset, Audio

class MultilingualDataset(Dataset):
    def __init__(self, data_path, tokenizer, device, languages=None):
        self.tokenizer = tokenizer
        self.device = device
        
        print(f"Loading local dataset from {data_path}...")
        data_files = [str(p) for p in Path(data_path).rglob("*.parquet")]
        if not data_files:
            raise FileNotFoundError(f"No parquet files found in {data_path}")
        
        self.full_ds = load_dataset("parquet", data_files=data_files, split="train")
        self.full_ds = self.full_ds.cast_column("audio", Audio(sampling_rate=S3_SR, decode=False))
        
        # Filter by language if specified
        if languages:
            lang_set = set(languages)
            print(f"🔍 Filtering dataset for languages: {lang_set}")
            self.full_ds = self.full_ds.filter(
                lambda x: (x.get('language') or x.get('lang') or '') in lang_set,
                num_proc=4
            )
            print(f"✅ After filtering: {len(self.full_ds)} items")
        
        self.full_ds = self.full_ds.shuffle(seed=42)
        print(f"✅ Loaded {len(self.full_ds)} items from parquet files.")

    def __len__(self):
        return len(self.full_ds)

    def __getitem__(self, idx):
        item = self.full_ds[idx]
        audio_data = item['audio']
        text = item.get('text') or ""
        lang = item.get('language') or item.get('lang') or "ne"

        # Prepend logic handled securely by explicit language mapping
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=lang, lowercase=False).squeeze(0)
        
        # T3 expects [START] and [STOP] tokens (IDs 255 and 0)
        text_tokens = F.pad(text_tokens, (1, 0), value=255) # start_text_token
        text_tokens = F.pad(text_tokens, (0, 1), value=0)   # stop_text_token

        # Decode parquet audio
        import io
        import soundfile as sf
        import torchaudio.functional as F_audio
        if isinstance(audio_data, dict) and 'bytes' in audio_data and audio_data['bytes'] is not None:
            with io.BytesIO(audio_data['bytes']) as f:
                wav, orig_sr = sf.read(f)
            if orig_sr != S3_SR:
                wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
                wav_t = F_audio.resample(wav_t, orig_sr, S3_SR)
                wav = wav_t.squeeze(0).numpy()
        else:
            wav = audio_data['array']
        
        wav = wav.astype(np.float32)
        # Skip or clip long audio (14.5s safe limit)
        if len(wav) > 14.5 * S3_SR: 
            wav = wav[:int(14.5 * S3_SR)]

        return {
            "text_tokens": text_tokens,
            "wav": wav
        }

def collate_fn(batch):
    text_tokens = [item['text_tokens'] for item in batch]
    wavs = [item['wav'] for item in batch]
    
    text_token_lens = torch.tensor([len(t) for t in text_tokens])
    
    # Pad sequences
    text_tokens_padded = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=705)  # [PAD] token, not [STOP]
    
    return {
        "text_tokens": text_tokens_padded,
        "text_token_lens": text_token_lens,
        "wavs": wavs
    }

def train(args):
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        device = torch.device(args.device)
        rank = 0

    if rank == 0 and not args.no_wandb:
        wandb.init(project="chatterbox-nepali-finetune")
        wandb.config.update(args)

    api = HfApi() if args.push_to_hub else None
    
    # We want to keep track of the base model for PEFT metadata
    base_model_id = "ResembleAI/chatterbox"

    # Load pretrained components
    # We check if ckpt_dir is a full model or just a LoRA adapter
    is_lora_ckpt = args.ckpt_dir and os.path.exists(os.path.join(args.ckpt_dir, "adapter_config.json"))
    
    if args.ckpt_dir and not is_lora_ckpt:
        print(f"📂 Loading full model checkpoint from {args.ckpt_dir}")
        model_wrapper = ChatterboxMultilingualTTS.from_local(args.ckpt_dir, device)
    else:
        print("🌐 Loading base model from pretrained weights")
        model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)
    
    t3 = model_wrapper.t3
    tokenizer = model_wrapper.tokenizer
    
    # ─── RESIZE TEXT EMBEDDINGS ───
    # The tokenizer dynamically adds [ne] and [mai] tokens, so the embedding matrix must be resized to prevent CUDA out-of-bounds errors.
    vocab_size = len(tokenizer.tokenizer.get_vocab())
    print(f"📏 Resizing T3 vocabulary to match tokenizer: {vocab_size} tokens")
    t3.resize_text_embeddings(vocab_size)
    
    # Optional: Resume from an intermediate training checkpoint
    # ─── CONFIGURE / RESUME LORA ───
    is_resumed = False
    if args.resume_t3_weights:
        resume_path = Path(args.resume_t3_weights)
        if resume_path.is_dir():
            print(f"🔄 Resuming from PEFT checkpoint: {resume_path}")
            emb_path = resume_path / "text_emb.pt"
            if emb_path.exists():
                print(f"   📥 Loading text embeddings from {emb_path}")
                t3.text_emb.load_state_dict(torch.load(emb_path, map_location=device, weights_only=True))
            
            # Freeze base before loading PEFT
            for param in t3.parameters():
                param.requires_grad = False
                
            t3.tfmr = PeftModel.from_pretrained(t3.tfmr, resume_path, is_trainable=True)
            is_resumed = True
            print("   ✅ LoRA adapter loaded.")
        else:
            print(f"🔄 Resuming training from file {args.resume_t3_weights}...")
            resume_state = torch.load(args.resume_t3_weights, map_location="cpu", weights_only=True)
            cleaned_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in resume_state.items()}
            
            # Resize if needed
            if "text_emb.weight" in cleaned_state:
                ckpt_vocab_size = cleaned_state["text_emb.weight"].shape[0]
                if ckpt_vocab_size != t3.hp.text_tokens_dict_size:
                    t3.resize_text_embeddings(ckpt_vocab_size)
            
            t3.load_state_dict(cleaned_state, strict=False)
            is_resumed = True
            
            # ─── LORA INJECTION ───
            if is_lora_ckpt:
                print(f"🔄 Resuming from LoRA adapter: {args.ckpt_dir}")
                t3.tfmr = PeftModel.from_pretrained(t3.tfmr, args.ckpt_dir, is_trainable=True)
                # Also load text embeddings if they exist in the adapter folder
                emb_path = os.path.join(args.ckpt_dir, "text_emb.pt")
                if os.path.exists(emb_path):
                    print(f"📥 Loading text embeddings from adapter folder")
                    t3.text_emb.load_state_dict(torch.load(emb_path, map_location=device))
            else:
                print("Injecting LoRA adapters into T3 Transformer...")
                lora_config = LoraConfig(
                    r=32,
                    lora_alpha=64,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type=None,
                )
                t3.tfmr = get_peft_model(t3.tfmr, lora_config)
    else:
        # Standard LoRA injection
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        print("Injecting LoRA adapters into T3 Transformer...")
        t3.tfmr = get_peft_model(t3.tfmr, lora_config)
    
    t3.tfmr.print_trainable_parameters()
    
    # Allow text embeddings to be updated so it perfectly learns the new [ne] tag
    t3.text_emb.requires_grad = True 
    
    # Feature extractors stay on GPU for FAST batch processing
    s3_tokenizer = model_wrapper.s3gen.tokenizer.to(device)
    voice_encoder = model_wrapper.ve.to(device)
    
    # OPTIMIZATION: Initialize [ne] and [mai] tags from [hi] (Hindi) tag
    # This prevents the initial gibberish by starting with a related language!
    with torch.no_grad():
        vocab = tokenizer.tokenizer.get_vocab()
        hi_idx = vocab.get("[hi]")
        ne_idx = vocab.get("[ne]")
        mai_idx = vocab.get("[mai]")
        
        if not args.skip_hi_init and hi_idx is not None and not is_resumed:
            for tgt_idx, name in [(ne_idx, "[ne]"), (mai_idx, "[mai]")]:
                if tgt_idx is not None and t3.text_emb.weight.shape[0] > tgt_idx:
                    print(f"🎯 Initializing {name} tag ({tgt_idx}) weights from [hi] tag ({hi_idx})...")
                    t3.text_emb.weight[tgt_idx] = t3.text_emb.weight[hi_idx].clone()
                    t3.text_head.weight[tgt_idx] = t3.text_head.weight[hi_idx].clone()
        elif is_resumed:
            print("🎯 Skipping tag re-initialization (resuming from checkpoint)")

    t3.to(device)
    
    # ─── FREEZE BASE MODEL ───
    # (Handled above in resume/config block)
    
    if args.distributed:
        t3 = DDP(t3, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    t3.train()
    
    # Dataset
    data_path = args.wav_dir if args.wav_dir else "data/chatterbox-multilingual-data"
    languages = args.languages.split(',') if args.languages else None
    dataset = MultilingualDataset(data_path, tokenizer, device="cpu", languages=languages)
    
    sampler = DistributedSampler(dataset) if args.distributed else None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type != "cpu" else False
    )
    
    trainable_params = [p for p in t3.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)
    
    # Gradient checkpointing safely turned back on
    if device.type == "cuda":
        t3_model = t3.module if args.distributed else t3
        t3_model.tfmr.gradient_checkpointing_enable()
        if rank == 0:
            print("🧠 Gradient checkpointing enabled to prevent OOM")
    
    # Mixed precision for memory savings (critical for T4 GPUs)
    use_amp = args.fp16 and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None
    if use_amp and rank == 0:
        print("⚡ Mixed precision (float16) enabled")
    
    # ─── TRAINING LOOP ───
    start_epoch = 0
    resume_source = args.ckpt_dir or args.resume_t3_weights
    if resume_source and "epoch_" in str(resume_source):
        import re
        match = re.search(r"epoch_(\d+)", str(resume_source))
        if match:
            start_epoch = int(match.group(1)) + 1
            if rank == 0:
                print(f"⏩ Resuming from Epoch {start_epoch}")
            
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
            
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if rank == 0 else dataloader
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            
            text_tokens = batch['text_tokens'].to(device)
            text_token_lens = batch['text_token_lens'].to(device)
            wavs = batch['wavs']
            
            # Fast GPU feature extraction inside the training loop to avoid CPU bottlenecks
            with torch.no_grad():
                speech_tokens_list = []
                speaker_embs_list = []
                for w in wavs:
                    st, _ = s3_tokenizer.forward([w])
                    speech_tokens_list.append(st.squeeze(0).to(device))
                    ve = torch.from_numpy(voice_encoder.embeds_from_wavs([w], sample_rate=16000)).to(device)
                    speaker_embs_list.append(ve.mean(axis=0, keepdim=True))
                
                speech_token_lens = torch.tensor([len(s) for s in speech_tokens_list], device=device)
                speech_tokens = torch.nn.utils.rnn.pad_sequence(speech_tokens_list, batch_first=True, padding_value=6562)
                speaker_emb = torch.cat(speaker_embs_list, dim=0)

            # Prepare T3Cond
            prompt_len = t3.module.hp.speech_cond_prompt_len if args.distributed else t3.hp.speech_cond_prompt_len
            cond_prompt_tokens = speech_tokens[:, :prompt_len] if speech_tokens.size(1) > prompt_len else None
            
            t3_cond = T3Cond(
                speaker_emb=speaker_emb,
                cond_prompt_speech_tokens=cond_prompt_tokens,
                emotion_adv=0.5 * torch.ones(text_tokens.size(0), 1, 1, device=device)
            )
            
            with autocast("cuda", enabled=use_amp, dtype=torch.float16):
                loss_text, loss_speech = t3.module.loss(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_token_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_token_lens
                ) if args.distributed else t3.loss(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_token_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_token_lens
                )
            
            loss = (loss_text + loss_speech) / args.accum_steps
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(dataloader):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if rank == 0:
                pbar.set_postfix({"loss": loss.item() * args.accum_steps, "loss_speech": loss_speech.item()})
                
                if not args.no_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "loss": loss.item() * args.accum_steps,
                        "loss_text": loss_text.item(),
                        "loss_speech": loss_speech.item(),
                        "lr": optimizer.param_groups[0]['lr']
                    })
            
            # Memory cleanup
            del loss, loss_text, loss_speech, t3_cond
            if i % 10 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            
        # Save checkpoint and generate sample
        if rank == 0 and epoch % args.save_every == 0:
            name_prefix = args.push_to_hub.split('/')[-1] if args.push_to_hub else "lora_extended"
            ckpt_dir = f"{name_prefix}_epoch_{epoch}"
            os.makedirs(ckpt_dir, exist_ok=True)
            
            model_to_save = t3.module if args.distributed else t3
            
            # Save the PEFT adapter and embeddings
            model_to_save.tfmr.config.base_model_name_or_path = base_model_id
            model_to_save.tfmr.save_pretrained(ckpt_dir)
            torch.save(model_to_save.text_emb.state_dict(), f"{ckpt_dir}/text_emb.pt")
            
            # Create a basic README for HF compatibility
            readme_path = os.path.join(ckpt_dir, "README.md")
            with open(readme_path, "w") as f:
                f.write(f"---\nbase_model: {base_model_id}\nlibrary_name: peft\ntags:\n- text-to-speech\n- nepali\n- maithili\n---\n")
            
            print(f"   💾 Saved LoRA adapter to: {ckpt_dir}")
            
            # --- LOCAL CHECKPOINT ROTATION ---
            if epoch > 0:
                prev_ckpt_dir = f"{name_prefix}_epoch_{epoch-1}"
                if os.path.exists(prev_ckpt_dir):
                    import shutil
                    print(f"🧹 Cleaning up previous local checkpoint: {prev_ckpt_dir}")
                    shutil.rmtree(prev_ckpt_dir)
            
            # --- PUSH TO HUB --- 
            if args.push_to_hub:
                try:
                    token = os.environ.get("HF_TOKEN")
                    print(f"🚀 Pushing checkpoint {ckpt_dir} to {args.push_to_hub}...")
                    api.upload_folder(
                        folder_path=ckpt_dir,
                        path_in_repo=ckpt_dir,
                        repo_id=args.push_to_hub,
                        token=token
                    )
                    print("✅ Checkpoint pushed successfully!")
                except Exception as e:
                    print(f"⚠️ Could not push checkpoint: {e}")
            
            # --- VALIDATION SAMPLE GENERATION ---
            print(f"\n📊 Generating validation sample for epoch {epoch}...")
            model_to_save.eval()
            with torch.no_grad():
                test_text = "नमस्ते, म नेपाली बोलिरहेको छु। यो तालिम कस्तो चलिरहेको छ?"
                
                try:
                    # We reuse the model_wrapper's high-level generate logic
                    old_t3 = model_wrapper.t3
                    model_wrapper.t3 = model_to_save
                    
                    val_wav = model_wrapper.generate(
                        test_text, 
                        language_id="ne", 
                        audio_prompt_path=args.eval_audio_path,
                        exaggeration=0.5,
                        temperature=0.8
                    )
                    
                    samples_dir = Path("samples")
                    samples_dir.mkdir(exist_ok=True)
                    output_sample_path = samples_dir / f"sample_epoch_{epoch}.wav"
                    
                    import soundfile as sf
                    sf.write(str(output_sample_path), val_wav.squeeze(0).cpu().numpy(), model_wrapper.sr)
                    print(f"✅ Sample saved to {output_sample_path}")
                except Exception as e:
                    print(f"⚠️ Could not generate sample: {e}")
                finally:
                    model_wrapper.t3 = old_t3
                    torch.cuda.empty_cache()
            
            model_to_save.train()
            # Memory cleanup after validation
            gc.collect()
            torch.cuda.empty_cache()
            # -----------------------------------
    
    if rank == 0:
        model_to_save = t3.module if args.distributed else t3
        
        final_dir = "lora_extended_final"
        os.makedirs(final_dir, exist_ok=True)
        model_to_save.tfmr.config.base_model_name_or_path = base_model_id
        model_to_save.tfmr.save_pretrained(final_dir)
        torch.save(model_to_save.text_emb.state_dict(), f"{final_dir}/text_emb.pt")
        with open(os.path.join(final_dir, "README.md"), "w") as f:
            f.write(f"---\nbase_model: {base_model_id}\nlibrary_name: peft\ntags:\n- text-to-speech\n- nepali\n- maithili\n---\n")
        print("Training finished. Saved LoRA to lora_extended_final directory.")
        
        if args.push_to_hub:
            try:
                token = os.environ.get("HF_TOKEN")
                print(f"🚀 Pushing final model to {args.push_to_hub}...")
                api.upload_folder(
                    folder_path=final_dir,
                    path_in_repo=final_dir,
                    repo_id=args.push_to_hub,
                    token=token
                )
                print("✅ Final model pushed successfully!")
            except Exception as e:
                print(f"⚠️ Could not push final model: {e}")
    
    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    def get_default_device():
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest (jsonl, csv, or tsv)")
    parser.add_argument("--wav_dir", type=str, help="Optional explicit directory for audio files")
    parser.add_argument("--ckpt_dir", type=str, help="Path to base pretrained model dir")
    parser.add_argument("--resume_t3_weights", type=str, help="Path to a t3_nepali_epoch_X.pt file to resume from")
    parser.add_argument("--device", type=str, default=get_default_device(), help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--eval_audio_path", type=str, help="Path to reference audio for validation samples")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (M2 Max can handle 16-32)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps to simulate larger batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU threads for background data loading")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for fine-tuning")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (usually 1e-5 to 5e-5)")
    parser.add_argument("--save_every", type=int, default=5, help="Save interval in epochs")
    parser.add_argument("--skip_hi_init", action="store_true", help="Skip initializing Nepali weights from Hindi (helps with accent if you have enough data)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training (DDP)")
    parser.add_argument("--fp16", action="store_true", help="Enable float16 mixed precision (saves ~40%% GPU memory)")
    parser.add_argument("--push_to_hub", type=str, help="Hugging Face repo ID to push checkpoints to (e.g. officialuser/chatterbox-nepali)")
    parser.add_argument("--languages", type=str, help="Comma-separated list of language codes to filter for (e.g. ne,mai)")
    
    args = parser.parse_args()
    
    if args.device == "mps":
        print("🚀 Using MPS (Metal Performance Shaders) for Mac acceleration")
    
    train(args)
