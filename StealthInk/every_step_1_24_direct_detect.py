import os
import math
import time
import argparse
import random
import json
from functools import partial
from decimal import Decimal

import numpy as np
import torch
from scipy.stats import binom, norm
from sklearn import metrics

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LogitsProcessorList,
    LogitsProcessor,
)

from every_step_processor import WatermarkBase


# -------------------------
# Utilities
# -------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def count_decimal_digits(number):
    s = str(number)
    if "." in s:
        return len(s.split(".")[1])
    return 0


def hamming_distance(bit_str1, bit_str2):
    if len(bit_str1) != len(bit_str2):
        raise ValueError("The input strings must have the same length.")
    return sum(c1 != c2 for c1, c2 in zip(bit_str1, bit_str2))


# -------------------------
# Argparse
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal working example of applying a watermark via LogitsProcessor."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HF model id or local path.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Prompt truncation length (overrides model config).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1,
        help="Maximum number of new tokens per generate() call.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Use multinomial sampling (else greedy/beam).",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=1.0,
        help="Sampling temperature (when --use_sampling=True).",
    )
    parser.add_argument(
        "--chunk_capacity",
        type=int,
        default=1,
        help="bits per chunk.",
    )
    parser.add_argument(
        "--msg_len",
        type=int,
        default=24,
        help="total embedded bits.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="start index of prompts file.",
    )
    parser.add_argument(
        "--generation_num",
        type=int,
        default=400,
        help="generate 400 responses.",
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        default=200,
        help="number of tokens in each response.",
    )
    parser.add_argument(
        "--prompts_fp",
        type=str,
        default="prompts.json",
        help="directory to store responses.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output",
        help="directory to store responses.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Num beams for beam search when --use_sampling=False.",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Run model/watermark ops on GPU if available.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Load model in float16 when possible.",
    )
    return parser.parse_args()


# -------------------------
# Model Loading
# -------------------------
def load_model(args):
    want_fp16 = args.load_fp16
    torch_dtype = torch.float16 if want_fp16 else None

    if args.model_name_or_path == "meta-llama/Llama-2-7b-hf":
        base_model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    if device == "cpu":
        base_model = base_model.to(device)

    base_model.eval()
    return base_model, tokenizer, device


# -------------------------
# Watermark Components
# -------------------------
class ReweightProcessor(WatermarkBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reweight(self, seed, original_token_probs, pos_embedded_message, base): 
        """
        implementing reweight function as fig. 2 in paper
        """
        self._seed_rng(seed)
        vocab_perm = torch.randperm(self.vocab_size, device='cpu', generator=self.rng).detach().cpu().tolist()
        colorlist = torch.chunk(torch.tensor(vocab_perm), base)
        original_probs_tensor = torch.tensor([original_token_probs[tok] for tok in vocab_perm], dtype=torch.float64)

        red_tokens_alpha = 0
        red_tokens_beta = 0
        for i in range(base):
            if i < pos_embedded_message:
                red_tokens_alpha += len(colorlist[i])
            if i == pos_embedded_message:
                red_tokens_beta = red_tokens_alpha + len(colorlist[i])

        if red_tokens_alpha == 0:
            alpha = torch.tensor(0.0, dtype=torch.float64)
        else:
            alpha = original_probs_tensor.cumsum(dim=0)[red_tokens_alpha - 1]
        beta = original_probs_tensor.cumsum(dim=0)[red_tokens_beta - 1]

        acc = torch.zeros_like(original_probs_tensor, dtype=torch.float64)
        acc += original_probs_tensor.cumsum(dim=0)
        acc = torch.cat((torch.tensor([0.0], dtype=torch.float64), acc))

        if alpha >= 0.5 or beta <= 0.5:
            if alpha >= 0.5:  # 2p p 0
                a, b, c, d = 1 - beta, 1 - alpha, alpha, beta
                mapped = torch.where(
                    acc <= a, acc - d,
                    torch.where(
                        acc <= b, 2 * acc - 1,
                        torch.where(
                            acc <= c, acc - c,
                            torch.where(acc <= d, torch.zeros(1, dtype=torch.float64), acc - d),
                        ),
                    ),
                )
            else:  # beta <= 0.5, 0 p 2p
                a, b, c, d = alpha, beta, 1 - beta, 1 - alpha
                mapped = torch.where(
                    acc <= a, acc - a,
                    torch.where(
                        acc <= b,
                        torch.zeros(1, dtype=torch.float64),
                        torch.where(
                            acc <= c,
                            acc - b,
                            torch.where(acc <= d, 2 * acc - 1, acc - a),
                        ),
                    ),
                )
        else:
            if alpha <= 1 - beta <= beta <= 1 - alpha:  # alpha+beta<1 -> 0 p 2p
                a, b, c, d = alpha, 1 - beta, beta, 1 - alpha
                mapped = torch.where(
                    acc <= a, acc - a,
                    torch.where(
                        acc <= b,
                        torch.zeros(1, dtype=torch.float64),
                        torch.where(
                            acc <= c,
                            acc - b,
                            torch.where(acc <= d, 2 * acc - 1, acc - a),
                        ),
                    ),
                )
            else:  # alpha+beta>1 -> 2p p 0
                a, b, c, d = 1 - beta, alpha, 1 - alpha, beta
                mapped = torch.where(
                    acc <= a, acc - d,
                    torch.where(
                        acc <= b,
                        2 * acc - 1,
                        torch.where(
                            acc <= c,
                            acc - c,
                            torch.where(acc <= d, torch.zeros(1, dtype=torch.float64), acc - d),
                        ),
                    ),
                )

        reweighted_probs = mapped[1:] - mapped[:-1]
        combined = {k: v for k, v in zip(vocab_perm, reweighted_probs)}
        sorted_vals = torch.tensor([combined[k] for k in sorted(combined.keys())], dtype=torch.float64)
        v_non_zero = torch.where(sorted_vals > 0, sorted_vals, torch.tensor(1e-50, dtype=torch.float64))
        logits = torch.log(v_non_zero).to(dtype=torch.float32)
        return logits


class ReweightLogitsProcessor(LogitsProcessor):
    def __init__(self, reweight_processor, embedded_message, n_gram_len, R, converted_msg_length, seen_seeds=None, cache_max=50000):
        super().__init__()
        self.reweight_processor = reweight_processor
        self.n_gram_len = n_gram_len
        self.base = int(1 / R)
        self.converted_msg_length = converted_msg_length
        self.embedded_message = embedded_message
        self.output_logits = None

        self.seen_seeds = seen_seeds if seen_seeds is not None else set()
        self.is_r = False

        # cache: seed_tuple -> (vocab_perm (cpu tensor), colorlist_indices)
        self._perm_cache = {}
        self._cache_max = cache_max

    def _get_perm_and_chunks(self, seed, base, vocab_size):
        seed_tuple = tuple(seed.view(-1).tolist())
        hit = self._perm_cache.get(seed_tuple)
        if hit is not None:
            return hit

        self.reweight_processor._seed_rng(seed)
        vocab_perm = torch.randperm(vocab_size, device='cpu', generator=self.reweight_processor.rng)
        colorlist = torch.chunk(vocab_perm, base)

        if len(self._perm_cache) >= self._cache_max:
            self._perm_cache.clear()
        self._perm_cache[seed_tuple] = (vocab_perm, colorlist)
        return self._perm_cache[seed_tuple]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = scores.device
        logits = scores
        seed = input_ids[:, -self.n_gram_len:]  # [1, H]
        seed_tuple = tuple(seed.view(-1).tolist())

        # skip repeated seed (purely in-memory; no disk I/O)
        if seed_tuple in self.seen_seeds:
            # print("repeated!")
            self.is_r = True
            self.output_logits = logits
            return logits
        self.seen_seeds.add(seed_tuple)
        self.is_r = False

        # bit position from CPU RNG (deterministic)
        self.reweight_processor._seed_rng(seed)
        bit_pos = torch.randint(low=0, high=self.converted_msg_length, size=(1,), generator=self.reweight_processor.rng).item()
        # print("no r, bit_pos in generation:", bit_pos)
        pos_embedded_message = self.embedded_message[bit_pos]

        # probs on the same device
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # [V] on device

        # get vocab perm on CPU, then index on device with a moved view
        vocab_size = probs.shape[-1]
        vocab_perm_cpu, colorlist = self._get_perm_and_chunks(seed, self.base, vocab_size)
        vocab_perm = vocab_perm_cpu.to(device)

        # reorder probs by permutation (vectorized)
        original_probs_tensor = probs.index_select(dim=0, index=vocab_perm).to(torch.float64)

        # compute alpha/beta via cumsum (vectorized)
        cdf = original_probs_tensor.cumsum(dim=0)
        chunk_sizes = [len(t) for t in colorlist]
        red_alpha = sum(chunk_sizes[:pos_embedded_message])
        red_beta  = red_alpha + chunk_sizes[pos_embedded_message]

        alpha = cdf[red_alpha - 1] if red_alpha > 0 else torch.tensor(0.0, dtype=torch.float64, device=device)
        beta  = cdf[red_beta - 1]

        # build acc = [0, cdf]
        acc = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), cdf], dim=0)

        # piecewise mapping (same logic, fully tensorized on device)
        if alpha >= 0.5 or beta <= 0.5:
            if alpha >= 0.5:  # 2p p 0
                a, b, c, d = 1 - beta, 1 - alpha, alpha, beta
                z = torch.where(
                    acc <= a, acc - d,
                    torch.where(acc <= b, 2 * acc - 1,
                    torch.where(acc <= c, acc - c,
                    torch.where(acc <= d, torch.zeros_like(acc), acc - d))))
            else:  # beta <= 0.5 (0 p 2p)
                a, b, c, d = alpha, beta, 1 - beta, 1 - alpha
                z = torch.where(
                    acc <= a, acc - a,
                    torch.where(acc <= b, torch.zeros_like(acc),
                    torch.where(acc <= c, acc - b,
                    torch.where(acc <= d, 2 * acc - 1, acc - a))))
        else:
            if alpha <= 1 - beta <= beta <= 1 - alpha:  # alpha+beta<1 -> 0 p 2p
                a, b, c, d = alpha, 1 - beta, beta, 1 - alpha
                z = torch.where(
                    acc <= a, acc - a,
                    torch.where(acc <= b, torch.zeros_like(acc),
                    torch.where(acc <= c, acc - b,
                    torch.where(acc <= d, 2 * acc - 1, acc - a))))
            else:  # alpha+beta>1 -> 2p p 0
                a, b, c, d = 1 - beta, alpha, 1 - alpha, beta
                z = torch.where(
                    acc <= a, acc - d,
                    torch.where(acc <= b, 2 * acc - 1,
                    torch.where(acc <= c, acc - c,
                    torch.where(acc <= d, torch.zeros_like(acc), acc - d))))

        reweighted_probs = (z[1:] - z[:-1]).clamp_min(1e-50)  # avoid log(0)
        # map back to original vocab order
        logits_out = torch.full_like(probs, fill_value=-1e9, dtype=torch.float32)
        logits_out.index_copy_(0, vocab_perm, reweighted_probs.log().to(torch.float32))
        self.output_logits = logits_out

        return logits_out.unsqueeze(0)  # [1, V]


class DetectorProcessor(WatermarkBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_colorlist_ids(self, input_ids: torch.LongTensor, base) -> torch.LongTensor:
        self._seed_rng(input_ids)
        vocab_perm = torch.randperm(self.vocab_size, device='cpu', generator=self.rng)
        colorlist = torch.chunk(vocab_perm, base)
        return colorlist


def _compute_norm_p_val(cl_total, R):
    T_total = 0
    t_total = 0
    min_p_value = 10.0
    msg = []

    for _, value in cl_total.items():
        T = sum(value)
        if T:
            t = min(value)
            cur_msg = [i for i, v in enumerate(value) if v == t]
            msg.append(cur_msg)
            z = (t - R * T) / (math.sqrt(R * (1 - R) * T))
            cur_p_value = 1 - pow((1 - norm.cdf(z)), len(value))
            if cur_p_value < min_p_value:
                min_p_value = cur_p_value
            T_total += T
            t_total += t
        else:
            cur_msg = [int(random.choice(np.arange(len(value))))]
            msg.append(cur_msg)

    p_value = norm.cdf((t_total - R * T_total) / (math.sqrt(R * (1 - R) * T_total))) if T_total > 0 else 0.5
    return p_value, msg


# -------------------------
# Exact-length generation
# -------------------------
@torch.no_grad()
def generate_exact_n_tokens(
    model,
    tokenizer,
    inputs,                   # [1, prompt_len] on correct device
    logits_processor,  # includes watermark processor
    n_new_tokens=300,
    do_sample=True,
    temperature=1.0,
    top_k=0,           # 0 = no truncation
    eos_id=None,       # optional: for soft penalty only
    soft_eos_penalty: float = 0.0,  # e.g., 5.0; 0.0 disables
):
    device = inputs.device
    seq = inputs.clone()
    cur_input_ids = inputs
    past_key_values = None

    for step in range(n_new_tokens):
        out = model(input_ids=cur_input_ids, past_key_values=past_key_values, use_cache=True)
        logits = out.logits[:, -1, :]   # [1, V]
        past_key_values = out.past_key_values

        # Apply processors (need full seq as input_ids for seed-based logic)
        logits = logits_processor(seq, logits)

        # Optional soft penalty on EOS before last step
        if soft_eos_penalty > 0.0 and eos_id is not None and step < n_new_tokens - 1:
            logits[:, eos_id] = logits[:, eos_id] - soft_eos_penalty

        # Sampling or greedy
        if do_sample:
            if temperature != 1.0:
                logits = logits / temperature
            if top_k and top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                filtered = torch.full_like(logits, float('-inf'))
                filtered.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                probs = torch.softmax(filtered, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]

        seq = torch.cat([seq, next_token], dim=1)
        cur_input_ids = next_token  # only feed the last token next

    return seq  # [1, prompt_len + n_new_tokens]


# -------------------------
# Core Generate/Eval
# -------------------------
def generate(
    args, prompt, model, device, tokenizer,
    processor, converted_msg_length, chunk_capacity, binary_mapping,
    generation_length, n_gram_len, detector_processor, length_candi
):
    """
    processor: logits processor for generation (including reweight function)
    converted_msg_length: length of total message bits, eg., 24
    chunk_capacity: length of message bits in one chunk, eg., 1
    binary_mapping: binary representation corresponding to each message
    generation_length: length of tokens to generate
    n_gram_len: length of seed
    detector_processor: help do detection once the reponse is generated
    length_candi: the length of tokens for detection, eg., [50,100,150,200]
    """
    # Ensure prompt_max_length before tokenization
    if args.prompt_max_length is None:
        max_pos = getattr(model.config, "max_position_embeddings", 2048)
        # 预留 generation_length，避免越界
        args.prompt_max_length = max(32, max_pos - generation_length)

    tokd_input = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True,
        truncation=True, max_length=args.prompt_max_length
    ).to(device)
    inputs = tokd_input["input_ids"]
    prompt_len = inputs.shape[1]

    # Message config
    num_value = 2 ** chunk_capacity # the range of value for message at each position: [0, num_value - 1]
    R = 1.0 / num_value # corresponding red list ratio for non-watermarked text

    # Watermarked generation (exact N tokens)
    seen_seeds = set()
    embedded_message = [random.randint(0, num_value - 1) for _ in range(converted_msg_length)]
    lp = ReweightLogitsProcessor(
        processor, embedded_message=embedded_message,
        n_gram_len=n_gram_len, R=R,
        converted_msg_length=converted_msg_length,
        seen_seeds=seen_seeds
    )
    logits_processor = LogitsProcessorList([lp])

    torch.set_grad_enabled(False)
    start_t = time.time()

    # generate exactly {generation_length} tokens
    seq = generate_exact_n_tokens(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        logits_processor=logits_processor,
        n_new_tokens=generation_length,
        do_sample=args.use_sampling,
        temperature=args.sampling_temp if args.use_sampling else 1.0,
        top_k=0,
        eos_id=tokenizer.eos_token_id,
        soft_eos_penalty=0.0,  # optional soft penalty, make eos less likely to be sampled. set 0.0 to disable
    )
    total_time = time.time() - start_t

    # Prepare outputs
    full_ids = seq  # [1, prompt+gen]
    gen_ids = full_ids[0, -generation_length:].tolist()
    decoded_watermarked_output = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Detection replay (watermarked)
    bp_res = {}
    cl_total = {i: [0 for _ in range(num_value)] for i in range(converted_msg_length)}
    detector_cache = {}  # seed_tuple -> colorlist

    for step in range(generation_length):
        if step < n_gram_len:
            continue
        # slice seed in full sequence: [prompt + step - H : prompt + step]
        l = prompt_len + step - n_gram_len
        r = prompt_len + step
        cur_seed = full_ids[:, l:r]  # [1, H]
        seed_tuple = tuple(cur_seed[0].tolist())

        # deterministic bit position
        processor._seed_rng(cur_seed)
        bit_position = torch.randint(low=0, high=converted_msg_length, size=(1,), generator=processor.rng).item()

        # colorlist cache
        hit = detector_cache.get(seed_tuple)
        if hit is None:
            detector_processor._seed_rng(cur_seed)
            vocab_perm = torch.randperm(detector_processor.vocab_size, device='cpu', generator=detector_processor.rng)
            colorlist = torch.chunk(vocab_perm, num_value)
            detector_cache[seed_tuple] = colorlist
        else:
            colorlist = hit
        # print("wm bit pos, hit:", bit_position, hit)
        new_token = gen_ids[step]
        for guessed_info in range(num_value):
            if new_token in colorlist[guessed_info]:
                cl_total[bit_position][guessed_info] += 1

        cur_len = step + 1
        if cur_len in length_candi:
            bp_res[cur_len] = {k: v[:] for k, v in cl_total.items()}

    # calculate p value for each length in length_candi
    wm_res = {}
    for cur_length, cl_res in bp_res.items():
        wmp, msg = _compute_norm_p_val(cl_res, R)
        total_correct_bits = 0
        for pos in range(converted_msg_length):
            correct_bits = 0
            for pos_extracted_msg in msg[pos]:
                pos_bin_extracted = binary_mapping[pos_extracted_msg]
                pos_bin_true = binary_mapping[embedded_message[pos]]
                hamming = hamming_distance(pos_bin_true, pos_bin_extracted)
                correct_bits += chunk_capacity - hamming
            num_msg = max(1, len(msg[pos]))
            total_correct_bits += (correct_bits / num_msg)
        wm_res[cur_length] = {"wmp": -wmp, "correct_bits": total_correct_bits}

    # Baseline generation (no watermark), same exact-N decoding
    seq_baseline = generate_exact_n_tokens(
        model=model,
        tokenizer=tokenizer,
        inputs=tokd_input["input_ids"],
        logits_processor=LogitsProcessorList([]),  # no watermark
        n_new_tokens=generation_length,
        do_sample=args.use_sampling,
        temperature=args.sampling_temp if args.use_sampling else 1.0,
        top_k=0,
        eos_id=tokenizer.eos_token_id,
        soft_eos_penalty=0.0,  
    )
    baseline_seq = seq_baseline
    baseline_gen = baseline_seq[0, -generation_length:].tolist()

    # replay detection on baseline
    bp_res = {}
    cl_total = {i: [0 for _ in range(num_value)] for i in range(converted_msg_length)}
    detector_cache.clear()

    for step in range(generation_length):
        if step < n_gram_len:
            continue
        l = prompt_len + step - n_gram_len
        r = prompt_len + step
        cur_seed = baseline_seq[:, l:r]
        seed_tuple = tuple(cur_seed[0].tolist())

        processor._seed_rng(cur_seed)
        bit_position = torch.randint(low=0, high=converted_msg_length, size=(1,), generator=processor.rng).item()

        hit = detector_cache.get(seed_tuple)
        # print("nonwm bit pos, hit:", bit_position, hit)
        if hit is None:
            detector_processor._seed_rng(cur_seed)
            vocab_perm = torch.randperm(detector_processor.vocab_size, device='cpu', generator=detector_processor.rng)
            colorlist = torch.chunk(vocab_perm, num_value)
            detector_cache[seed_tuple] = colorlist
        else:
            colorlist = hit

        new_token = baseline_gen[step]
        for guessed_info in range(num_value):
            if new_token in colorlist[guessed_info]:
                cl_total[bit_position][guessed_info] += 1

        cur_len = step + 1
        if cur_len in length_candi:
            bp_res[cur_len] = {k: v[:] for k, v in cl_total.items()}

    nonwm_res = {cur_length: {"nonwmp": -_compute_norm_p_val(cl_res, R)[0]} for cur_length, cl_res in bp_res.items()}

    return decoded_watermarked_output, gen_ids, tokenizer.decode(baseline_gen, skip_special_tokens=True), embedded_message, wm_res, nonwm_res

def load_prompts(json_path, nsamples):
    """
    choose first n samples from a dataset
    """
    with open(json_path, "r") as f:
        all_prompts = json.load(f)

    prompts = all_prompts[:nsamples]
    print(f"Extracted first {len(prompts)} prompts from {len(all_prompts)}")
    return prompts

def main(
    args,
    temp,
    start,
    generation_num,
    capacity,
    msg_len,
    generation_length,
    n_gram_len,
    prompts_fp,
    out_dir,
):
    """
    temp: temperature
    start: start of prompt index
    generation_num: number of generation
    capacity: capacity for each chunk, eg., 1
    msg_len: number of chuncks, eg., 24
    generation_length: length of tokens for each generation
    n_gram_len: length of seed
    """
    model, tokenizer, device = load_model(args)

    # the number of message positions = msg_len / bits_per_symbol
    converted_msg_length = int(msg_len / capacity)

    # binary representation corresponding to each message
    num_value = 2 ** capacity
    max_len_bits = len(bin(num_value - 1)[2:])
    binary_mapping = {gamma: bin(gamma)[2:].zfill(max_len_bits) for gamma in range(num_value)}

    # initialize processor 
    vocab = list(tokenizer.get_vocab().values())
    reweight_processor = ReweightProcessor(vocab=vocab) # for reweight function
    detector_processor = DetectorProcessor(vocab=vocab) # for detection

    length_candi = np.arange(50, generation_length+50, 50)  # the length of tokens for detection, eg., [50,100,150,200]
    # record the results (p values, number of correct bits) for each generation length in length_candi
    nonwm_pvals_dict = {L: [] for L in length_candi}
    wm_pvals_dict = {L: [] for L in length_candi}
    wm_cb_dict = {L: 0 for L in length_candi} 

    # IO setup for path to store generated responses
    os.makedirs(out_dir, exist_ok=True)
    out_json_path = os.path.join(out_dir, f"c{capacity}_{msg_len}_res{start+1}_{start+generation_num}_1000.json")

    # load {generation_num} prompts
    prompts = load_prompts(json_path=promts_fp, nsamples=generation_num)
    with open(out_json_path, "a", encoding="utf-8") as fw_wm:
        for i in range(start, start + generation_num):
            t1 = time.time()

            # prompt
            # input_text = ("Earthquake research shows quakes cluster along specific belts rather than uniformly "
            #     "across the globe. These concentrations are called seismic belts.")

            input_text = prompts[i]
            decoded_wm, token_ids, decoded_nonwm, embedded_message, wm_res, nonwm_res = generate(
                args=args,
                prompt=input_text,
                model=model,
                device=device,
                tokenizer=tokenizer,
                processor=reweight_processor,
                converted_msg_length=converted_msg_length,
                chunk_capacity=capacity,
                binary_mapping=binary_mapping,
                generation_length=generation_length,
                n_gram_len=n_gram_len,
                detector_processor=detector_processor,
                length_candi=length_candi
            )

            length = len(token_ids)
            res = [capacity, length, embedded_message, token_ids, [decoded_wm]]
            fw_wm.write(json.dumps(res) + "\n")

            for L in length_candi:
                nonwm_pvals_dict[L].append(nonwm_res[L]["nonwmp"].item())
                wm_pvals_dict[L].append(wm_res[L]["wmp"].item())
                wm_cb_dict[L] += wm_res[L]["correct_bits"]

            print(f"{i+1}: length={length}")
            print(f"elapsed: {time.time() - t1:.3f}s\n")


    for L in length_candi:
        print(f"\n************** length={L} **************")
        nonwm_pvals = nonwm_pvals_dict[L]
        wm_pvals = wm_pvals_dict[L]
        print("nonwm_pvals = ", [-x for x in nonwm_pvals])
        print("wm_pvals    = ", [-x for x in wm_pvals])

        bit_acc = wm_cb_dict[L] / (generation_num * msg_len)
        print(f"bit_acc: {bit_acc:.6f}")

        preds = nonwm_pvals + wm_pvals
        t_labels = [0] * generation_num + [1] * generation_num

        fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr)) / 2)

        def _tpr_at(fpr_thresh):
            idx = np.where(fpr < fpr_thresh)[0]
            return tpr[idx[-1]] if idx.size > 0 else float("nan")

        low0 = _tpr_at(1e-4)
        low1 = _tpr_at(1e-3)
        low2 = _tpr_at(1e-2)
        low3 = _tpr_at(1e-1)

        print(f"auc : {auc:.6f}")
        print(f"acc : {acc:.6f}")
        print(f"TPR@FPR<1e-4: {low0}")
        print(f"TPR@FPR<1e-3: {low1}")
        print(f"TPR@FPR<1e-2: {low2}")
        print(f"TPR@FPR<1e-1: {low3}")


if __name__ == "__main__":
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # torch.set_grad_enabled(False)

    args = parse_args()
    capacity = args.chunk_capacity        # bits per symbol
    msg_len = args.msg_len        # total embedded bits
    start = args.start           # start index of prompts file
    generation_num = args.generation_num  # generate 400 responses
    generation_length = args.generation_length # number of tokens in each response
    out_dir = args.out_dir # directory to store responses
    prompts_fp = args.prompts_fp
    
    temp = args.sampling_temp # temperature
    n_gram_len = 3 # length of seed

    s = time.time()
    main(
        args=args,
        temp=temp,
        start=start,
        generation_num=generation_num,
        capacity=capacity,
        msg_len=msg_len,
        generation_length=generation_length,
        n_gram_len=n_gram_len,
        out_dir=out_dir
        prompts_fp=prompts_fp
    )
    e = time.time()
    print("total hours:", (e - s) / 3600.0)
