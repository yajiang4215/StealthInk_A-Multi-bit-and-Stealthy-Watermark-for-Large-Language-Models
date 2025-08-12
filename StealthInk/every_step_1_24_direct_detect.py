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
    AutoModelForCausalLM,
    AutoTokenizer,
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
        default="facebook/opt-1.3b",
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
    elif args.model_name_or_path == "facebook/opt-1.3b":
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        # Fallback: try generic auto classes
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
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

    def seed_rng(self, seed):
        self._seed_rng(seed)

    def get_initial_probs(self, logits):
        probs = torch.softmax(logits, dim=-1).detach().squeeze(0)  # [V]
        token_ids = torch.arange(self.vocab_size).tolist()
        return {tid: p.item() for tid, p in zip(token_ids, probs)}

    def reweight(self, seed, original_token_probs, gamma_index, base):
        self.seed_rng(seed)
        vocab_perm = torch.randperm(self.vocab_size, device=seed.device, generator=self.rng).detach().cpu().tolist()

        original_probs_tensor = torch.tensor([original_token_probs[tok] for tok in vocab_perm], dtype=torch.float64)
        colorlist = torch.chunk(torch.tensor(vocab_perm), base)

        red_tokens_alpha = 0
        red_tokens_beta = 0
        for i in range(base):
            if i < gamma_index:
                red_tokens_alpha += len(colorlist[i])
            if i == gamma_index:
                red_tokens_beta = red_tokens_alpha + len(colorlist[i])

        if red_tokens_alpha == 0:
            alpha = torch.tensor(0.0, dtype=torch.float64)
        else:
            alpha = original_probs_tensor.cumsum(dim=0)[red_tokens_alpha - 1]
        beta = original_probs_tensor.cumsum(dim=0)[red_tokens_beta - 1]

        acc = torch.zeros_like(original_probs_tensor, dtype=torch.float64)
        acc += original_probs_tensor.cumsum(dim=0)
        acc = torch.cat((torch.tensor([0.0], dtype=torch.float64), acc))

        # piecewise mapping
        if alpha >= 0.5 or beta <= 0.5:
            if alpha >= 0.5:  # 2p p 0
                a, b, c, d = 1 - beta, 1 - alpha, alpha, beta
                mapped = torch.where(
                    acc <= a,
                    acc - d,
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
            else:  # beta <= 0.5, 0 p 2p
                a, b, c, d = alpha, beta, 1 - beta, 1 - alpha
                mapped = torch.where(
                    acc <= a,
                    acc - a,
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
            # alpha < 0.5 and beta > 0.5
            if alpha <= 1 - beta <= beta <= 1 - alpha:  # alpha+beta<1 -> 0 p 2p
                a, b, c, d = alpha, 1 - beta, beta, 1 - alpha
                mapped = torch.where(
                    acc <= a,
                    acc - a,
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
                    acc <= a,
                    acc - d,
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

        # Avoid log(0)
        v_non_zero = torch.where(sorted_vals > 0, sorted_vals, torch.tensor(1e-50, dtype=torch.float64))
        logits = torch.log(v_non_zero).to(dtype=torch.float32)  # model expects float32/16
        return logits


def is_repeated(seed_str, filename):
    # maintain a lightweight set persisted on disk
    existing = set()
    try:
        with open(filename, "r") as f:
            for line in f:
                existing.add(line.strip())
    except FileNotFoundError:
        pass
    if seed_str not in existing:
        with open(filename, "a") as f:
            f.write(seed_str + "\n")
        return False
    return True


class ReweightLogitsProcessor(LogitsProcessor):
    def __init__(self, reweight_processor, gamma_indices, register_log_fp, n_gram, R, converted_msg_length):
        super().__init__()
        self.reweight_processor = reweight_processor
        self.fp = register_log_fp
        self.n_gram = n_gram
        self.R = R
        self.base = int(1 / R)
        self.converted_msg_length = converted_msg_length
        self.gamma_indices = gamma_indices
        self.is_r = False
        self.original_logits = None
        self.output_logits = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = scores
        self.original_logits = logits.clone()
        seed = input_ids[:, -self.n_gram:]

        seed_scalar = torch.sum(seed).item()
        random.seed(seed_scalar)
        bit_pos = random.randint(1, self.converted_msg_length) - 1
        gamma_index = self.gamma_indices[bit_pos]

        seed_string = f"{bit_pos}{seed.tolist()}"
        if is_repeated(seed_string, self.fp):
            self.is_r = True
            self.output_logits = logits
            return logits

        self.is_r = False
        init_probs = self.reweight_processor.get_initial_probs(logits)
        reweighted = self.reweight_processor.reweight(seed, init_probs, gamma_index, self.base)
        self.output_logits = reweighted
        return reweighted.unsqueeze(0).to(scores.device)

    def get_original_logits(self):
        if self.original_logits is None:
            raise RuntimeError("Original logits have not been computed. Call __call__ first.")
        return self.original_logits


class DetectorProcessor(WatermarkBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_colorlist_ids(self, input_ids: torch.LongTensor, base) -> torch.LongTensor:
        self._seed_rng(input_ids)
        vocab_perm = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        colorlist = torch.chunk(vocab_perm, base)
        return colorlist

    def _compute_binom_p_val(self, cl_total, R):
        T = sum(cl_total)
        p_vals = []
        for observed in cl_total:
            p_vals.append(binom.cdf(observed, T, R))
        return p_vals


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
            # If no counts, pick a random guess
            cur_msg = [int(random.choice(np.arange(len(value))))]
            msg.append(cur_msg)

    p_value = norm.cdf((t_total - R * T_total) / (math.sqrt(R * (1 - R) * T_total))) if T_total > 0 else 0.5
    return p_value, msg


# -------------------------
# Core Generate/Eval
# -------------------------
def generate(
    prompt,
    converted_msg_length,
    args,
    num_value,
    R,
    generation_length,
    n_gram,
    log_fp,
    detector_processor,
    length_candi,
    model,
    device,
    tokenizer,
    processor,
):
    # local capacity from num_value
    capacity_local = int(math.log2(num_value))

    gamma_indices = [random.randint(0, num_value - 1) for _ in range(converted_msg_length)]
    lp = ReweightLogitsProcessor(processor, gamma_indices, log_fp, n_gram, R, converted_msg_length)

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    inputs = tokd_input["input_ids"]

    max_len_bits = len(bin(num_value - 1)[2:])
    gammas = [i for i in range(num_value)]
    binary_gammas = {gamma: bin(gamma)[2:].zfill(max_len_bits) for gamma in gammas}

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    if args.use_sampling:
        gen_kwargs.update(dict(do_sample=True, top_k=0, temperature=args.sampling_temp))
    else:
        gen_kwargs.update(dict(num_beams=args.n_beams))

    if args.prompt_max_length is None:
        if hasattr(model.config, "max_position_embeddings"):
            args.prompt_max_length = model.config.max_position_embeddings - args.max_new_tokens
        else:
            args.prompt_max_length = 2048 - args.max_new_tokens

    logits_processor = LogitsProcessorList([lp])

    generate_with_watermark = partial(model.generate, logits_processor=logits_processor, **gen_kwargs)

    bp_res = {}
    wm_res = {}

    with torch.no_grad():
        cl_total = {i: [0 for _ in range(len(gammas))] for i in range(converted_msg_length)}
        seen_seed = set()

        for generation_step in range(generation_length):
            output_with_watermark = generate_with_watermark(inputs)

            if generation_step >= n_gram:
                seed = inputs[:, -n_gram:]
                random.seed(torch.sum(inputs[:, -n_gram:]).item())
                bit_position = random.randint(1, converted_msg_length) - 1

                seed_string = f"{bit_position}{output_with_watermark[:, -n_gram:].tolist()}"
                if not lp.is_r and seed_string not in seen_seed:
                    seen_seed.add(seed_string)
                    new_token = output_with_watermark[0, -1]

                    detector_processor._seed_rng(inputs)
                    vocab_permutation = torch.randperm(detector_processor.vocab_size, device=inputs.device, generator=detector_processor.rng)
                    colorlist = torch.chunk(vocab_permutation, num_value)

                    for guessed_info in range(num_value):
                        if new_token in colorlist[guessed_info]:
                            cl_total[bit_position][guessed_info] += 1

                if (generation_step + 1) in length_candi:
                    bp_res[generation_step + 1] = {k: v[:] for k, v in cl_total.items()}

            inputs = output_with_watermark

        for cur_length, cl_res in bp_res.items():
            wmp, msg = _compute_norm_p_val(cl_res, R)
            total_correct_bits = 0
            for pos in range(converted_msg_length):
                correct_bits = 0
                for extracted_msg in msg[pos]:
                    bin_extracted_msg = binary_gammas[extracted_msg]
                    bin_true_msg = binary_gammas[gamma_indices[pos]]
                    hamming = hamming_distance(bin_true_msg, bin_extracted_msg)
                    correct_bits += capacity_local - hamming
                num_msg = max(1, len(msg[pos]))
                correct_bits /= num_msg
                total_correct_bits += correct_bits

            wm_res[cur_length] = {"wmp": -wmp, "correct_bits": total_correct_bits}

        output_tokens = output_with_watermark[0][-generation_length:].tolist()
        decoded_output = tokenizer.decode(output_tokens, skip_special_tokens=True)

    # Non-watermarked baseline
    generate_without_watermark = partial(model.generate, **gen_kwargs)

    bp_res = {}
    nonwm_res = {}

    with torch.no_grad():
        cl_total = {i: [0 for _ in range(len(gammas))] for i in range(converted_msg_length)}
        seen_seed = set()

        for generation_step in range(generation_length):
            output_without_watermark = generate_without_watermark(inputs)

            if generation_step >= n_gram:
                random.seed(torch.sum(inputs[:, -n_gram:]).item())
                bit_position = random.randint(1, converted_msg_length) - 1
                seed_string = f"{bit_position}{inputs[:, -n_gram:].tolist()}"

                if seed_string not in seen_seed:
                    seen_seed.add(seed_string)
                    new_token = output_without_watermark[0, -1]

                    detector_processor._seed_rng(inputs)
                    vocab_permutation = torch.randperm(detector_processor.vocab_size, device=inputs.device, generator=detector_processor.rng)
                    colorlist = torch.chunk(vocab_permutation, num_value)

                    for guessed_info in range(num_value):
                        if new_token in colorlist[guessed_info]:
                            cl_total[bit_position][guessed_info] += 1

                if (generation_step + 1) in length_candi:
                    bp_res[generation_step + 1] = {k: v[:] for k, v in cl_total.items()}

            inputs = output_without_watermark

        for cur_length, cl_res in bp_res.items():
            nonwmp, _ = _compute_norm_p_val(cl_res, R)
            nonwm_res[cur_length] = {"nonwmp": -nonwmp}

        print("\n[watermarked sample]")
        print(decoded_output)
        print("\n[non-watermarked sample]")
        print(tokenizer.decode(output_without_watermark[0][-generation_length:].tolist(), skip_special_tokens=True))

    return decoded_output, output_tokens, gamma_indices, wm_res, nonwm_res


# -------------------------
# Driver
# -------------------------
def main(
    args,
    temp,
    start,
    generation_num,
    capacity,
    msg_len,
    generation_length,
    register_log_path,
    n_gram,
):
    model, tokenizer, device = load_model(args)

    num_value = 2 ** capacity
    R = 1 / num_value
    round_number_R = count_decimal_digits(Decimal(str(R)))

    # message positions = msg_len / bits_per_symbol
    converted_msg_length = int(msg_len / capacity)

    vocab = list(tokenizer.get_vocab().values())
    reweight_processor = ReweightProcessor(vocab)
    detector_processor = DetectorProcessor(vocab=vocab)

    length_candi = np.arange(50, 250, 50)  # {50,100,150,200}
    nonwm_pvals_dict = {L: [] for L in length_candi}
    wm_pvals_dict = {L: [] for L in length_candi}
    wm_cb_dict = {L: 0 for L in length_candi}

    # IO setup
    os.makedirs(register_log_path, exist_ok=True)
    out_dir = os.path.join("out", f"temp{temp}", "multi", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_json_path = os.path.join(out_dir, f"c{capacity}_{msg_len}_res{start+1}_{start+generation_num}_1000.json")

    with open(out_json_path, "a", encoding="utf-8") as fw_wm:
        for i in range(start, start + generation_num):
            t1 = time.time()

            # You may customize or sample prompts; keep a deterministic default for stability
            input_text = (
                "Earthquake research shows quakes cluster along specific belts rather than uniformly "
                "across the globe. These concentrations are called seismic belts."
            )

            register_log = os.path.join(register_log_path, f"c{capacity}_{msg_len}_register_log_{i+1}")

            decoded_wm, token_ids, gamma_indices, wm_res, nonwm_res = generate(
                prompt=input_text,
                converted_msg_length=converted_msg_length,
                args=args,
                num_value=num_value,
                R=R,
                generation_length=generation_length,
                n_gram=n_gram,
                log_fp=register_log,
                detector_processor=detector_processor,
                length_candi=length_candi,
                model=model,
                device=device,
                tokenizer=tokenizer,
                processor=reweight_processor,
            )

            length = len(token_ids)
            res = [capacity, length, gamma_indices, token_ids, [decoded_wm]]
            fw_wm.write(json.dumps(res) + "\n")

            for L in length_candi:
                nonwm_pvals_dict[L].append(nonwm_res[L]["nonwmp"])
                wm_pvals_dict[L].append(wm_res[L]["wmp"])
                wm_cb_dict[L] += wm_res[L]["correct_bits"]

            print(f"{i+1}: length={length}")
            print(f"elapsed: {time.time() - t1:.3f}s\n")

    # summary metrics
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
    args = parse_args()
    # sensible defaults (you can override via CLI)
    args.model_name_or_path = "meta-llama/Llama-2-7b-hf"
    args.sampling_temp = 1.0
    args.max_new_tokens = 1

    capacity = 1        # bits per symbol
    msg_len = 24        # total embedded bits
    start = 0
    generation_num = 10 # reduced for quick run; adjust as needed
    generation_length = 300
    temp = 1
    register_log_path = os.path.join("out", f"temp{temp}", "multi", "log")
    n_gram = 3

    s = time.time()
    main(
        args=args,
        temp=temp,
        start=start,
        generation_num=generation_num,
        capacity=capacity,
        msg_len=msg_len,
        generation_length=generation_length,
        register_log_path=register_log_path,
        n_gram=n_gram,
    )
    e = time.time()
    print("total hours:", (e - s) / 3600.0)
