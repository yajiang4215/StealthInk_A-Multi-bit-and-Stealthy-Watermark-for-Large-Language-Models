# Codes for paper StealthInk: A Multi-bit and Stealthy Watermark for Large Language Models (ICML 2025)

## Link: https://arxiv.org/abs/2506.05502

## Usage

Generate 400 responses for 400 prompts. For each response, embed 24 bits in 200 tokens and do detection and decoding for the watermark.

```python3 every_step_1_24_direct_detect.py --chunk_capacity 1 --msg_len 24 --start 0 --generation_num 400 --generation_length 200 --out_dir "output" --sampling_temp 1.0 ```

capacity: bits per symbol

msg_len: total embedded bits

start: start index of prompts file

generation_num: the number of generated responses

generation_length: the number of tokens in each response

sampling_temp: temperature in generation

out_dir: directory to store responses

## Citation
```bibtex
@inproceedings{jiang2025stealthink,
  title   = {StealthInk: A Multi-bit and Stealthy Watermark for Large Language Models},
  author  = {Jiang, Ya and Wu, Chuxiong and Kordi Boroujeny, Massieh and Mark, Brian and Zeng, Kai},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  year    = {2025}
}
