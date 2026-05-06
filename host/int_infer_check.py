#!/usr/bin/env python3
"""
int_infer_check.py — exact integer simulation of the FPGA MLP engine.

Mirrors mlp_engine.v arithmetic exactly:
  - INT8 weights packed 4-per-word (little-endian)
  - INT32 biases (one per word)
  - ACC += bias + sum(w_i * x_i)  in pure Python int (no float)
  - After FC1: act = relu_clip(acc >> FC1_REQ_SHIFT)
  - After FC2: act = relu_clip(acc >> FC2_REQ_SHIFT)
  - FC3: argmax of raw INT32 logits

Usage:
    python tools/int_infer_check.py <image_path>

Example:
    python tools/int_infer_check.py test_images/MNIST/png/mnist_12_label_9.png
"""

import re
import sys
from pathlib import Path
from PIL import Image, ImageOps

HEADER = Path(__file__).parent.parent / "hardware" / "vitis" / "firmware" / "mlp_accelerator_app" / "include" / "weights_biases.h"

INPUT_MEAN  = 0.1307
INPUT_STD   = 0.3081
INPUT_SCALE = 0.02221643
FC1_REQ_SHIFT = 11
FC2_REQ_SHIFT = 8


# ── Parse weights_biases.h ────────────────────────────────────────────────────

def _parse_uint32_array(text: str, name: str) -> list[int]:
    m = re.search(
        rf'static const uint32_t\s+{name}\s*\[.*?\]\s*=\s*\{{(.*?)\}};',
        text, re.DOTALL)
    if not m:
        raise ValueError(f"Could not find array '{name}' in header")
    return [int(v, 16) for v in re.findall(r'0x([0-9A-Fa-f]+)U', m.group(1))]


def _parse_int32_array(text: str, name: str) -> list[int]:
    m = re.search(
        rf'static const int32_t\s+{name}\s*\[.*?\]\s*=\s*\{{(.*?)\}};',
        text, re.DOTALL)
    if not m:
        raise ValueError(f"Could not find array '{name}' in header")
    return [int(v) for v in re.findall(r'-?\d+', m.group(1))]


def load_params():
    text = HEADER.read_text()
    return {
        'fc1_w': _parse_uint32_array(text, 'fc1_weights'),
        'fc1_b': _parse_int32_array(text,  'fc1_bias'),
        'fc2_w': _parse_uint32_array(text, 'fc2_weights'),
        'fc2_b': _parse_int32_array(text,  'fc2_bias'),
        'fc3_w': _parse_uint32_array(text, 'fc3_weights'),
        'fc3_b': _parse_int32_array(text,  'fc3_bias'),
    }


# Input quantization (matches firmware mlp_bram_init.c) 
def quantize_input(pixels: bytes) -> list[int]:
    q = []
    for p in pixels:
        x = (p / 255.0 - INPUT_MEAN) / INPUT_STD
        xq = x / INPUT_SCALE
        xq = int(xq + 0.5) if xq >= 0 else int(xq - 0.5)
        q.append(max(-128, min(127, xq)))
    return q


# Signed byte extraction from uint32_t word
def s8(v: int) -> int:
    v &= 0xFF
    return v if v < 128 else v - 256


# relu_clip (matches mlp_engine.v relu_clip function)
def relu_clip(x: int) -> int:
    if x <= 0:
        return 0
    if x > 127:
        return 127
    return x


# Integer arithmetic right shift
def arith_rshift(x: int, k: int) -> int:
    if x >= 0:
        return x >> k
    # Python right-shift already arithmetic for negative ints
    return x >> k


# FC layer (mirrors FSM: BIAS_LOAD → W_ADDR/LATCH/ACC loop  SAVE)

def fc_layer(w_words: list[int], b_vals: list[int],
             x_in: list[int], n_out: int, n_in: int,
             req_shift: int, do_relu: bool,
             layer_name: str, verbose: bool) -> list[int]:
    acts = []
    words_per_neuron = n_in // 4

    for j in range(n_out):
        acc = b_vals[j]  # INT32 bias (matches S_BIAS_LOAD)

        for i in range(words_per_neuron):
            word_idx = j * words_per_neuron + i
            w = w_words[word_idx]
            # Match S_W_LATCH: w_latch0 = param_rdata[7:0], etc.
            w0 = s8(w & 0xFF)
            w1 = s8((w >> 8)  & 0xFF)
            w2 = s8((w >> 16) & 0xFF)
            w3 = s8((w >> 24) & 0xFF)

            x0 = x_in[i*4]
            x1 = x_in[i*4 + 1]
            x2 = x_in[i*4 + 2]
            x3 = x_in[i*4 + 3]

            # Match S_W_ACC: acc += mac4_sum
            acc += w0*x0 + w1*x1 + w2*x2 + w3*x3

        if do_relu:
            act = relu_clip(arith_rshift(acc, req_shift))
        else:
            act = acc  # raw logit for FC3

        acts.append(act)

        if verbose and j < 10:
            print(f"  [{layer_name}] neuron {j:2d}: acc={acc:10d}  "
                  f"{'>> '+str(req_shift)+' = '+str(arith_rshift(acc, req_shift)) if do_relu else ''}"
                  f"  out={act}")

    return acts


# Main
def run(image_path: str, invert: bool = False, verbose: bool = True):
    params = load_params()

    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    if invert:
        img = ImageOps.invert(img)
    pixels = img.tobytes()

    q_input = quantize_input(pixels)

    if verbose:
        print(f"Image: {image_path}")
        print(f"Input word[0] = 0x{(q_input[0]&0xFF)|((q_input[1]&0xFF)<<8)|((q_input[2]&0xFF)<<16)|((q_input[3]&0xFF)<<24):08X}")
        print(f"Input word[60] = 0x{(q_input[240]&0xFF)|((q_input[241]&0xFF)<<8)|((q_input[242]&0xFF)<<16)|((q_input[243]&0xFF)<<24):08X}")
        print()

    print("── FC1 (first 10 neurons) ──")
    act1 = fc_layer(params['fc1_w'], params['fc1_b'], q_input,
                    64, 784, FC1_REQ_SHIFT, True, 'FC1', verbose)

    print(f"\nFC1 activations (first 16): {act1[:16]}")

    print("\n── FC2 (first 10 neurons) ──")
    act2 = fc_layer(params['fc2_w'], params['fc2_b'], act1,
                    32, 64, FC2_REQ_SHIFT, True, 'FC2', verbose)

    print(f"\nFC2 activations (first 16): {act2[:16]}")

    print("\n── FC3 logits ──")
    logits = fc_layer(params['fc3_w'], params['fc3_b'], act2,
                      10, 32, 0, False, 'FC3', verbose)

    print(f"\nFC3 logits: {logits}")
    pred = logits.index(max(logits))
    print(f"\n>>> Integer simulation predicts class: {pred}")
    return pred


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()
    run(args.image, invert=args.invert, verbose=not args.quiet)
