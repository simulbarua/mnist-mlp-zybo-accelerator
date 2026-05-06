#!/usr/bin/env python3
"""Send one image to the Zybo board over UART for MLP inference.

Frame format:
    b"IMG1" + 784 grayscale bytes (28x28, row-major)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial
from PIL import Image, ImageOps


FRAME_MAGIC = b"IMG1"
IMAGE_SIDE = 28
IMAGE_PIXELS = IMAGE_SIDE * IMAGE_SIDE
MLP_INPUT_MEAN = 0.1307
MLP_INPUT_STD = 0.3081
MLP_INPUT_SCALE = 0.02221643
SAMPLE_WORDS = [0, 1, 2, 3, 4, 5, 6, 7, 40, 60, 80, 100, 140, 180, 195]


def load_image_bytes(image_path: Path, invert: bool) -> bytes:
    image = Image.open(image_path).convert("L")
    image = ImageOps.fit(image, (IMAGE_SIDE, IMAGE_SIDE), method=Image.Resampling.LANCZOS)
    if invert:
        image = ImageOps.invert(image)
    data = image.tobytes()
    if len(data) != IMAGE_PIXELS:
        raise ValueError(f"expected {IMAGE_PIXELS} bytes, got {len(data)}")
    return data


def quantize_payload(payload: bytes) -> bytes:
    out = bytearray(len(payload))

    for i, pixel in enumerate(payload):
        x = ((pixel / 255.0) - MLP_INPUT_MEAN) / MLP_INPUT_STD
        x_scaled = x / MLP_INPUT_SCALE
        xq = int(x_scaled + 0.5) if x_scaled >= 0.0 else int(x_scaled - 0.5)
        if xq < -128:
            xq = -128
        elif xq > 127:
            xq = 127
        out[i] = xq & 0xFF

    return bytes(out)


def read_until_ready(port: serial.Serial, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    line = bytearray()

    while time.time() < deadline:
        ch = port.read(1)
        if not ch:
            continue
        if ch == b"\n":
            text = line.decode("utf-8", errors="replace").strip()
            # if text:
            #     print(f"[board] {text}")
            if text == "READY":
                return
            line.clear()
        elif ch != b"\r":
            line.extend(ch)

    raise TimeoutError("timed out waiting for READY from board")


def read_result_lines(port: serial.Serial, timeout_s: float) -> str:
    deadline = time.time() + timeout_s
    line = bytearray()

    while time.time() < deadline:
        ch = port.read(1)
        if not ch:
            continue
        if ch == b"\n":
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                print(f"[board] {text}")
                if text.startswith("RESULT "):
                    return text
            line.clear()
        elif ch != b"\r":
            line.extend(ch)

    raise TimeoutError("timed out waiting for RESULT from board")

def dump_packed_words(title: str, payload: bytes, sample_words: list[int] | None = None) -> None:
    total_words = len(payload) // 4
    # print(title)
    indices = sample_words if sample_words is not None else list(range(min(8, total_words)))
    for i in indices:
        if i >= total_words:
            continue
        w = (
            payload[4*i + 0]
            | (payload[4*i + 1] << 8)
            | (payload[4*i + 2] << 16)
            | (payload[4*i + 3] << 24)
        )
        # print(f"word[{i:03d}] = 0x{w:08X}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a 28x28 image frame to the Zybo board.")
    parser.add_argument("image", type=Path, help="path to the image file")
    parser.add_argument("--port", required=True, help="serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200, help="UART baud rate")
    parser.add_argument("--invert", action="store_true", help="invert grayscale before sending")
    parser.add_argument("--no-wait-ready", action="store_true", help="send immediately without waiting for READY")
    parser.add_argument("--timeout", type=float, default=10.0, help="serial read timeout in seconds")
    args = parser.parse_args()

    payload = load_image_bytes(args.image, invert=args.invert)
    quantized = quantize_payload(payload)
    dump_packed_words("raw packed words:", payload, SAMPLE_WORDS)
    dump_packed_words("quantized packed words:", quantized, SAMPLE_WORDS)


    with serial.Serial(args.port, args.baud, timeout=0.2) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()

        if not args.no_wait_ready:
            read_until_ready(port, args.timeout)

        port.write(FRAME_MAGIC + payload)
        port.flush()
        print(f"sent {len(payload)} image bytes from {args.image}")

        result = read_result_lines(port, args.timeout)
        print(f"[host] {args.image.name}: {result}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
