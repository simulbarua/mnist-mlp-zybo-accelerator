# Design Notes — MNIST MLP Zybo Accelerator

## Contents

- [System overview](#system-overview)
- [1. Network architecture](#1-network-architecture)
- [2. Training and quantization](#2-training-and-quantization)
  - [2.1 Float training](#21-float-training)
  - [2.2 Post-training quantization](#22-post-training-quantization-ptq)
  - [2.3 Input quantization](#23-input-quantization)
  - [2.4 Inter-layer requantization](#24-inter-layer-requantization)
  - [2.5 Exported artifacts](#25-exported-artifacts)
- [3. Hardware design](#3-hardware-design)
  - [3.1 Block design](#31-block-design)
  - [3.2 AXI address map](#32-axi-address-map)
  - [3.3 RTL modules](#33-rtl-modules)
  - [3.4 BRAM memory layout](#34-bram-memory-layout-param-bram)
  - [3.5 Timing](#35-timing)
- [4. Firmware](#4-firmware)
  - [4.1 Boot sequence](#41-boot-sequence-mainc)
  - [4.2 UART protocol](#42-uart-protocol)
  - [4.3 Input normalization](#43-input-normalization-ps-side)
  - [4.4 Inference handshake](#44-inference-handshake)
- [5. Host scripts](#5-host-scripts)
- [6. Retrain workflow](#6-retrain-workflow)

---

## System overview

The system classifies 28×28 MNIST digits using a three-layer fully-connected (MLP) network accelerated in the Zybo Z7-10 FPGA fabric. The ARM Cortex-A9 (PS) handles UART communication and BRAM loading; the PL fabric runs the fixed-point MAC engine.

```
PC ──UART──► PS (ARM Cortex-A9)
                │  load weights → Param BRAM (once at boot)
                │  write image  → Input BRAM (per inference)
                │  write 0x1    → AXI-Lite CTRL
                ▼
            PL Fabric (MLP Engine @ 100 MHz)
                │  reads Param BRAM + Input BRAM
                │  runs FC1 → FC2 → FC3 → argmax
                ▼
            AXI-Lite STATUS/RESULT ──► PS ──UART──► PC
```

---

## 1. Network architecture

| Layer | Operation | Input | Output | Activation |
|---|---|---|---|---|
| FC1 | Linear | 784 | 64 | ReLU |
| FC2 | Linear | 64 | 32 | ReLU |
| FC3 | Linear | 32 | 10 | argmax |

- Trained with **dropout p=0.2** on hidden layers (disabled at inference)
- Loss: cross-entropy; optimizer: Adam (lr=1e-3, StepLR ×0.1 at epoch 10)
- 20 epochs on MNIST 60k train split; best checkpoint saved by test accuracy
- Target: ≥ 95% float accuracy, ≥ 94% after quantization

---

## 2. Training and quantization

### 2.1 Float training

`train_and_export.py` trains the model in PyTorch. Inputs are normalized:

```
x_float = (pixel / 255.0 − 0.1307) / 0.3081
```

MNIST mean = 0.1307, std = 0.3081 (standard values for the dataset).

### 2.2 Post-training quantization (PTQ)

Symmetric INT8 quantization is applied per-layer after training. No quantization-aware training (QAT) is used.

**Weight quantization:**
```
scale  = max(|W|) / 127
W_int8 = round(W / scale)  clipped to [−128, 127]
```

**Bias quantization** (into the INT32 accumulator domain):
```
bias_scale = w_scale × x_scale
bias_int32 = round(bias_float / bias_scale)
```

where `x_scale` is the input scale for that layer (see §2.3).

### 2.3 Input quantization

Raw uint8 pixels from the host are normalized and quantized on the PS before being written to the Input BRAM:

```
x_float = (pixel / 255.0f − 0.1307f) / 0.3081f
x_int8  = clamp(round(x_float / INPUT_SCALE), −128, 127)
```

`INPUT_SCALE = max(|(1.0 − 0.1307) / 0.3081|, |(0.0 − 0.1307) / 0.3081|) / 127`

This matches the normalization applied during training exactly.

### 2.4 Inter-layer requantization

After each FC layer, the INT32 accumulator must be brought back to INT8 range before the next layer can use it. Without this step, every positive accumulator value (~10 000–50 000 for MNIST) clips to 127, making the hidden layer effectively binary and collapsing test accuracy to ~27%.

The right-shift `k` is computed during export:

```
M_req = (w_scale × x_scale) / act_scale
k     = round(log2(1 / M_req))
```

where `act_scale` is the calibrated maximum post-ReLU activation range divided by 127, measured over 50 calibration batches. After the shift:

```
act_int8 = clip(ReLU(acc_int32 >> k), 0, 127)
```

The two shift constants (`MLP_REQ_SHIFT_FC1`, `MLP_REQ_SHIFT_FC2`) are written to `hardware/vivado/src/rtl/mlp_params.vh` automatically, so the RTL picks them up without manual editing.

### 2.5 Exported artifacts

| File | Description |
|---|---|
| `training/outputs/coe/fc{1,2,3}_weights.coe` | INT8 weights, 1 byte per address, hex |
| `training/outputs/coe/fc{1,2,3}_bias.coe` | INT32 biases, 1 word per address, hex |
| `training/outputs/coe/scales.txt` | All scale factors and shift values |
| `hardware/vivado/src/rtl/mlp_params.vh` | Verilog `define` for FC1/FC2 shifts |
| `hardware/vitis/mlp_accelerator_app/include/weights_biases.h` | C header with packed uint32 arrays for BRAM loading |

---

## 3. Hardware design

### 3.1 Block design

The Vivado block design (`mnist_mlp_accelerator`) connects:

| IP | Role |
|---|---|
| `processing_system7_0` | Zynq PS — AXI master, FCLK_CLK0 @ 100 MHz |
| `proc_sys_reset_0` | Synchronous reset for PL |
| `smartconnect_0` | AXI interconnect (1 master → 3 slaves) |
| `axi_bram_ctrl_0` | PS read/write access to Param BRAM |
| `blk_mem_gen_0` | Param BRAM — 64 KB true dual-port |
| `axi_bram_ctrl_1` | PS read/write access to Input BRAM |
| `blk_mem_gen_1` | Input BRAM — 1 KB true dual-port |
| `mlp_top_0` | Custom MLP accelerator (RTL module reference) |
| `ilslice_{0..3}` | Address bit slicers (strip byte-lane bits before BRAM Port B) |

Both BRAMs are true dual-port: Port A is connected to the AXI BRAM Controller (PS access); Port B is connected to the MLP engine (PL read-only).

### 3.2 AXI address map

| Peripheral | Base address | Size |
|---|---|---|
| Param BRAM | `0x40000000` | 64 KB |
| Input BRAM | `0x40010000` | 1 KB |
| MLP AXI-Lite | `0x40020000` | 4 KB |

### 3.3 RTL modules

**`mlp_top.v`** — top-level wrapper instantiated in the block design as a module reference. Connects AXI4-Lite slave ports to `axilite_ctrl` and BRAM Port B signals to `mlp_engine`. Implements a busy SR latch: set on start pulse, cleared on done pulse.

**`axilite_ctrl.v`** — minimal AXI4-Lite slave with three 32-bit registers:

| Offset | Direction | Description |
|---|---|---|
| `0x00` | Write | Control — write `0x1` to issue a start pulse |
| `0x04` | Read | Status — bit 0: done (read-to-clear), bit 1: busy |
| `0x08` | Read | Result — bits [3:0] predicted class (0–9) |

**`mlp_engine.v`** — serial MAC inference engine. FSM states:

```
IDLE → FC_INIT → BIAS_ADDR → BIAS_LATCH → BIAS_LOAD
     → W_ADDR  → W_LATCH   → W_READ    → W_ACC
     → SAVE    → NEXT_LAYER → ARGMAX_INIT → ARGMAX_CMP → DONE
```

Each BRAM fetch takes 3 cycles (ADDR → LATCH → READ) due to the synchronous BRAM pipeline. One 32-bit word fetched from the Param BRAM feeds four parallel signed INT8×INT8 multiplies, results accumulated into a 32-bit register.

**`mlp_params.vh`** — auto-generated by `train_and_export.py`. Defines `MLP_REQ_SHIFT_FC1` and `MLP_REQ_SHIFT_FC2` as Verilog `define` constants. Do not edit manually.

### 3.4 BRAM memory layout (Param BRAM)

Weights and biases are stored contiguously from `MLP_PARAM_BRAM_BASE_ADDR`:

```
0x00000000  FC1 weights   784×64 INT8   = 50 176 B  (12 544 words)
0x0000C400  FC1 biases         64 INT32 =    256 B  (    64 words)
0x0000C500  FC2 weights    64×32 INT8   =  2 048 B  (   512 words)
0x0000CD00  FC2 biases         32 INT32 =    128 B  (    32 words)
0x0000CD80  FC3 weights    32×10 INT8   =    320 B  (    80 words)
0x0000CEC0  FC3 biases         10 INT32 =     40 B  (    10 words)
────────────────────────────────────────────────────────────────────
            Total                        52 968 B  (~51.7 KB)
```

Weights are packed 4 INT8 bytes per 32-bit word, little-endian, row-major (`[out_features, in_features]`). The hardware engine reads one word per MAC step and processes 4 multiplies in parallel using DSP48 primitives.

### 3.5 Timing

At 100 MHz PL clock:

| Layer | Output neurons | Words per neuron | Cycles |
|---|---|---|---|
| FC1 | 64 | 196 (784÷4) | ~38 016 |
| FC2 | 32 | 16 (64÷4) | ~1 728 |
| FC3 | 10 | 8 (32÷4) | ~300 |
| **Total** | | | **~40 044 (~0.40 ms)** |

---

## 4. Firmware

### 4.1 Boot sequence (`main.c`)

1. Enable ARM Global Timer (required for `XTime_GetTime`)
2. Zero both BRAMs (work around AXI BRAM Controller first-write drop bug)
3. Call `mlp_bram_init()` — writes all weights and biases from `weights_biases.h`
4. Call `mlp_bram_readback_check()` — spot-checks first word of each parameter block
5. Enter UART inference loop

### 4.2 UART protocol

**Frame format (PC → board):**
```
[ 'I' 'M' 'G' '1' ] [ 784 bytes ]
  4-byte magic        raw uint8 pixels, row-major
```

The firmware scans for the magic header using a simple state machine, then reads exactly 784 bytes.

**Response (board → PC):**
```
READY\r\n                      (sent at start of each loop iteration)
RESULT class=N time_us=T\r\n   (after inference completes)
```

On error:
```
RESULT error=busy_timeout time_us=0\r\n
RESULT error=done_timeout time_us=0\r\n
```

### 4.3 Input normalization (PS side)

`mlp_write_input()` in `mlp_bram_init.c` normalizes each pixel before writing:

```c
x_float = (pixel / 255.0f - MLP_INPUT_MEAN) / MLP_INPUT_STD;
x_int8  = clamp(round(x_float / MLP_INPUT_SCALE), -128, 127);
```

Four INT8 pixels are packed into one uint32 word (little-endian) and written to the Input BRAM via `Xil_Out32`.

### 4.4 Inference handshake

```
PS writes 0x1 to CTRL (0x40020000)    → start pulse
PS reads STATUS (0x40020004)           → polls busy=1 then done=1
done is read-to-clear in axilite_ctrl
PS reads RESULT (0x40020008)           → bits [3:0] = predicted class
```

Timing is measured with the ARM Cortex-A9 Global Timer (counts at CPU_CLK/2 = 333 MHz).

---

## 5. Host scripts

### `send_image.py`

Sends one image to the board and prints the result. Normalizes the image to match the firmware's expected format. Supports `--invert` for white-digit-on-black images.

### `int_infer_check.py`

Pure-Python fixed-point simulation of `mlp_engine.v`. Mirrors the RTL arithmetic exactly: INT8 MACs, INT32 accumulators, right-shift requantization, ReLU clip, argmax. Use this to verify a new set of weights produces the expected output before flashing to hardware.

---

## 6. Retrain workflow

When weights are updated:

1. `python training/train_and_export.py` — updates `weights_biases.h` and `mlp_params.vh`
2. Re-synthesise in Vivado — `mlp_params.vh` change updates the shift constants in PL logic
3. Generate bitstream + export XSA
4. Rebuild Vitis app — `weights_biases.h` change updates BRAM load data
5. Reprogram board
