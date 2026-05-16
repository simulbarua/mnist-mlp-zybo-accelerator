# Architecture — MNIST MLP Zybo Accelerator

---

## 1. System-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          HOST PC                                │
│                                                                 │
│   send_image.py                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  1. Load 28×28 PNG                                       │  │
│   │  2. Send magic header "IMG1" + 784 raw uint8 bytes       │  │
│   │  3. Wait for "RESULT class=N time_us=T"                  │  │
│   └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │  USB / UART  115200 baud
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                     ZYBO Z7-10 BOARD                            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               PS  (ARM Cortex-A9 @ 667 MHz)             │    │
│  │                                                         │    │
│  │  main.c                                                 │    │
│  │  ┌───────────────────────────────────────────────────┐  │    │
│  │  │ Boot:                                             │  │    │
│  │  │   mlp_bram_init()  ─── load weights into 4 BRAMs │  │    │
│  │  │   mlp_bram_readback_check()  ── verify contents  │  │    │
│  │  │                                                   │  │    │
│  │  │ Per inference:                                    │  │    │
│  │  │   mlp_write_input()  ─── normalize + write image │  │    │
│  │  │   write 0x1 → AXI-Lite CTRL  ─── start engine   │  │    │
│  │  │   poll AXI-Lite STATUS until done                │  │    │
│  │  │   read AXI-Lite RESULT  ─── predicted digit      │  │    │
│  │  │   send "RESULT class=N time_us=T" over UART      │  │    │
│  │  └───────────────────────────────────────────────────┘  │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │  AXI4 (GP0 master port)           │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐    │
│  │               PL  (FPGA Fabric @ 100 MHz)               │    │
│  │                   [see diagram 2]                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Explanation

The system has two physically separate processors on the same chip — the **PS** (Processing System, an ARM CPU) and the **PL** (Programmable Logic, the FPGA fabric). They talk to each other over an internal bus called AXI.

**What the PC does:** A Python script loads a handwritten digit image, sends it over USB as raw pixel bytes, and waits for the board to reply with a predicted digit and how long it took.

**What the ARM (PS) does:** It handles all the "housekeeping" — receiving the image over UART, normalizing the pixel values from 0–255 into a signed 8-bit range the neural network expects, writing the image and weights into on-chip memory (BRAM), telling the FPGA engine to start, and sending the result back to the PC. The ARM does not run the neural network math — that's the FPGA's job.

**What the FPGA (PL) does:** It runs the actual neural network computation at 100 MHz. It reads weights and input pixels from BRAM, performs multiply-accumulate operations in hardware, and asserts a "done" signal with the predicted class when it finishes. It does this much faster than software on the ARM could.

**Why split it this way?** The ARM is easy to program and handles variable-length tasks (UART parsing, printf, timing). The FPGA is hard to program but can do fixed math extremely fast. Each part does what it's best at.

---

## 2. FPGA Block Design (PL Fabric)

```
                PS AXI Master (GP0)
                        │
                        ▼
          ┌─────────────────────────┐
          │     SmartConnect        │
          │   (1 master → 6 slaves) │
          └──┬──┬──┬──┬──┬─────────┘
             │  │  │  │  │
    ┌────────┘  │  │  │  └──────────────────────┐
    │      ┌───┘  │  └────────┐                 │
    │      │      │           │                 │
    ▼      ▼      ▼           ▼                 ▼
┌──────┐┌──────┐┌──────┐┌──────────┐  ┌──────────────┐
│AXI   ││AXI   ││AXI   ││AXI       │  │  AXI-Lite    │
│BRAM  ││BRAM  ││BRAM  ││BRAM      │  │  Ctrl        │
│Ctrl 0││Ctrl 2││Ctrl 3││Ctrl 1    │  │  (mlp_top_0) │
│Bank 0││Bank 1││Bank 2││Input BRAM│  │  0x40020000  │
└──┬───┘└──┬───┘└──┬───┘└────┬─────┘  └──────┬───────┘
   │       │       │         │                │
   ▼       ▼       ▼         ▼                │
┌──────┐┌──────┐┌──────┐┌──────────┐         │
│BRAM  ││BRAM  ││BRAM  ││BRAM      │         │
│Bank 0││Bank 1││Bank 2││ Input    │         │
│16 KB ││16 KB ││16 KB ││  1 KB   │         │
│      ││      ││      ││         │         │
│┌────┐││┌────┐││┌────┐││┌───────┐│         │
││PortA│││PortA│││PortA│││ Port A ││         │
││(PS)│││(PS) │││(PS) │││  (PS)  ││         │
│└────┘││└────┘││└────┘││└───────┘│         │
│┌────┐││┌────┐││┌────┐││┌───────┐│         │
││PortB│││PortB│││PortB│││ Port B ││         │
││(PL)│││(PL) │││(PL) │││  (PL)  ││         │
│└──┬─┘││└──┬─┘││└──┬─┘││└───┬───┘│         │
└───┼──┘└───┼──┘└───┼──┘└────┼────┘         │
    │       │       │         │              │
    └───────┴───────┴─────────┘              │
                    │                        │
                    ▼                        ▼
          ┌─────────────────────────────────────────┐
          │              mlp_top_0                  │
          │        (RTL Module Reference)           │
          │                                         │
          │  param_bram_addr  ──────────────────►   │
          │  param_bram_en    ──────────────────►   │
          │  ◄── param_bram0_rdata  (bank 0)        │
          │  ◄── param_bram1_rdata  (bank 1)        │
          │  ◄── param_bram2_rdata  (bank 2)        │
          │  ◄── param_bram3_rdata  (bank 3)        │
          │  input_bram_addr  ──────────────────►   │
          │  input_bram_en    ──────────────────►   │
          │  ◄── input_bram_rdata                   │
          │                                         │
          │  [contains axilite_ctrl + mlp_engine]   │
          └─────────────────────────────────────────┘

  Also present but not shown: AXI BRAM Ctrl 4 (Bank 3, 0x4000C000),
  BRAM Bank 3, and ilslice address bit-slicers (0–9) on all Port B
  address lines.
```

### Explanation

This diagram shows everything inside the FPGA fabric and how the ARM connects to it.

**SmartConnect** is an Xilinx IP block that acts like a traffic router. The ARM has one AXI port going out; the SmartConnect splits it into 6 separate paths, one for each peripheral the ARM needs to talk to. Think of it as a USB hub.

**BRAM (Block RAM)** is fast on-chip memory built into the FPGA. Each BRAM chip has two completely independent ports — Port A and Port B — that can read/write simultaneously. We use this to our advantage: Port A is wired to the ARM (so the ARM can load weights into it at boot time), and Port B is wired to the MLP engine (so the engine can read weights during inference). Neither side blocks the other.

**Why 4 separate weight BRAMs?** A BRAM only has one Port B — one read channel for the engine. If all weights lived in one BRAM, the engine could only read one neuron's weights per clock cycle. By splitting the weights into 4 banks (one per every 4th neuron), the engine can read from all 4 banks simultaneously, computing 4 neurons per cycle instead of 1.

**The Input BRAM** (1 KB) holds the 784 pixel values for the current image. The ARM writes it before each inference; the engine reads from it during FC1.

**mlp_top_0** is our custom RTL module — it contains the AXI-Lite control logic and the MLP inference engine. It receives one address/enable signal (broadcast to all 4 banks), gets back 4 separate data words, and routes start/done/result signals to/from the ARM.

**ilslices** (not shown for clarity) are small Xilinx IP blocks that strip the bottom 2 bits off an AXI byte address before feeding it to a BRAM word address. AXI addresses are in bytes; BRAM addresses are in 32-bit words — the ilslice does the conversion.

---

## 3. MLP Engine Internals (mlp_engine.v)

```
        ┌────────────────────────────────────────────────────────────┐
        │                     mlp_engine.v                          │
        │                                                            │
        │  Input: start pulse                                        │
        │  Output: done pulse + result[3:0] (digit 0–9)             │
        │                                                            │
        │  ┌──────────────────────────────────────────────────────┐  │
        │  │                      FSM                             │  │
        │  │  IDLE → FC_INIT → BIAS_ADDR → BIAS_LATCH →          │  │
        │  │  BIAS_LOAD → W_ADDR → W_LATCH → W_READ → W_ACC →   │  │
        │  │  SAVE → NEXT_LAYER → ARGMAX_INIT → ARGMAX_CMP →     │  │
        │  │  DONE                                                │  │
        │  └──────────────────────────────────────────────────────┘  │
        │                                                            │
        │  ── Per MAC step (S_W_ACC) ──────────────────────────────  │
        │                                                            │
        │  param_addr ──────────────────────────────────────────►    │
        │  (shared to all 4 banks)                                   │
        │                                                            │
        │  Bank 0 rdata [31:0]          Bank 1 rdata [31:0]          │
        │  [b3|b2|b1|b0]               [b3|b2|b1|b0]                │
        │   │   │   │   │               │   │   │   │               │
        │  ×  ×  ×  ×                 ×  ×  ×  ×                   │
        │  input input input input   input input input input         │
        │   │   │   │   │               │   │   │   │               │
        │   └─┬─┘   └─┬─┘               └─┬─┘   └─┬─┘              │
        │  4 DSPs   4 DSPs             4 DSPs   4 DSPs              │
        │     │         │                 │         │                │
        │     └────┬────┘                 └────┬────┘                │
        │       sum → acc0                  sum → acc1               │
        │                                                            │
        │  Bank 2 rdata [31:0]          Bank 3 rdata [31:0]          │
        │  [b3|b2|b1|b0]               [b3|b2|b1|b0]                │
        │   │   │   │   │               │   │   │   │               │
        │  ×  ×  ×  ×                 ×  ×  ×  ×                   │
        │  input input input input   input input input input         │
        │   │   │   │   │               │   │   │   │               │
        │   └─┬─┘   └─┬─┘               └─┬─┘   └─┬─┘              │
        │  4 DSPs   4 DSPs             4 DSPs   4 DSPs              │
        │     │         │                 │         │                │
        │     └────┬────┘                 └────┬────┘                │
        │       sum → acc2                  sum → acc3               │
        │                                                            │
        │                    Total: 16 DSP48E1                       │
        │                                                            │
        │  ── At S_SAVE ──────────────────────────────────────────── │
        │                                                            │
        │   acc0 → >> FC1_REQ_SHIFT → ReLU → act[j*4+0]            │
        │   acc1 → >> FC1_REQ_SHIFT → ReLU → act[j*4+1]            │
        │   acc2 → >> FC1_REQ_SHIFT → ReLU → act[j*4+2]            │
        │   acc3 → >> FC1_REQ_SHIFT → ReLU → act[j*4+3]            │
        │                  (all 4 in one clock cycle)                │
        │                                                            │
        │  ── Activations flow ───────────────────────────────────── │
        │                                                            │
        │   Input BRAM (784 INT8) ──► FC1 (64 neurons) ─► act1[]   │
        │   act1[] (64 INT8)      ──► FC2 (32 neurons) ─► act2[]   │
        │   act2[] (32 INT8)      ──► FC3 (10 neurons) ─► logits[] │
        │   logits[0..9]          ──► argmax ──► result[3:0]        │
        │                                                            │
        └────────────────────────────────────────────────────────────┘
```

### Explanation

This is the heart of the accelerator. It's a hardware state machine (FSM) written in Verilog that steps through the neural network math one state at a time.

**The FSM states:** The engine starts in IDLE, waiting for a start pulse from the ARM. When it gets one, it runs through a fixed sequence of states for each layer:
- **FC_INIT** — set up which layer and how many neuron groups to process
- **BIAS_ADDR / BIAS_LATCH / BIAS_LOAD** — fetch the bias values from BRAM (takes 3 cycles because BRAM is synchronous — you ask for data, wait a cycle, wait another cycle, then read it)
- **W_ADDR / W_LATCH / W_READ / W_ACC** — fetch one column of weights and multiply-accumulate into the running sums; this loop repeats once per input word
- **SAVE** — apply requantization and ReLU, store the 4 output activations
- **NEXT_LAYER** — move on to FC2, then FC3
- **ARGMAX_INIT / ARGMAX_CMP** — scan the 10 logits to find the largest
- **DONE** — pulse the done signal with the result

**DSP48E1 slices** are dedicated multiply-accumulate hardware units built into the Xilinx FPGA. Each one can do one signed multiply per clock cycle. We use 16 of them in parallel — 4 per bank. Each 32-bit BRAM word contains 4 packed INT8 weight bytes, so 4 multipliers process one word. With 4 banks active simultaneously, that's 16 multiplications per clock cycle.

**Why 4 accumulators?** Each output neuron needs its own running sum. Since we're computing 4 neurons in parallel, we maintain 4 separate 32-bit accumulators (acc0–acc3). They all advance in lockstep — same input words, different weight words (from different banks).

**Requantization (the right-shift):** The multiply-accumulate produces a 32-bit integer result, but the next layer expects 8-bit inputs. Simply truncating would destroy the values. Instead, we right-shift by a pre-computed number of bits (calculated during training export) to bring the range back to [−128, 127], then apply ReLU (clamp negatives to 0). This is done in one clock cycle for all 4 accumulators simultaneously at SAVE.

**Activations flow:** The engine uses three on-chip register arrays — act1[64], act2[32], and logits[12] — to hold intermediate results between layers. No external memory is needed between layers.

---

## 4. BRAM Bank Layout (Weight Interleaving)

```
  Full weight matrix for FC1 (64 output neurons × 784 inputs):

  Neuron index:   0    1    2    3    4    5    6    7   ...  63
                  │    │    │    │    │    │    │    │        │
                  ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼        ▼
  Bank assigned:  0    1    2    3    0    1    2    3   ...   3
                  │         │         │         │
                  ▼         ▼         ▼         ▼
              Bank 0    Bank 1    Bank 2    Bank 3
             16 rows   16 rows   16 rows   16 rows
           ×784 bytes ×784 bytes ×784 bytes ×784 bytes
             =12544B   =12544B   =12544B   =12544B

  All 4 banks use the same byte-offset map:

  ┌──────────────┬──────────────────────────────────────────────────┐
  │   Offset     │  Contents                                        │
  ├──────────────┼──────────────────────────────────────────────────┤
  │  0x00000000  │  FC1 weights  (16 neurons × 784 B = 12,544 B)   │
  │  0x00003100  │  FC1 biases   (16 neurons ×   4 B =     64 B)   │
  │  0x00003140  │  FC2 weights  ( 8 neurons ×  64 B =    512 B)   │
  │  0x00003340  │  FC2 biases   ( 8 neurons ×   4 B =     32 B)   │
  │  0x00003360  │  FC3 weights  ( 3 neurons ×  32 B =     96 B)   │
  │  0x000033C0  │  FC3 biases   ( 3 neurons ×   4 B =     12 B)   │
  ├──────────────┼──────────────────────────────────────────────────┤
  │  Total       │  13,260 B per bank  (<16 KB)                    │
  └──────────────┴──────────────────────────────────────────────────┘

  FC3 note: only 10 real neurons. Padded to 12 (3 groups of 4) by
  adding neurons 10 and 11 with zero weights. Argmax checks [0..9] only.
```

### Explanation

**The interleaving rule** is simple: neuron 0 goes to bank 0, neuron 1 to bank 1, neuron 2 to bank 2, neuron 3 to bank 3, neuron 4 back to bank 0, and so on. This is called stride-4 interleaving. The result is that at any given address offset, the 4 banks collectively hold one row of weights for 4 different neurons — exactly what the 4 parallel accumulators need.

**Why do all 4 banks have the same offset map?** Because from each bank's point of view, it only stores every 4th neuron. So each bank holds the same *fraction* of each layer. The offsets are identical because the fractions are the same size. The engine sends the same address to all 4 banks and reads back 4 different words in the same cycle.

**Why does each weight take a whole byte?** The weights are INT8 — 8-bit signed integers. Four of them are packed into one 32-bit word (little-endian). The engine reads one word and unpacks all 4 bytes, multiplying each by the corresponding input byte.

**FC3 padding:** The network has 10 output classes (digits 0–9). 10 is not divisible by 4, so we pad FC3 to 12 neurons by appending 2 fake neurons with zero weights and zero biases. This means the engine always processes exactly 3 groups of 4, with no special-case logic. The 2 extra neurons produce logit values of 0, which can never win the argmax over any real neuron with a non-zero score.

**Per-bank sizes shrink across layers:** FC1 has 64 neurons (16 per bank × 784 inputs = 12,544 bytes); FC2 has 32 neurons (8 per bank × 64 inputs = 512 bytes); FC3 has 12 padded neurons (3 per bank × 32 inputs = 96 bytes). All of this fits comfortably within 16 KB per bank.

---

## 5. AXI Address Map

```
  0x40000000  ┌──────────────────────────────┐
              │   Param BRAM  Bank 0  (16KB) │  neurons 0,4,8,...
  0x40004000  ├──────────────────────────────┤
              │   Param BRAM  Bank 1  (16KB) │  neurons 1,5,9,...
  0x40008000  ├──────────────────────────────┤
              │   Param BRAM  Bank 2  (16KB) │  neurons 2,6,10,...
  0x4000C000  ├──────────────────────────────┤
              │   Param BRAM  Bank 3  (16KB) │  neurons 3,7,11,...
  0x40010000  ├──────────────────────────────┤
              │   Input BRAM          (1KB)  │  784 INT8 pixels
  0x40010400  ├──────────────────────────────┤
              │        (gap)                 │
  0x40020000  ├──────────────────────────────┤
              │   AXI-Lite Control    (4KB)  │
              │   0x00 → CTRL  (write 0x1)   │
              │   0x04 → STATUS (busy/done)  │
              │   0x08 → RESULT (class 0-9)  │
  0x40021000  └──────────────────────────────┘
```

### Explanation

The ARM CPU talks to all peripherals through a single unified address space — the same way a computer's RAM and I/O devices share one address bus. Each peripheral is assigned a range of addresses.

**The 4 param BRAM banks** each get 16 KB of address space, spaced 16 KB apart starting at 0x40000000. The ARM writes to each bank independently at boot to load the interleaved weights. During inference, only the FPGA engine reads them (via Port B), not the ARM.

**The Input BRAM** lives at 0x40010000. The ARM writes 784 bytes here before each inference. It only needs 784 bytes (196 words) but gets a 1 KB window, which is the minimum AXI BRAM Controller allocation.

**The AXI-Lite control block** at 0x40020000 has just 3 registers:
- **CTRL (offset 0x00):** Write `0x1` here to fire a one-cycle start pulse into the MLP engine. The engine latches it and begins inference.
- **STATUS (offset 0x04):** The ARM polls this. Bit 1 = busy (engine is running). Bit 0 = done (inference finished, read-to-clear). The ARM spins in a loop reading this until done=1.
- **RESULT (offset 0x08):** After done=1, the ARM reads bits [3:0] here to get the predicted digit (0–9).

---

## 6. Inference Timing Breakdown

```
  At 100 MHz PL clock.  Each BRAM fetch = 3 cycles (ADDR→LATCH→READ).
  Each group = 4 output neurons computed simultaneously.

  ┌────────────┬──────────────┬─────────────────┬──────────────────┐
  │  Layer     │   Groups     │  Words/group    │  ~Cycles         │
  ├────────────┼──────────────┼─────────────────┼──────────────────┤
  │  FC1       │  16 (64÷4)  │  196  (784÷4)   │  ~9,408          │
  │  FC2       │   8 (32÷4)  │   16   (64÷4)   │    ~432          │
  │  FC3       │   3 (12÷4)  │    8   (32÷4)   │     ~96          │
  │  Overhead  │      —      │      —          │     ~90          │
  ├────────────┼──────────────┼─────────────────┼──────────────────┤
  │  Total     │             │                 │  ~10,026         │
  │            │             │                 │  ≈ 0.10 ms       │
  └────────────┴──────────────┴─────────────────┴──────────────────┘

  Previous serial design:  ~40,044 cycles  ≈ 0.40 ms
  Speedup:                  4×
  DSP improvement:          4 → 16
```

### Explanation

**How cycles are counted:** For each neuron group, the engine loops over all input words — one BRAM fetch per word, each taking 3 cycles (issue address → pipeline stage 1 → pipeline stage 2 → data valid). Then it does one accumulate cycle and eventually one save cycle. The dominant cost is the weight fetch loop.

**FC1 dominates** because its weight matrix is by far the largest: 784 inputs × 64 neurons. Even with 4-way parallelism, that's 16 groups × 196 words/group = 3,136 BRAM fetches, each costing ~3 cycles = ~9,408 cycles.

**FC2 and FC3 are cheap** because the hidden layer sizes are small (64 and 32 inputs respectively). Their cycles are almost negligible compared to FC1.

**Overhead** includes state transitions (BIAS_LOAD, SAVE, NEXT_LAYER, ARGMAX), which add a small flat cost regardless of layer size.

**How to read the speedup:** The old design processed one neuron at a time, so FC1 alone took 64 × 196 × 3 ≈ 37,632 cycles. The new design processes 4 at a time, so FC1 takes 16 × 196 × 3 ≈ 9,408 cycles — exactly 4× fewer. The overall 4× improvement holds across all layers because the parallelism factor is the same everywhere.

**The 0.10 ms figure** is the hardware-only time. The total round-trip time reported by `send_image.py` will be larger because it includes UART transmission (~55 ms at 115200 baud for 784 bytes) and the ARM's normalization + BRAM write time (~1–2 ms).
