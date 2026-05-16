# Architecture Diagrams — MNIST MLP Zybo Accelerator

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
