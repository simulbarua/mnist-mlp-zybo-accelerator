# What We Changed — 4-Neuron Parallel MAC Engine

## Why we made these changes

The original design computed **one output neuron at a time**, using 4 DSP48E1 multipliers to process the 4 INT8 bytes packed in a single BRAM word. Inference took ~0.40 ms (40,044 cycles at 100 MHz).

The new design computes **four output neurons simultaneously** by splitting the weight BRAM into 4 independent banks. All 4 banks are read in the same clock cycle, giving 4 banks × 4 multiplies/bank = **16 DSP48E1 multipliers active per cycle**. Inference is now ~0.10 ms (10,026 cycles) — a 4× speedup.

---

## The core idea: BRAM banking

A true dual-port BRAM only has one Port B (the PL-read port). To read 4 neurons' weights simultaneously, we need 4 separate BRAMs — one per neuron group.

**Bank assignment rule:** bank `k` holds all output neurons whose index `≡ k (mod 4)`:

```
Bank 0: neurons 0, 4,  8, 12, 16, …  (rows 0, 4,  8, … of the weight matrix)
Bank 1: neurons 1, 5,  9, 13, 17, …
Bank 2: neurons 2, 6, 10, 14, 18, …
Bank 3: neurons 3, 7, 11, 15, 19, …
```

All 4 banks share the same Port B address and enable signal from the engine. Each returns its own 32-bit word. In one clock cycle the engine reads 4 words (16 INT8 weights) and feeds 16 DSP multipliers.

**FC3 padding:** FC3 has 10 output neurons (0–9). 10 is not divisible by 4, so we pad to 12 by adding 2 neurons (indices 10 and 11) with all-zero weights and biases. The argmax only checks indices 0–9, so the padding neurons never affect the output.

---

## Files changed

### 1. `hardware/vivado/src/rtl/mlp_engine.v` — complete rewrite

**What changed:**
- **Added 4 parallel accumulators** (`acc0`–`acc3`), one per output neuron being computed.
- **Added 4 BRAM read data ports** (`param0_rdata`–`param3_rdata`). Previously there was a single `param_rdata`.
- **16 DSP48E1 multipliers** declared with `(* use_dsp = "yes" *)`. Previously 4.
- **`param_addr` width**: `[15:0]` → `[13:0]` (each bank is 16 KB, not 64 KB).
- **`n_out` now counts neuron groups, not individual neurons**: FC1=16 groups (64÷4), FC2=8 groups (32÷4), FC3=3 groups (12÷4 after padding).
- **`logits` array**: `[0:9]` → `[0:11]` to hold the 2 padding neurons.
- **S_BIAS_LOAD**: loads all 4 accumulators simultaneously from the 4 bank outputs.
- **S_W_ACC**: all 4 accumulators accumulate in parallel each cycle.
- **S_SAVE**: requantizes and stores 4 activations at once.
- **Per-bank byte offsets** updated to the new 16 KB bank layout (FC1_W at 0x0000, FC1_B at 0x3100, FC2_W at 0x3140, etc.).

**Why:** This is the core change. Every output neuron requires its own accumulator and its own stream of weights. By assigning each accumulator to one BRAM bank, we can run all 4 in lock-step on the same loop counter.

---

### 2. `hardware/vivado/src/rtl/mlp_top.v` — complete rewrite

**What changed:**
- **Param BRAM interface**: removed single `param_bram_rdata [31:0]`; added four separate ports `param_bram0_rdata`, `param_bram1_rdata`, `param_bram2_rdata`, `param_bram3_rdata`.
- **`param_bram_addr`**: `[15:0]` → `[13:0]` to match the smaller 16 KB banks.
- All 4 banks still share the single `param_bram_addr` and `param_bram_en` output (broadcast in the block design).

**Why:** The top-level module is the boundary between the block design and the RTL. It exposes ports that the Vivado block diagram wires up — we need one port per bank's read-data bus.

---

### 3. `training/train_and_export.py` — `export_header` function replaced

**What changed:**
- **`_split_into_banks(W, b)`**: new helper that slices a weight matrix into 4 bank-indexed sub-matrices (`bank k = rows {k, k+4, k+8, …}`).
- **FC3 padding**: adds 2 zero-weight, zero-bias rows to FC3 before splitting, bringing it from 10 to 12 neurons.
- **`weights_biases.h` output**: now exports 24 arrays instead of 6 (4 banks × 3 layers × 2 types = 24): `fc1_bank0_weights`, `fc1_bank0_bias`, …, `fc3_bank3_bias`.
- **Offset defines**: added `FC1_WEIGHT_BANK_OFFSET`, `FC1_BIAS_BANK_OFFSET`, etc. (same values for all 4 banks).
- **Word-count defines**: `FC1_BANK_WEIGHT_WORDS`, `FC1_BANK_BIAS_WORDS`, etc. (per-bank sizes, ¼ of the full layer).

**Why:** The C firmware loads weights from this header into the 4 physical BRAMs. The header must match the bank layout that the RTL engine expects.

---

### 4. `hardware/vitis/mlp_accelerator_app/include/mlp_bram_init.h` — complete rewrite

**What changed:**
- **Removed** `MLP_PARAM_BRAM_BASE_ADDR 0x40000000` (single address for the old 64 KB BRAM).
- **Added** four bank base addresses:
  ```c
  #define MLP_PARAM_BRAM_BANK0_BASE  0x40000000UL  // 16 KB
  #define MLP_PARAM_BRAM_BANK1_BASE  0x40004000UL  // 16 KB
  #define MLP_PARAM_BRAM_BANK2_BASE  0x40008000UL  // 16 KB
  #define MLP_PARAM_BRAM_BANK3_BASE  0x4000C000UL  // 16 KB
  ```
- `MLP_INPUT_BRAM_BASE_ADDR 0x40010000` unchanged.

**Why:** The PS writes each bank through its own AXI BRAM Controller at a distinct base address. Four separate addresses are needed.

---

### 5. `hardware/vitis/mlp_accelerator_app/src/mlp_bram_init.c` — complete rewrite

**What changed:**
- **`mlp_bram_init()`**: now calls `write_bank()` four times, once per bank, passing that bank's weight/bias arrays and base address.
- **`write_bank(base, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b)`**: new helper that writes all 6 parameter blocks to one bank at their respective byte offsets.
- **Priming writes**: extended from 1 to 4 dummy writes (one per bank) to work around the AXI BRAM Controller v4.1 first-transaction-after-reset drop bug.
- **`mlp_bram_readback_check()`**: extended to verify all 4 banks × 3 layers × 2 types = 24 checks, using `CHK_W`/`CHK_B` macros.
- Input BRAM functions (`mlp_write_input`, `mlp_verify_input_bram`) **unchanged**.

**Why:** At boot the ARM must fill all 4 physical BRAMs with the correct interleaved weight data. Writing to the wrong base address, or in the wrong order, would produce wrong inference results.

---

### 6. `hardware/vivado/create_vivado_project.tcl` — block design section replaced

**What changed:**
- **SmartConnect `NUM_MI`**: `3` → `6` (was: param BRAM + input BRAM + ctrl; now: 4 param BRAMs + input BRAM + ctrl).
- **Added 3 new AXI BRAM Controllers** (`axi_bram_ctrl_2`, `_3`, `_4`) for banks 1–3.
- **Added 3 new Block Memory Generators** (`blk_mem_gen_2`, `_3`, `_4`) for banks 1–3. Each is 16 KB (depth=4096), matching bank 0 (reduced from the old 64 KB depth=16384).
- **Bank 0 `blk_mem_gen_0` depth** reduced from 16384 to 4096 to match the new 16 KB per-bank size.
- **Added 6 new ilslices** (`ilslice_4`–`ilslice_9`) to route Port A and Port B addresses for the 3 new BRAMs. `DIN_FROM=13, DIN_TO=2, DIN_WIDTH=14` (14-bit byte address → 12-bit word address for 16 KB BRAM), down from `DIN_FROM=15` for the old 64 KB BRAM.
- **`param_bram_addr` broadcast**: the engine's single 14-bit address output is connected to all 4 banks' Port B address inputs (via their respective ilslices).
- **`param_bram_en` broadcast**: the engine's enable signal drives all 4 banks' Port B enable pins.
- **Each bank's `doutb`** connected to its own dedicated `mlp_top_0` port (`param_bram0_rdata`–`param_bram3_rdata`).
- **Address assignments** updated: 4 × 16 KB at `0x40000000`–`0x4000FFFF`, input BRAM at `0x40010000`, ctrl at `0x40020000`.

**Why:** The block design is Vivado's "wiring diagram." Every new BRAM bank needs its own controller, memory primitive, address slicer (ilslice), and SmartConnect port. The address map must match what the firmware uses.

---

### 7. `README.md` and `docs/design.md` — documentation updated

**What changed:**
- Throughput: `~0.40 ms` → `~0.10 ms (4-neuron parallel MAC engine, 16 DSPs)`
- AXI address map table: single 64 KB Param BRAM row replaced with 4 × 16 KB bank rows
- `docs/design.md §3.1`: block design IP table updated with 4 controllers and 4 BRAMs
- `docs/design.md §3.3`: `mlp_engine.v` description rewritten for parallel design
- `docs/design.md §3.4`: BRAM layout table replaced with per-bank interleaved layout
- `docs/design.md §3.5`: timing table updated to groups-of-4 model (~10,026 cycles / ~0.10 ms)

---

## Summary table

| File | Type of change | Key metric |
|---|---|---|
| `mlp_engine.v` | Complete rewrite | 4 → 16 DSPs, 4× throughput |
| `mlp_top.v` | Complete rewrite | 1 → 4 param BRAM rdata ports |
| `train_and_export.py` | `export_header` replaced | 6 → 24 bank-interleaved arrays |
| `mlp_bram_init.h` | Complete rewrite | 1 → 4 bank base addresses |
| `mlp_bram_init.c` | Complete rewrite | Writes all 4 banks at boot |
| `create_vivado_project.tcl` | Block design replaced | 1 → 4 param BRAMs, NUM_MI 3→6 |
| `README.md` / `docs/design.md` | Updated | New address map, timing, DSP count |
