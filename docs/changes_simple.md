# What We Changed (Simple Version)

## The one-line summary

We made the FPGA do 4 things at once instead of 1. It's now 4× faster and uses 4× more of the chip's math units.

---

## Background: what the project does

We have a neural network that recognizes handwritten digits (0–9). It runs on an FPGA chip on the Zybo board. The FPGA does all the math; a laptop sends it an image over USB and gets back a digit.

The neural network has three layers of math:
- **FC1**: 784 inputs → 64 outputs
- **FC2**: 64 inputs → 32 outputs
- **FC3**: 32 inputs → 10 outputs (one per digit 0–9)

Each output is computed by multiplying a bunch of numbers together and adding them up. These multiply-and-add operations are done by dedicated hardware units on the chip called **DSPs**.

---

## What was slow before

The old design computed **one output at a time**. It would finish neuron 0, then neuron 1, then neuron 2… all the way through all 106 neurons across the three layers. Like doing homework problems one by one, in order.

- **DSPs used:** 4
- **Time per image:** ~0.40 ms

---

## What we changed

We made it compute **four outputs at the same time**. Like four people each doing their own homework problem simultaneously.

To do this, we split the weight storage (BRAM — think of it as the chip's internal memory) into **4 separate banks**. Each bank holds the weights for every 4th neuron:

```
Bank 0: neuron 0, neuron 4, neuron 8, ...
Bank 1: neuron 1, neuron 5, neuron 9, ...
Bank 2: neuron 2, neuron 6, neuron 10, ...
Bank 3: neuron 3, neuron 7, neuron 11, ...
```

Now in one clock cycle, the chip reads from all 4 banks at once and runs all 4 neurons' math in parallel.

- **DSPs used:** 16 (was 4)
- **Time per image:** ~0.10 ms (was 0.40 ms)
- **Speedup:** 4×

One small detail: FC3 only has 10 real outputs (digits 0–9), but 10 doesn't divide evenly by 4. So we padded it to 12 by adding 2 fake neurons with zero weights. They never affect the answer because we only look at outputs 0–9 at the end.

---

## Files we changed and why (plain English)

### `mlp_engine.v` — the FPGA math engine
**Before:** one accumulator, one stream of weights, 4 DSPs  
**After:** four accumulators running in parallel, four streams of weights, 16 DSPs  
This is the core of the change. Everything else supports this.

### `mlp_top.v` — the top-level wiring
**Before:** one wire bringing weight data into the engine  
**After:** four wires, one per memory bank  
Think of it as upgrading from a single-lane road to a 4-lane highway.

### `train_and_export.py` — the Python training script
**Before:** exported weights as one big block  
**After:** exports weights split into 4 interleaved groups that match the 4 banks  
The FPGA has to find each neuron's weights in the right bank, so the weights have to be arranged correctly when we load them.

### `mlp_bram_init.h` / `mlp_bram_init.c` — the firmware that loads weights onto the board
**Before:** wrote all weights to one memory address  
**After:** writes each bank's weights to its own memory address  
The ARM processor (inside the Zybo) loads the weights at startup. It now has 4 separate destinations to write to instead of 1.

### `create_vivado_project.tcl` — the script that builds the chip design
**Before:** 1 weight memory block, 1 memory controller, 3 connections  
**After:** 4 weight memory blocks, 4 memory controllers, 6 connections  
Vivado is the tool we use to design the FPGA layout. This script automates recreating the design. We updated it to add the 3 new memory banks and wire everything together.

### `README.md` and `docs/design.md` — documentation
Updated to reflect the new speed (0.10 ms), new DSP count (16), and new memory layout (4 banks).

### `docs/changes.md` — this changes log
New file. Written for the presentation.

---

## How to run it

Exactly the same as before — nothing changed on the user side:

1. Source `create_vivado_project.tcl` in Vivado
2. Build bitstream → export XSA
3. Build firmware in Vitis
4. Flash the board
5. Run `send_image.py` from the `host/` folder

The only difference you'll notice is the `time_us` number in the output will be ~4× smaller.

---

## Quick reference numbers

| | Before | After |
|---|---|---|
| DSPs used | 4 | 16 |
| Neurons computed per cycle | 1 | 4 |
| Inference time | ~0.40 ms | ~0.10 ms |
| Weight memory banks | 1 × 64 KB | 4 × 16 KB |
| AXI memory connections | 3 | 6 |
