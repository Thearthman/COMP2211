# Lab 8 model experiments

This directory contains the reproducible experiment used to replace the
original shallow CNN with a compact U-Net built from standard Keras layers.

## Result

The selected run is `runs/unet_basic_40`.

- Global validation foreground IoU: **0.811564**
- Mean per-image IoU: **0.798832**
- Worst breed IoU: **0.708202** (`miniature_pinscher`)
- Best breed IoU: **0.896267** (`newfoundland`)
- Breeds above 0.61: **37 / 37**

The target is interpreted as IoU **0.61 (61%)**, because IoU is in `[0, 1]`.
Per-breed IoU is accumulated over every validation image of that breed with the
same strict `prediction > 0.5` rule used by the grader.

## Training

The run uses the official seed-42 80/20 split and mask-aligned augmentation.
The model is trained from scratch using only convolution, pooling, transposed
convolution, concatenation, batch-normalization, and dropout layers. No
pretrained backbone or external model code is used.

```bash
cd /home/johnmich/work/Coding/COMP2211/lab8
nix develop -c python lab8_experiments/experiment.py \
  --model unet --epochs 40 --freeze-epochs 0 \
  --batch-size 32 --output lab8_experiments/runs/unet_basic_40
```

The trained checkpoint was copied to `lab8_work/model.weights.h5`. The submitted
`BackgroundSeparater` constructs the same U-Net with no arguments and loads all
parameters from that checkpoint, so grading does not require a network
download.

## Verification

```bash
cd /home/johnmich/work/Coding/COMP2211/lab8
nix develop -c python lab8_experiments/verify.py
nix develop -c python lab8_experiments/compliance_check.py
```

This checks batch sizes 1 and 32, finite probabilities, save/reload parity, and
the per-breed validation scores. It writes the complete result to
`lab8_experiments/final_report.json`.

The compliance check additionally verifies the Lab 8 HTML submission rules:
import safety, the deterministic 80/20 split, cumulative and graph-compatible
IoU state, the no-argument model constructor and checkpoint, the checkpoint size,
and synchronization between `lab8_tasks.py` and `lab8_tasks.ipynb`. The submitted
`lab8_task.py` must remain byte-for-byte identical to `lab8_tasks.py`.

If the source or notebook code cells are edited, synchronize the notebook and
regenerate the submission export before running the checks:

```bash
nix develop -c python lab8_experiments/sync_notebook.py
cp lab8_tasks.py lab8_task.py
nix develop -c python lab8_experiments/compliance_check.py
```
