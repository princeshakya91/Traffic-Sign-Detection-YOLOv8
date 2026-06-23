"""
YOLO-TS-DRF Training Script
============================
Implements three novelties:
  1. DRF  - Dynamic Receptive Field Module (architecture in YAML)
  2. SARO - Small-Object Aware Resolution Optimization (imgsz=1024)
  3. ATATS - Adaptive Two/Three-Stage Augmentation Training Strategy

Training from scratch (no pretrained weights).
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1 exclusively

from ultralytics import YOLO
from ultralytics.utils import LOGGER


# ---------------------------------------------------------------------------
# ATATS: Adaptive Two/Three-Stage Augmentation Training Strategy
# ---------------------------------------------------------------------------
# Stage 1 (Epoch 1-160):   mosaic=1.0, mixup=0.2,  copy_paste=0.15
# Stage 2 (Epoch 161-190): mosaic=1.0, mixup=0.05, copy_paste=0.05
# Stage 3 (Epoch 191-200): mosaic=0,   mixup=0,    copy_paste=0  (close_mosaic=10)
# ---------------------------------------------------------------------------

def atats_callback(trainer):
    """
    ATATS callback: adjusts augmentation parameters at stage boundaries.
    Stage 3 is automatically handled by Ultralytics' close_mosaic mechanism.
    """
    epoch = trainer.epoch  # 0-indexed

    # Transition to Stage 2 at epoch 160 (= epoch 161 in 1-indexed)
    if epoch == 160:
        LOGGER.info(
            "\n" + "=" * 60 +
            "\n[ATATS] Stage 2 activated (epoch 161-190)"
            "\n  mixup:      0.2  -> 0.05"
            "\n  copy_paste: 0.15 -> 0.05"
            "\n  mosaic:     1.0  (unchanged)"
            "\n" + "=" * 60
        )
        trainer.args.mixup = 0.05
        trainer.args.copy_paste = 0.05

        # Rebuild data augmentation transforms with new parameters
        if hasattr(trainer.train_loader.dataset, 'build_transforms'):
            trainer.train_loader.dataset.transforms = \
                trainer.train_loader.dataset.build_transforms(trainer.args)
        if hasattr(trainer.train_loader, 'reset'):
            trainer.train_loader.reset()

    # Stage 3 (epoch >= 190, 0-indexed) is handled by close_mosaic=10
    # which sets mosaic=0, mixup=0, copy_paste=0 automatically


if __name__ == '__main__':

    # -----------------------------------------------------------------------
    # 1. Load YOLO-TS-DRF architecture from scratch (NO pretrained weights)
    # -----------------------------------------------------------------------
    model = YOLO("./YOLO-TS-DRF_TT100K.yaml")

    # -----------------------------------------------------------------------
    # 2. Register ATATS callback
    # -----------------------------------------------------------------------
    model.add_callback("on_train_epoch_start", atats_callback)

    # -----------------------------------------------------------------------
    # 3. Train with SARO (imgsz=1024) + full hyperparameter configuration
    # -----------------------------------------------------------------------
    model.train(
        data="./TT100K-2016.yaml",
        epochs=200,
        batch=16,                   # SARO: adjusted for 1024 res on A100 40GB
        imgsz=1024,                 # SARO: high-res input for tiny signs
        device='0',                 # GPU 1 mapped to device 0 via CUDA_VISIBLE_DEVICES
        save_period=10,

        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=5,

        # Regularization
        label_smoothing=0.1,

        # ATATS Stage 1 augmentation (initial values)
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.15,
        close_mosaic=10,            # Stage 3: disable mosaic for last 10 epochs

        # Output
        project='results',
        name='YOLO-TS-DRF_TT100K',
    )

    # -----------------------------------------------------------------------
    # 4. Evaluate on validation set
    # -----------------------------------------------------------------------
    metrics = model.val(
        data="./TT100K-2016.yaml",
        imgsz=1024,
        batch=1,
        device='0'
    )

    # -----------------------------------------------------------------------
    # 5. Export optimized model
    # -----------------------------------------------------------------------
    path = model.export(format="engine", device='0', half=True, opset=12)
