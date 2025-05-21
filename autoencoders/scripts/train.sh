#!/usr/bin/env bash
python -m src.cli \
  model=vae \
  dataset=cifar10 \
  trainer.max_epochs=100 \
  run_name=vae_colab_demo
