# VGRefine: Inference-Time Visual Grounding Refinement for Medical MLLMs

This repository provides the code and evaluation scripts for our paper **"How Do Medical MLLMs Fail? A Study on Visual Grounding in Medical Images"** .

As mentioned in the paper, Our method is built on top of the [HuatuoGPT-Vision] codebase and improves medical VQA performance through simple inference-time attention refinement.

## Setup
This project builds on the HuatuoGPT-Vision framework. Please follow its installation instructions before running our code.

## Visualization
To reproduce the visual grounding analysis from the paper, run:

```bash
measure_attention_ratio.ipynb
'''

## Evaluation
To evaluate VGRefine on the VQA-RAD dataset, run:

```bash
eval_RAD.ipynb
'''
