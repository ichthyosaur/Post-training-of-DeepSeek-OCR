# **Post-training of DeepSeek-OCR**

## **Introduction**

DeepSeek-OCR is an open-source OCR model by DeepSeek-AI, pioneering "contextual optical compression" to balance efficiency and accuracy. Its architecture combines a Deep Encoder (SAM-Base + CLIP-Large) and DeepSeek-3B-MoE decoder, compressing 1024×1024 document images into 256 visual tokens (94% memory reduction) while retaining 97% accuracy at 10x compression. Supporting nearly 100 languages, it parses tables, formulas, and geometric figures with native Markdown output, processing 200k+ pages/day on a single A100-40G.

Notably, the model currently relies solely on two-stage supervised fine-tuning (no Instruct-Tuning or Reinforcement Learning). This limits its ability to follow natural language instructions and align with human preferences—the core goal of our assignment is to enhance DeepSeek-OCR by implementing these two critical training stages.

[github link](https://github.com/deepseek-ai/deepseek-ocr)

## **Experiment**

-  Data
-  Memory Efficiency
-  Instruct Tuning & Reinforce Learning

## **Expected Results**

-  Model can understand natural language instructions and complete diverse OCR tasks
-  Achieve metric improvements across multiple datasets

## **Reference & Related Works**

[1] [DeepSeek-OCR: Contexts Optical Compression]()