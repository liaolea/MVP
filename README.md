# Modeling Variants of Prompts for Vision-Language Models

<h5 align="center">
[![arXiv](https://img.shields.io/badge/Arxiv-2501.13106-AD1C18.svg?logo=arXiv)](https://github.com/xiaoyaoxinyi/MVP) 
</h5>

## Introduction
we introduce the RobustPrompt Benchmark, a systematic benchmark to evaluate robustness to different prompt templates for VLMs. It includes a dataset with hundreds of carefully designed prompt templates, divided into six types, covering a wide variety of commonly used templates. 

<div align="center">
  <img src="benchmark.png"/>
</div>

Beside the benchmark, we propose Modeling Variants of Prompts (MVP), a simple yet effective method that mitigates sensitivity by modeling variants of prompt structures. The innovation of MVP lies in decoupling prompts into templates and class names, and using Variational Autoencoders (VAE) to model the distribution of diverse prompt structures. 

<div align="center">
  <img src="model.png"/>
</div>

