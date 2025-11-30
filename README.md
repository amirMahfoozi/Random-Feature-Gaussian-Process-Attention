# Random Feature Gaussian Process Attention

This repository contains the code for our work  
**"Random Feature Gaussian Process Attention: Linear-Time Probabilistic Attention with Calibrated Uncertainty"**  

The goal of this project is to provide a plug-and-play attention module that:
- treats Transformer attention as a Gaussian process with a random Fourier feature kernel approximation,
- provides calibrated uncertainty estimates for attention outputs, and
- scales linearly in the sequence length.

The code here is a cleaned-up version of the implementation used in our experiments.
