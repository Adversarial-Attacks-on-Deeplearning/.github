    .github

# Adversarial Attacks on Deeplearning

## Cairo University – Faculty of Engineering

Department of Electronics & Electrical Communication

------

## Overview

This organization hosts the code and experiments for the graduation project **Adversarial Attacks on Deep Learning**, conducted at Cairo University (2024/2025). Our project presents the first large-scale, unified empirical study of adversarial attacks and defenses across key computer vision and language modeling tasks—including image classification, segmentation, object detection, NLP, LLMs, and automatic speech recognition.

**Key Project Goals:**

- Evaluate the vulnerability of deep learning models (CV, NLP, ASR, LLM) to adversarial attacks.
- Develop and test robust, novel defense mechanisms.
- Provide reproducible experiments via open-source Jupyter notebooks and PyTorch/TensorFlow implementations.

------

## Repository Structure

| Repository                                     | Content Overview                                             | Tasks/Domains          |
| ---------------------------------------------- | ------------------------------------------------------------ | ---------------------- |
| **YOLO**                                       | Adversarial attacks & defenses for YOLOv3/v8 object detection (traffic signs) | Object Detection       |
| **EfficientNet**                               | Attacks & defenses for EfficientNet models (GTSRB data)      | Image Classification   |
| **U_net**                                      | U-Net segmentation with CV attack & defense methods          | Image Segmentation     |
| **Adversarial-Attacks-on-Deeplearning-Models** | Core attack/defense algorithms and pipelines, cross-task utilities | Shared Core Code/Utils |
| **NLP**                                        | Text attack (TextFooler, TextBugger, DeepWordBug), hybrid LSTM & LLM defense | Sentiment, LLM, Prompt |
| **speech_to_text**                             | Speech attacks (FGSM, PGD, Cramér-IPM), ASR defenses (quant, smoothing, MP3, etc) | Speech Recognition     |

All repositories are implemented as **public Jupyter notebooks** (Python, PyTorch/TensorFlow).

------

## Main Features

- **Attacks Implemented:**
  - Vision: FGSM, PGD, MI-FGSM, JSMA, DeepFool, UAP, HopSkipJump, SimBA, One-Pixel, DAG, FoolDetectors, Square Attack, Physical attacks
  - NLP/LLM: TextFooler, TextBugger, DeepWordBug, Jailbreak attacks (GCG, BEAST), prompt injection & universal adversarial prompts
  - ASR: FGSM, PGD, Cramér-IPM, psychoacoustic (imperceptible) attacks, transferability experiments
- **Defenses Implemented:**
  - Proactive: Adversarial Training (vision, segmentation, NLP), Virtual Adversarial Training (VAT)
  - Reactive: JPEG/MP3 compression, Noise Fusion, Gaussian Blur, Spectral Gating, Input Quantization, Dynamic preprocessing pipelines
  - Certified: Randomized Smoothing
  - Hybrid/pipeline: Character-level purification (NLP), Dual-layer jailbreak filters (LLM: Detoxify + LoRA DistilBERT), LLM Self-Reflection
- **Empirical Evaluation:**
  - Full metrics (accuracy, mAP, Dice, CER/WER, ASR, F1, precision/recall, timing)
  - Transferability tests (esp. ASR/SSL)
  - Attack/defense trade-off analysis (latency, robustness, clean accuracy

------

## Contributors

- **Graduation Project Members:** Ahmed Tamer Samir Mohamed, Salma Mohamed Hamed Mostafa, Saad Ahmed Saad Ali, Omar Khaled Abdel Aleem Ali, Fatma Hussein Abdul Wahed Zaher, Youssef Hesham Abdel Fattah Mohamed
- **Supervisors:** Prof. Dr. Hanan Ahmed Kamal, Dr. Mohamed Abdo Tolba
- **Sponsor:** Si-Vision
