# Structured-Scene-Extraction-with-a-Fine-Tuned-VLM
A vision language model fine-tuned to turn an image into structured JSON, with follow-up question answering and distance measurement over the extracted objects.

## What it does

Drop in an image. The model returns every object in it as structured data —
colour, shape, size, and position — shown as a table. You can then ask questions
about the scene in plain language, or select two objects and measure the distance
between them.

The dataset was created by me. You will find both dataset generation code and training code here: `dataset_generation_and_Training.ipynb`

This is demo data:
<img width="512" height="512" alt="shapes_00005" src="https://github.com/user-attachments/assets/6124f32d-c22a-4c5d-9b8b-a4fe7bc3d40c" />
It has different shapes, colors, and sizes. These are randomly generated while ensuring that there is no overlap. 

## Approach

**Fine-tuned for format, not for knowledge.** Qwen2.5-VL-3B-Instruct, adapted with
LoRA on roughly 3.6M trainable parameters. The base model could already describe
scenes; what it could not do was emit a strict, deterministic output every time. The training
target was deterministic, parseable output.
