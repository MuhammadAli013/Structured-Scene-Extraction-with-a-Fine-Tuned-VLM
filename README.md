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
The deterministic output for this particular image is mentioned below:
```
0	
shape	"triangle"
color	"blue"
size	"small"
center	[ 334, 121 ]
area	878.1
dimension	{ name: "side", value: 45.0JS:45 }
1	
shape	"circle"
color	"green"
size	"big"
center	[ 97, 415 ]
area	6647.6
dimension	{ name: "radius", value: 46 }
2	
shape	"circle"
color	"cyan"
size	"small"
center	[ 342, 242 ]
area	2123.7
dimension	{ name: "radius", value: 26 }
3	
shape	"square"
color	"cyan"
size	"small"
center	[ 276, 388 ]
area	2704
dimension	{ name: "side", value: 52 }
4	
shape	"circle"
color	"blue"
size	"big"
center	[ 78, 70 ]
area	6647.6
dimension	{ name: "radius", value: 46 }
5	
shape	"square"
color	"yellow"
size	"big"
center	[ 213, 244 ]
area	8464
dimension	{ name: "side", value: 92 }
6	
shape	"triangle"
color	"red"
size	"big"
center	[ 401, 441 ]
area	2748.8
dimension	{ name: "side", value: 79.7 }
7	
shape	"square"
color	"blue"
size	"small"
center	[ 400, 209 ]
area	2704
dimension	{ name: "side", value: 52 }
8	
shape	"triangle"
color	"green"
size	"small"
center	[ 87, 213 ]
area	878.1
dimension	{ name: "side", value: 45.0JS:45 }
```
## Approach

**Fine-tuned for format, not for knowledge.** I used the Qwen2.5-VL-3B model for this purpose. The model was already very strong. It could answer all the questions related to the image even without any training. My target was not QA. I tried to train a model that gives strict and deterministic values and can answer questions. So, I had to train the model with deterministic output and demo QA. I used LoRA with only 3.6M trainable parameters. 

These are some demo prompts:
```
4	
image_filename	"shapes_00000.png"
question	"Is there a red triangle above the blue circle?"
target	'{"answer": "No", "objects": [{"shape": "square", "color": "blue", "size": "small", "center": [194, 227], "area": 2704, "dime…rcle", "color": "red", "size": "big", "center": [156, 136], "area": 6647.6, "dimension": {"name": "radius", "value": 46}}]}'
5	
image_filename	"shapes_00000.png"
question	"How many blue objects are there?"
target	'{"answer": "3", "objects": [{"shape": "square", "color": "blue", "size": "small", "center": [194, 227], "area": 2704, "dimen…rcle", "color": "red", "size": "big", "center": [156, 136], "area": 6647.6, "dimension": {"name": "radius", "value": 46}}]}'
6	
image_filename	"shapes_00000.png"
question	"What is the radius of the red circle?"
target	'{"answer": "46", "objects": [{"shape": "square", "color": "blue", "size": "small", "center": [194, 227], "area": 2704, "dime…rcle", "color": "red", "size": "big", "center": [156, 136], "area": 6647.6, "dimension": {"name": "radius", "value": 46}}]}'
```
You can see it has answers and some deterministic values. 

**Question answering generated under rules, not sampled.** Counting questions may
have an answer of zero, and referential questions are only produced when the
referent is unique in that scene. This keeps every training question answerable
from the image alone.

**Validation-based checkpointing.** A held-out split is evaluated periodically and
the adapter is saved only when validation loss improves, rather than on training
loss.

## Interface

FastAPI backend with a hand-built front end. Drag and drop an image, the object
table fills in colour-coded by object, a chat panel answers questions about the
scene, and a distance tool draws measured lines between selected objects on a
canvas overlay.

You can run the interface with this command:
```
cd Interface
uvicorn app:app --host 127.0.0.1 --port 8000
```
<img width="1889" height="947" alt="image" src="https://github.com/user-attachments/assets/54af6ff3-0705-4819-9e64-f5765caf1779" />
Also, you can measure the distance between them.

## Stack

Qwen2.5-VL-3B-Instruct, PEFT/LoRA, PyTorch, Transformers, FastAPI
