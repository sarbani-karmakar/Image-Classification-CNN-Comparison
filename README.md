# Image-Classification-CNN-Comparison
# Image Classification with Pretrained CNNs: A Case Study in Model Behaviour

This project explores image classification using pretrained convolutional neural networks (MobileNetV2 and ResNet50) via the [ImageAI](https://github.com/OlafenwaMoses/ImageAI) library and raw PyTorch/torchvision. What began as a straightforward classification exercise turned into an investigation of a persistent misclassification, revealing a well-documented but often overlooked characteristic of CNNs: **texture bias over shape bias**.

## Project Overview

Three test images (a giraffe, a Godzilla figure, and a house) were run through pretrained ImageNet classifiers to observe how well general-purpose models handle everyday and out-of-distribution images.

**Tech stack:** Python 3.11, PyTorch, torchvision, ImageAI 3.0.3, OpenCV

## Setup

- ImageAI's newer `imageai.Classification.ImageClassification` API uses a **PyTorch backend**, not the TensorFlow/Keras stack used in older ImageAI tutorials and documentation. This distinction matters if following older guides, which reference a deprecated `imageai.Prediction` module built on SqueezeNet/TensorFlow.
- Two pretrained architectures were compared:
  - **MobileNetV2** — a lightweight architecture (~14MB) designed for speed and mobile/edge deployment
  - **ResNet50** — a deeper, larger architecture (~98MB) generally associated with higher accuracy at the cost of inference speed

## Experiment 1: Baseline Classification (MobileNetV2, default preprocessing)

Using ImageAI's default center-crop preprocessing pipeline:

| Image | Top Prediction | Confidence | Assessment |
|---|---|---|---|
| house.jpg | boathouse | 96.6% | Reasonable — ImageNet's 1000 classes have no generic "house" category |
| godzilla.jpg | common iguana | 47% | Reasonable — no "kaiju" class exists; reptilian features correctly matched |
| giraffe.jpg | impala | 29% | **Misclassification** |

## Experiment 2: Ruling Out Preprocessing (MobileNetV2, squish-resize)

**Hypothesis:** The giraffe photo is portrait-oriented; ImageAI's default resize-then-center-crop preprocessing might be cropping out the giraffe's head and neck, leaving mostly legs and grass — features shared with other savanna animals.

**Method:** Bypassed ImageAI's built-in preprocessing and ran the same MobileNetV2 weights directly through torchvision, using a direct `Resize((224,224))` squish (no cropping), to test whether preserving the full image improved the result.

| Image | Top Prediction | Confidence |
|---|---|---|
| house.jpg | boathouse | 96.6% |
| godzilla.jpg | American alligator | 44.1% |
| giraffe.jpg | leopard | 14.1% |

Giraffe's top-5 also included impala, gazelle, llama, and German short-haired pointer, all within a narrow 10–14% band.

**Finding:** Removing the crop did not fix the misclassification — the giraffe still didn't appear as the top prediction, and confidence became more spread out rather than more accurate. This ruled out cropping as the primary cause.

## Experiment 3: Testing a Larger Architecture (ResNet50, correct preprocessing)

**Hypothesis:** A larger, generally more accurate architecture might resolve the misclassification.

**Method:** Ran the same three images through ResNet50 (ImageNet-pretrained), using torchvision's officially recommended preprocessing for that model (correct resize, crop, and normalization as specified by the model's own weights metadata).

| Image | Top Prediction | Confidence |
|---|---|---|
| house.jpg | boathouse | 58.4% |
| godzilla.jpg | American alligator | 42.6% |
| giraffe.jpg | gazelle | 22.1% |

Giraffe's top-5 also included impala (16.0%), hartebeest, zebra, and leopard.

**Finding:** ResNet50 also failed to correctly classify the giraffe, despite being a larger and generally more accurate architecture. Gazelle and impala appeared in the top-5 across **every** test — every architecture, every preprocessing method.

## Summary Comparison

| Test | Model | Giraffe Top Prediction | Confidence |
|---|---|---|---|
| Baseline | MobileNetV2 (crop) | impala | 29% |
| Preprocessing fix | MobileNetV2 (squish) | leopard | 14% |
| Larger model | ResNet50 | gazelle | 22% |

## Conclusion

The giraffe misclassification persisted across two different architectures and two different preprocessing strategies, which rules out cropping artifacts and challenges a simple "bigger model = correct answer" assumption. Model size and depth generally *do* correlate with higher aggregate accuracy across ImageNet as a whole — but this case shows that improvement is statistical, not a guarantee on any individual hard example.

The consistent appearance of gazelle, impala, and other savanna animals across every test points to a well-documented phenomenon in computer vision research: **CNNs trained on ImageNet tend to be texture-biased rather than shape-biased**. Where a human observer immediately identifies a giraffe by its unmistakable silhouette (the long neck, in particular), these models appear to weight coat pattern, colour, and background context more heavily than overall shape — and giraffe's mottled pattern sits close to other patchy-coated savanna animals in the model's learned feature space.

## Key Takeaways

- Model architecture choice involves real tradeoffs between size, inference speed, and accuracy — but "more accurate on average" does not mean "correct on every input."
- Diagnosing a misclassification benefits from systematically isolating variables (here: preprocessing, then architecture) rather than assuming the first plausible explanation.
- Understanding known model behaviours (like texture bias) helps explain *why* a model fails, not just *that* it fails — a more useful skill than chasing a single "fix."

## Files

- `brain.py` — baseline classification using ImageAI's ImageClassification API (MobileNetV2)
- `resize_test.py` — squish-resize preprocessing experiment (raw PyTorch, MobileNetV2)
- `resnet_test.py` — ResNet50 comparison experiment (raw PyTorch, official preprocessing)
- `giraffe.jpg`, `godzilla.jpg`, `house.jpg` — test images

## How to Run

**1. Set up the environment**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install cython pillow numpy opencv-python tqdm scipy matplotlib
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install imageai==3.0.3
```

**2. Download the MobileNetV2 weights**

Required for `brain.py` and `resize_test.py`. Not included in this repo due to file size — download it directly and place it in the project folder:

```bash
curl.exe -L -o mobilenet_v2-b0353104.pth "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
```

**3. ResNet50 weights (for `resnet_test.py`)**

No manual download needed — torchvision automatically downloads and caches the ResNet50 weights (~98MB) the first time the script runs.

**4. Run any script**

```bash
python brain.py
python resize_test.py
python resnet_test.py
```
