# Text-Recognition-Practice

---

## Requirements
- Python 3.7.3
- torch 1.2.0
- torchvision 0.4.0
- scikit-learn 0.20.3
- tqdm 4.31.1
---

## Task
Recognize the words given by images.

---

## Train
```
python main.py 
```

---
## Note

### Architecture
I use CRNN as the backbone to train the text recognition model.

### Pre-processing
According to [clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/dataset.py) to rize the image to [32 x 100].

### Metric
Evaluation metric according to [Shi, Baoguang, et al. "Aster: An attentional scene text recognizer with flexible rectification."](https://ieeexplore.ieee.org/document/8395027), use lower cases word accuracy.

---

## Problem

When I trained the model, I found it was serious overfitting.
I simplified the model and got slightly improved results.

Guess two possible causes: **Insufficient data set** and **The model's generalization ability is not enough**

### Insufficient data set
DCGAN can be used to increase the number of pictures. The quality of the images produced by the current experiment is not good, so this method is temporarily put aside.

### The model's generalization ability is not enough
The reason for this surmise is because when I tune up the dropout rate, the accuracy is better than lower dropout rate.
But there is still a limit.

---

## TODO

Check the program structure for errors. If not, the problem is with the model.
More simplified model or augment the training set by GAN.


















