# Handwritten Vietnamese OCR

Implementation of Recurrent Neural Network in recognizing Vietnamese handwritings. The dataset is provided by CinnamonAI, within their [Hackathon](https://drive.google.com/drive/folders/1Qa2YA6w6V5MaNV-qxqhsHHoYFRK5JB39), 2018. 

![](https://i.imgur.com/OsHc8Vs.png)

![](https://i.imgur.com/1BjP57K.png)

<br>

## ❊ RESULT

The project successfully achieved 
- Character Error Rate: 0.04
- Word Error Rate: 0.14 
- Sentence Error Rate: 0.82

The [hackathon's winner](https://pbcquoc.github.io/vietnamese-ocr/) score is 0.1x on the Word Error Rate. Other metric results were not disclosed.

Sample predictions: *above - label, below - prediction*
<img src='https://i.imgur.com/SiLsa68.png'>


<br>

## ⌘ PRE-PROCESS DATA

- Thresholding using OpenCV 
- Resize into new size of (128, 1024, 1)
- *(optional)* Remove Recursive (reference to [A. Vinciarelli and J. Luettin](http://www.dcs.gla.ac.uk/~vincia/papers/normalization.pdf))

Original

<img src='https://i.imgur.com/pM7uo7o.png'>

Preprocessed

<img src='https://i.imgur.com/KmXMYX0.png'>

- Preprocess on the official dataset
```
python transform.py --path ../data/raw/0916_DataSamples_2 --type train --transform
python transform.py --path ../data/raw/1015_Private_Test --type test --transform
```
*Two new folders train/ and test/ and two json files containing the labels will be created in data/. The folders train/ and test/ contain the preprocessed images.* 

- To create a validation set of 15 sample images 
```
python transform.py --path ../data/raw/0825_DataSamples_1 --type val --transform
```
- Show a sample of 50 preprocessed images. 
```
python transform.py --type [train or test or val] --sample
```

<br>

## 🕸 MODEL

CRNN + CTC Loss is used to solve this challenge. The CRNN model comprised.
The CNN blocks with skip connections (inspired by ResNet50) are used to extract the features from the input image. After that, the feature map will be flattened into the Bi-directional LSTM layers.



<br>

## 🧠 TRAIN 

```
python train.py --train
```

I trained the model for 30 epochs with learning_rate of 1e-3, then after that decay it to 1e-5. Clearly. the training could have been stopped early at epoch 20. 

<img src='https://i.imgur.com/wjojtNZ.png'>

<br>

## 🤘🏻 TEST

```
python train.py --test --path [path to the test images]
```
*Example* `python3 train.py --test --path ../data/test.`

Then predicted texts and the ground truth texts will be stored in predictions_text.txt
