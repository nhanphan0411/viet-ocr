# handwritten-vns-recognization
BiLSTM DNN model to recognize handwritten Vietnamese. Dataset is acquired from Cinnamon AI Challenge. 

## Project Structure 
|--data/
|----raw/
|------0825_DataSamples_1/
|------0916_DataSamples_2/
|------1015_Private_Test/
|--src/

## Data preprocessing
Move to /src and run this to transform the data

```
python transform.py --path ../data/raw/0916_DataSamples_2 --type train --transform
python transform.py --path ../data/raw/1015_Private_Test --type test --transform
```
Two new folders train/ and test/ and two json files containing the labels will be created in data/. The folders train/ and test/ contain the preprocessed images. You can also run

```
python transform.py --path ../data/raw/0825_DataSamples_1 --type val --transform
```
to create a val/ set with 15 samples.

## Showing Examples

```
python transform.py --type [train or test or val] --sample
```
This will open a OpenCV window showing the preprocessed images (50 samples) one by one. The labels of the images will be shown in the terminal window.

## Train 
To start training the model, run:

```
python train.py --train
```

For testing, run:

```
python train.py --test --path [path to the test images]
```
Example python3 train.py --test --path ../data/test. Then predicted texts and the ground true texts will be stored in predictions_text.txt.
