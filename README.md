# CNN-2DS
## Introduction
- __Shape classification of cloud particles recorded by 2D-S imaging probe using convolutional neural network__<br />
We proposed a new classification method for 2D-S cloud particle images using a convolutional neural network, called CNN-2DS. It can classify 8 types of ice crystal particles, including linear, sphere, dendrite, aggregate, graupel, plate, donut, and irregular. The results of the experiments show that CNN-2DS can accurately identify cloud particles with an average classification accuracy of 97%.

- *Example of 2D-S cloud particle images:* <br />
![Examples](https://github.com/haixiaxiao/CNN-2DS/blob/master/img/img1.png)

## Getting Started
__Examples of training and testing the model__<br />
### If you want to train model:<br />
```
sh train_2ds.sh
```
### If you want to test the trained model:<br />
```
sh test_2ds.sh
```
## Results
Using test code, you can obtain the corresponding accuracy, precision, recall, F1 value and confusion matrix. In addition, you can input any new 2DS ice crystal particle observations into [test.py](https://github.com/haixiaxiao/CNN-2DS/blob/master/test.py) for classification, and obtain the predicted category and the corresponding classified folder.

## Citation
If you find this project useful in your research, please consider cite:
```
To be updated...
```
