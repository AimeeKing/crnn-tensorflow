# crnn-tensorflow
This software implements the Convolutional Recurrent Neural Network (CRNN) in tensorflow.Origin software could be found in [crnn](https://github.com/bgshih/crnn)

## run demo

A demo program can be found in demo_mjsyth.py.Before running the demo,you should download a a pretrained model (I would upload next time) to Put the downloaded model file into directory ./tmp

The demo reads an example image and recognizes its text content.

for example:
![demo](https://github.com/AimeeKing/crnn-tensorflow/blob/master/demo/1_Evangelically_26825.jpg?raw=true)

output:
label evangelically

识别结果 evangelically

## Train a new model

1.you could use utils in ./dataset to convert your dataset to tfrecords and save them in ./data

