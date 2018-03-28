# java-audio-embedding

Audio embedding in Java

The current project attempts to develop a pure Java audio encoder that can be used in pure Java or Android program. 
Such an audio encoder can be used for music genres classification or music search, or music recommend-er.



# Usage

### Audio classifier trained model in Keras

The machine learning package in Java is tensorflow, it loads a pre-trained audio classifier model (.pb format).
The audio classifier model was originally implemented and trained using Keras in Python. This trained
classifier model (in .h5 format) was then converted to .pb model file which can be directly loaded by tensorflow in Java.

The keras training of audio classifier model can be found in [README_Training.md](README_Training.md)

### Audio Classifier 
The [sample codes](java_audio_classifier/src/main/java/com/github/chen0040/tensorflow/classifiers/demo/Cifar10ImageClassifierDemo.java) 
below shows how to use the audio classifier to predict the genres of music:

```java

```  








