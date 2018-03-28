# java-audio-embedding

Audio embedding in Java

The current project attempts to develop a pure Java audio encoder that can be used in pure Java or Android program. 
Such an audio encoder can be used for music genres classification or music search, or music recommend-er.

The machine learning package in Java is tensorflow, it loads a pre-trained audio classifier model (.pb format).
The audio classifier model was originally implemented and trained using Keras in Python. This trained
classifier model (in .h5 format) was then converted to .pb model file which can be directly loaded by tensorflow in Java.

The keras training of audio classifier model can be found in [README_Training.md](README_Training.md)

# Usage








