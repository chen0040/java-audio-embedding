# java-audio-embedding

Audio embedding in Java

The current project attempts to develop a pure Java audio encoder that can be used in pure Java or Android program. 
Such an audio encoder can be used for music genres classification or music search, or music recommend-er.

The current project contains currently two deep learning networks adopted from:

* resnet
* cifar

The training and validation of these two models are showned below:

![compare-history](keras_audio_classifier/demo/models/training-history-comparison.png)

# Usage

### Train audio classifier in Keras

The machine learning package in Java is tensorflow, it loads a pre-trained audio classifier model (.pb format).
The audio classifier model was originally implemented and trained using Keras in Python. This trained
classifier model (in .h5 format) was then converted to .pb model file which can be directly loaded by tensorflow in Java.

The keras training of audio classifier model can be found in [README_Training.md](README_Training.md)

### Run audio classifier in Java
 
The [sample codes](java_audio_classifier/src/main/java/com/github/chen0040/tensorflow/classifiers/demo/Cifar10AudioClassifierDemo.java) 
below shows how to use the cifar audio classifier to predict the genres of music:

```java
import com.github.chen0040.tensorflow.classifiers.models.cifar10.Cifar10AudioClassifier;
import com.github.chen0040.tensorflow.classifiers.utils.ResourceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

InputStream inputStream = ResourceUtils.getInputStream("tf_models/cifar10.pb");
Cifar10AudioClassifier classifier = new Cifar10AudioClassifier();
classifier.load_model(inputStream);

List<String> paths = getAudioFiles();

Collections.shuffle(paths);

for (String path : paths) {
    System.out.println("Predicting " + path + " ...");
    File f = new File(path);
    String label = classifier.predict_audio(f);

    System.out.println("Predicted: " + label);
}
```  

 
The [sample codes](java_audio_classifier/src/main/java/com/github/chen0040/tensorflow/classifiers/demo/ResNetV2AudioClassifierDemo.java) 
below shows how to use the resnet v2 audio classifier to predict the genres of music:

```java
import com.github.chen0040.tensorflow.classifiers.resnet_v2.ResNetV2AudioClassifier;
import com.github.chen0040.tensorflow.classifiers.utils.ResourceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

InputStream inputStream = ResourceUtils.getInputStream("tf_models/resnet-v2.pb");
ResNetV2AudioClassifier classifier = new ResNetV2AudioClassifier();
classifier.load_model(inputStream);

List<String> paths = getAudioFiles();

Collections.shuffle(paths);

for (String path : paths) {
    System.out.println("Predicting " + path + " ...");
    File f = new File(path);
    String label = classifier.predict_audio(f);

    System.out.println("Predicted: " + label);
}
```  

### Extract features from audio in Java

The [sample codes](java_audio_classifier/src/main/java/com/github/chen0040/tensorflow/classifiers/demo/Cifar10AudioEncoderDemo.java) 
below shows how to use the cifar audio classifier to encode an audio file into an float array:

```java
import com.github.chen0040.tensorflow.classifiers.models.cifar10.Cifar10AudioClassifier;
import com.github.chen0040.tensorflow.classifiers.utils.ResourceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

InputStream inputStream = ResourceUtils.getInputStream("tf_models/cifar10.pb");
Cifar10AudioClassifier classifier = new Cifar10AudioClassifier();
classifier.load_model(inputStream);

List<String> paths = getAudioFiles();

Collections.shuffle(paths);

for (String path : paths) {
    System.out.println("Encoding " + path + " ...");
    File f = new File(path);
    float[] encoded_audio = classifier.encode_audio(f);

    System.out.println("Encoded: " + Arrays.toString(encoded_audio));
}
```  

 
The [sample codes](java_audio_classifier/src/main/java/com/github/chen0040/tensorflow/classifiers/demo/ResNetV2AudioEncoderDemo.java) 
below shows how to the resnet v2 audio classifier to encode an audio file into an float array:

```java
import com.github.chen0040.tensorflow.classifiers.resnet_v2.ResNetV2AudioClassifier;
import com.github.chen0040.tensorflow.classifiers.utils.ResourceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

InputStream inputStream = ResourceUtils.getInputStream("tf_models/resnet-v2.pb");
ResNetV2AudioClassifier classifier = new ResNetV2AudioClassifier();
classifier.load_model(inputStream);

List<String> paths = getAudioFiles();

Collections.shuffle(paths);

for (String path : paths) {
    System.out.println("Encoding " + path + " ...");
    File f = new File(path);
    float[] encoded_audio = classifier.encode_audio(f);

    System.out.println("Encoded: " + Arrays.toString(encoded_audio));
}
```  

### Audio Search Engine

The [sample codes](java_audio_search/src/main/java/com/github/chen0040/tensorflow/search/AudioSearchEngineDemo.java) 
below shows how to index and search for audio file using the [AudioSearchEngine](java_audio_search/src/main/java/com/github/chen0040/tensorflow/search/models/AudioSearchEngine.java) class:

```java
AudioSearchEngine searchEngine = new AudioSearchEngine();
if(!searchEngine.loadIndexDbIfExists()) {
    searchEngine.indexAll(new File("music_samples").listFiles());
    searchEngine.saveIndexDb();
}

int pageIndex = 0;
int pageSize = 20;
boolean skipPerfectMatch = true;
for(File f : new File("music_samples").listFiles()) {
    System.out.println("querying similar music to " + f.getName());
    List<AudioSearchEntry> result = searchEngine.query(f, pageIndex, pageSize, skipPerfectMatch);
    for(int i=0; i < result.size(); ++i){
        System.out.println("# " + i + ": " + result.get(i).getPath() + " (distSq: " + result.get(i).getDistanceSq() + ")");
    }
}
```  

### Music Recommend-er

The [sample codes](java_audio_recommender/src/main/java/com/github/chen0040/tensorflow/search/KnnAudioRecommenderDemo.java) 
below shows how to recommend musics based on user's music history using the [KnnAudioRecommender](java_audio_recommender/src/main/java/com/github/chen0040/tensorflow/search/models/KnnAudioRecommender.java) class:

```java
AudioUserHistory userHistory = new AudioUserHistory();

List<String> audioFiles = FileUtils.getAudioFiles();
Collections.shuffle(audioFiles);

for(int i=0; i < 40; ++i){
    String filePath = audioFiles.get(i);
    userHistory.logAudio(filePath);
    try {
        Thread.sleep(100L);
    } catch (InterruptedException e) {
        e.printStackTrace();
    }
}

KnnAudioRecommender recommender = new KnnAudioRecommender();
if(!recommender.loadIndexDbIfExists()) {
    recommender.indexAll(new File("music_samples").listFiles(a -> a.getAbsolutePath().toLowerCase().endsWith(".au")));
    recommender.saveIndexDb();
}

System.out.println(userHistory.head(10));

int k = 10;
List<AudioSearchEntry> result = recommender.recommends(userHistory.getHistory(), k);

for(int i=0; i < result.size(); ++i){
    AudioSearchEntry entry = result.get(i);
    System.out.println("Search Result #" + (i+1) + ": " + entry.getPath());
}

```











