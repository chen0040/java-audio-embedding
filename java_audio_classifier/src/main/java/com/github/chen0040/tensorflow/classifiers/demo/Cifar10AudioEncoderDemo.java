package com.github.chen0040.tensorflow.classifiers.demo;

import com.github.chen0040.tensorflow.classifiers.models.cifar10.Cifar10AudioClassifier;
import com.github.chen0040.tensorflow.classifiers.utils.ResourceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Cifar10AudioEncoderDemo {

    private static final Logger logger = LoggerFactory.getLogger(Cifar10AudioEncoderDemo.class);

    private static List<String> getAudioFiles() {
        List<String> result = new ArrayList<>();
        File dir = new File("music_samples");
        System.out.println(dir.getAbsolutePath());
        if (dir.isDirectory()) {

            for (File f : dir.listFiles()) {
                String file_path = f.getAbsolutePath();
                if (file_path.endsWith("au")) {
                    result.add(file_path);

                }
            }
        }

        return result;
    }

    public static void main(String[] args) throws IOException {
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
    }
}
