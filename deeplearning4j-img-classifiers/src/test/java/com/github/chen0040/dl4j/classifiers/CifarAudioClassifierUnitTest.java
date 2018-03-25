package com.github.chen0040.dl4j.classifiers;

import org.testng.annotations.Test;

import java.io.IOException;

public class CifarAudioClassifierUnitTest {
    @Test
    public void testFit() throws IOException {
        CifarAudioClassifier classifier = new CifarAudioClassifier();
        classifier.fit_images("gtzan/genres", new String[] { ".png"});
    }
}
