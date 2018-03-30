package com.github.chen0040.tensorflow.classifiers.models;

import java.io.IOException;
import java.io.InputStream;

public interface TrainedModelLoader {
    void load_model(InputStream inputStream) throws IOException;
}
