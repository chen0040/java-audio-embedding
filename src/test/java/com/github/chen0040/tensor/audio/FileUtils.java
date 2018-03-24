package com.github.chen0040.tensor.audio;

import java.io.File;

public class FileUtils {
    public static File getAudioFile() {
        String path = FileUtils.class.getClassLoader().getResource("audio_samples/blues.00000.au").getFile();
        return new File(path);
    }
}
