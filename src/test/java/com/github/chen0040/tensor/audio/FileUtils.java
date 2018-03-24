package com.github.chen0040.tensor.audio;

import java.io.File;

public class FileUtils {
    public static File getAudioFile() {
        String path = FileUtils.class.getClassLoader().getResource("audio_samples/example.mp3").getFile();
        return new File(path);
    }
}
