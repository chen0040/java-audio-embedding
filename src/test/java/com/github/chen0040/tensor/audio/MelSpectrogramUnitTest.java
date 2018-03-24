package com.github.chen0040.tensor.audio;

import com.github.chen0040.tensorflow.audio.MelSpectrogram;
import org.testng.annotations.Test;

import javax.imageio.ImageIO;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MelSpectrogramUnitTest {

    @Test
    public void generateMelSpetrogram() throws UnsupportedAudioFileException, IOException, LineUnavailableException {
        MelSpectrogram melGram = new MelSpectrogram();
        BufferedImage image = melGram.convertAudio(FileUtils.getAudioFile());
        File outputFile = new File("saved.png");
        ImageIO.write(image, "png", outputFile);
    }
}
