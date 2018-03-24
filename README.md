# java-audio-embedding

Audio embedding in Java

# Usage

### Convert an audio file to mel-spectrogram image

The following sample codes convert the audio file audio.au into a mel-spectrogram image:

```java
import com.github.chen0040.tensorflow.audio.MelSpectrogram;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

MelSpectrogram melGram = new MelSpectrogram();
BufferedImage image = melGram.convertAudio(new File("samples/audio.au"));
File outputFile = new File("outputs/saved.png");
ImageIO.write(image, "png", outputFile);
```


