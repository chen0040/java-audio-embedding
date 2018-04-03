package com.github.chen0040.tensorflow.search;

import com.github.chen0040.tensorflow.search.models.AudioSearchEngine;
import com.github.chen0040.tensorflow.search.models.AudioSearchEntry;

import java.io.File;
import java.util.List;

public class AudioSearchEngineDemo {
    public static void main(String[] args){
        AudioSearchEngine searchEngine = new AudioSearchEngine();
        if(!searchEngine.loadIndexDbIfExists()) {
            searchEngine.indexAll(new File("music_samples").listFiles());
            searchEngine.saveIndexDb();
        }

        int pageIndex = 0;
        int pageSize = 20;
        boolean skipPerfectMatch = true;
        File f = new File("mp3_samples/example.mp3");
        System.out.println("querying similar music to " + f.getName());
        List<AudioSearchEntry> result = searchEngine.query(f, pageIndex, pageSize, skipPerfectMatch);
        for(int i=0; i < result.size(); ++i){
            System.out.println("# " + i + ": " + result.get(i).getPath() + " (distSq: " + result.get(i).getDistance() + ")");
        }
    }
}
