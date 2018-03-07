package com.github.chen0040.tensor.audio.tarsos;

import java.awt.GridLayout;
import java.awt.event.ActionListener;

import javax.swing.ButtonGroup;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.border.TitledBorder;

import be.tarsos.dsp.pitch.PitchProcessor.PitchEstimationAlgorithm;

public class PitchDetectionPanel extends JPanel {

    /**
     *
     */
    private static final long serialVersionUID = -5107785666165487335L;

    private PitchEstimationAlgorithm algo;

    public PitchDetectionPanel(ActionListener algoChangedListener){
        super(new GridLayout(0,1));
        setBorder(new TitledBorder("2. Choose a pitch detection algorithm"));
        ButtonGroup group = new ButtonGroup();
        algo = PitchEstimationAlgorithm.YIN;
        for (PitchEstimationAlgorithm value : PitchEstimationAlgorithm.values()) {
            JRadioButton button = new JRadioButton();
            button.setText(value.toString());
            add(button);
            group.add(button);
            button.setSelected(value == algo);
            button.setActionCommand(value.name());
            button.addActionListener(algoChangedListener);
        }
    }
}
