package com.github.chen0040.tensor.audio.tarsos;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Mixer;
import javax.swing.ButtonGroup;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;
import javax.swing.border.TitledBorder;

public class InputPanel extends JPanel {

    /**
     *
     */
    private static final long serialVersionUID = 1L;

    Mixer mixer = null;

    public InputPanel(){
        super(new BorderLayout());
        this.setBorder(new TitledBorder("1. Choose a microphone input"));
        JPanel buttonPanel = new JPanel(new GridLayout(0,1));
        ButtonGroup group = new ButtonGroup();
        for(Mixer.Info info : Shared.getMixerInfo(false, true)){
            JRadioButton button = new JRadioButton();
            button.setText(Shared.toLocalString(info));
            buttonPanel.add(button);
            group.add(button);
            button.setActionCommand(info.toString());
            button.addActionListener(setInput);
        }
        this.add(new JScrollPane(buttonPanel,JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,JScrollPane.HORIZONTAL_SCROLLBAR_NEVER),BorderLayout.CENTER);
        this.setMaximumSize(new Dimension(300,150));
        this.setPreferredSize(new Dimension(300,150));
    }

    private ActionListener setInput = new ActionListener(){

        public void actionPerformed(ActionEvent arg0) {
            for(Mixer.Info info : Shared.getMixerInfo(false, true)){
                if(arg0.getActionCommand().equals(info.toString())){
                    Mixer newValue = AudioSystem.getMixer(info);
                    InputPanel.this.firePropertyChange("mixer", mixer, newValue);
                    InputPanel.this.mixer = newValue;
                    break;
                }
            }
        }
    };

}
