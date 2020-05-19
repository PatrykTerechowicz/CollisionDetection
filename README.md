# Collision Detector

This repo contains various scripts used to train CNN with own dataset. Trained CNN is Alexnet with 3 outputs, each of the outputs corresponds
to respectively: turn_left, go_forward, turn_right. CNN is fed with 224x224 3 channel images.
DataLabeler.py contains simple GUI app that can be used to classify your dataset and create .csv train file.


Robot running CNN trained with theese scripts can be seen on video: https://youtu.be/Vwwrvt3d2h8

As you can see CNN alone isn't enough for controlling robot, path planning and SLAM would upgrade robot's performance, which are my current focus for now.
