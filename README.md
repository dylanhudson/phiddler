# phiddler
Generating traditional celtic tunes with LSTM neural nets.

This project is mostly cloned from Hannah Shaw's wonderful tutorial on generative poetry - https://vivshaw.github.io/blog/electric-pentameter/

The orginal dataset comes from Jeremy's great work at The Session -
https://github.com/adactio/TheSession-data


Training took about 45 minutes/epoch on an AWS ec2 p2.xlarge GPU Ubuntu deep learning instance , using a TensorFlow/CUDA backend.


The files:

slicetunes.py extracts just the abc-formatted notes (and in this case, just from Reels to avoid potential Cronenberged odd-meter monstrosities).

build_models.py prepares the data and trains the neural net

generate.py seeds the model and produces several bars of raw abcjs formatted notes. 





