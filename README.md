The model here we have used is built with Keras using Convolutional
Neural Networks (CNN).

-> A convolutional neural network is a Deep Learning algo. which can take in an input image, assign importance (learnable weights and biases) to
   various aspects/objects in the image and be able to differentiate one from the other.
   The pre-processing required in a CNN is much lower as compared to other classification algo.

-> The haar cascade files are the XML files that are needed to detect objects from the image. Here we are detecting the eyes and face of the 
   person in real-time video.


In this project, we are having two python scripts namely: model.py and drowsiness detection.py.

model.py consists of codes, we are using to train the model.

And the second file is the main file which uses the model trained in model.py and captures the real-time eyes and face, accordingly
shows the result.

The project is almost ready but it has some bugs left like while wearing spectacles (as we have not tested it with sunglasses)
this system does not work when we are far.
But when we come closer to the camera, the system detects eyes with the glasses on.

And we will solve this issue and make the system almost perfect in coming days.


