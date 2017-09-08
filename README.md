# TensorFlow-FlappyBird


A TensorFlow implementation of DQN for FlappyBird.  The code originally came from Keras-FlappyBird and has be adapted to now strictly use TensorFlow.

Please read the orinal author's [blog post](https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html) for details and check out the [original repository](https://github.com/yanpanlau/Keras-FlappyBird) for the inspiration behind this repo.

![](animation1.gif)

# Installation Dependencies:

* Python 2.7
* Keras 2.0 
* pygame
* scikit-image
* TensorFlow >= 1.2

# How to Run?

**Inference**

```
python qlearn.py -m "Run"
```

**Training**

If you want to train the network from beginning, delete the logdir and run 

```
python qlearn.py -m "Train"
```