* files starting with capitalized letters are sample codes for respective topics
* files starting with lower-case letters are tutorials for packages
# Table of Contents
## Topics
### [Autoencoder.ipynb](https://github.com/hoihui/tutorial/blob/master/Autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/Autoencoder.ipynb)
  * MNIST w/ torch
    * Load Data
    * Linear Autoencoder
    * Convolutional Autoencoder
    * Denoising Convolutional Autoencoder
  * MNIST Conv VAE w/ tensorflow
### [CNN.ipynb](https://github.com/hoihui/tutorial/blob/master/CNN.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/CNN.ipynb)
  * MNIST
    * Torch
    * Keras
  * [Human or horse](https://github.com/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb)
    * Data
    * Model
    * Train
    * Evaluate
  * CIFAR-10
    * Torch
  * Cats and dogs
    * CNN w/ keras
    * Transfer DenseNet w/ torch
    * Transfer InceptionV3 w/ keras
    * Transfer MobileNetV2 w/ tensorflow
  * Transfer VGG16 to Flowers
    * Load VGG19 Model
    * Train
    * Evaluate
### [DeepRL.ipynb](https://github.com/hoihui/tutorial/blob/master/DeepRL.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/DeepRL.ipynb)
  * Implementations
    * DQN -- deep network for action-value function (Q) with off-policy learning
    * DRQN -- deep recurrent Q network
    * Deep Policy Network, MC
    * A2C = Deep policy+value (Actor-Critic) networks
  * [Catch](https://gist.github.com/EderSantana/c7222daa328f0e885093) using raw pixels
    * Setup Environment
    * Various Models
    * Visualization
  * [CartPole](https://gym.openai.com/envs/CartPole-v0/)
    * Network on default state representation
    * CNN value network on raw pixels
### [GAN.ipynb](https://github.com/hoihui/tutorial/blob/master/GAN.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/GAN.ipynb)
  * MNIST, MLP
    * Load Data
    * Model
    * Train
    * Evaluate
  * MNIST, Convolutional
    * Load Data
    * Model
    * Train
  * Street View House Number, Convolutional
    * Data
    * Model
    * Train
    * Evaluate
### [ImageTranslation.ipynb](https://github.com/hoihui/tutorial/blob/master/ImageTranslation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/ImageTranslation.ipynb)
  * Pix2Pix on building facades w/ tensorflow
    * Data
    * Model
    * Train
  * CycleGAN on Yosemite's summer/winter w/ torch
    * Data
    * Model
    * Train
    * Evaluate
### [MCTS.ipynb](https://github.com/hoihui/tutorial/blob/master/MCTS.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/MCTS.ipynb)
  * Barebone MCTS with UCT
    * Code: [mcts.ai](http://mcts.ai/code/python.html) recursive + node reuse + pruning
    * [m,n,k-game](https://en.wikipedia.org/wiki/M,n,k-game)
    * Connect4
    * Othello
  * MCTS with Multiprocessing
  * MCTS written in Cython
    * mcts module
    * m,n,k-game
    * Othello
  * MCTS in Cython with multiprocessing
    * mcts module
    * m,n,k-game
    * Othello
  * AlphaZero-style (guide tree search by policy+value network)
    * [Code](http://web.stanford.edu/~surag/posts/alphazero.html) (modified)
    * [m,n,k-game](https://en.wikipedia.org/wiki/M,n,k-game)
    * [Connect4](https://github.com/plkmo/AlphaZero_Connect4)
    * [Othello](https://github.com/suragnair/alpha-zero-general)
### [MLP.ipynb](https://github.com/hoihui/tutorial/blob/master/MLP.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/MLP.ipynb)
  * MNIST
    * Torch
    * Keras
### [NLP.ipynb](https://github.com/hoihui/tutorial/blob/master/NLP.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/NLP.ipynb)
  * Logistic Regression on imdb for sentiment
    * BOW - 1-gram
    * 2-gram?
  * Word2Vec
    * Data
    * SkipGram using Torch
    * SkipGram + Negative Sampling
    * Visualization
  * Word Embedding + sentiment of imdb
    * Load Data
    * Model
    * Train
    * Evaluation
  * Classification by CNN
  * Language modelling by MLP
### [Reinforcement.ipynb](https://github.com/hoihui/tutorial/blob/master/Reinforcement.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/Reinforcement.ipynb)
  * David Silver's Lectures
    * [Lecture 1](https://www.youtube.com/watch?v=2pWv7GOvuf0) [Intro](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/intro_RL.pdf)
    * [Lecture 2](https://www.youtube.com/watch?v=lfHX2hHRMVQ) [Markov Decision Processes](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf)
    * [Lecture 3](https://www.youtube.com/watch?v=Nd1-UUMVfz4) [Planning by Dynamic Programming](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf)
    * [Lecture 4](https://www.youtube.com/watch?v=PnHCvfgC_ZA) [Model-Free Prediction](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf)
    * [Lecture 5](https://www.youtube.com/watch?v=0g4j2k_Ggc4) [Model Free Control](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf)
    * [Lecture 6](https://www.youtube.com/watch?v=UoPei5o4fps) [Value Function Approximation](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/FA.pdf)
    * [Lecture 7](https://www.youtube.com/watch?v=KHZVXao4qXs) [Policy Gradient Methods](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf)
    * [Lecture 8](https://www.youtube.com/watch?v=ItMutbeOHtc) [Integrating Learning and Planning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/dyna.pdf)
    * [Lecture 9](https://www.youtube.com/watch?v=sGuiWX07sKw) [Exploration and Exploitation](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/XX.pdf)
    * [Lecture 10](https://www.youtube.com/watch?v=N1LKLc6ufGY) [Classic Games](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/games.pdf)
  * Examples
    * [Mountain Car by discrete Q-learning](https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/)
    * [Cart-Pole by Monte Carlo Q-table](https://gist.githubusercontent.com/karpathy/868459dad1883098fe55806e69f40c91/raw/af6730c668cc8674765e1bef0bc45aa9c598d954/gistfile1.py)
    * Policy Evaluation (Lecture 1.3-1.4)
    * Policy Gradient on CartPole (Lecture 1.7)
### [SequenceToNumbers.ipynb](https://github.com/hoihui/tutorial/blob/master/SequenceToNumbers.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/SequenceToNumbers.ipynb)
  * Sine Curve
    * Torch
  * char-rnn
    * Torch
    * Keras
  * IMDB Sentiment w/ torch
    * Data
    * Model
    * Train
    * Evaluate
  * IMDB sentiment w/ keras
    * Load Data
    * Models
    * Train and Evaluate
  * [HRNN](https://github.com/fchollet/keras/blob/master/examples/mnist_hierarchical_rnn.py)
    * Data
    * Model
### [SequenceToSequence.ipynb](https://github.com/hoihui/tutorial/blob/master/SequenceToSequence.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/SequenceToSequence.ipynb)
  * Self-generated data w/ Keras
    * Data Generator
    * Models
    * Train and Evaluate
  * Language translation
    * Data
    * Train
    * Evaluate
### [StyleTransfer.ipynb](https://github.com/hoihui/tutorial/blob/master/StyleTransfer.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/StyleTransfer.ipynb)
  * Deep Dreaming w/ Torch
    * Load Images and VGG19
  * Deep Dreaming w/ Tensorflow
  * Style Transfer w/ Torch
    * Load Images and VGG19
    * Extracting style/content features from vgg
    * Defining and Optimizing wrt Loss function
    * Final Display
  * Style Transfer w/ Tensorflow
    * Load Image
    * Load Model
    * Extract style/content
    * Basic training
    * Total variation loss
## Packages
### [cython.ipynb](https://github.com/hoihui/tutorial/blob/master/cython.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/cython.ipynb)
  * pyx compilation
    * `pxd` header file
    * [in notebook](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiling-with-a-jupyter-notebook)
    * `cython` function that works even if `gcc` compilation fails
  * using C/C++ libraries
    * [Using std C functions](https://cython.readthedocs.io/en/latest/src/tutorial/external.html)
    * STL
    * [Using self-defined C classes (externs)](https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html)
    * [define class in cython](https://cython.readthedocs.io/en/latest/src/tutorial/cdef_classes.html)
  * Examples
  * openmp
  * Using [Numpy](http://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html)
    * Pairwise Distances (numpy list)
### [datatable.ipynb](https://github.com/hoihui/tutorial/blob/master/datatable.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/datatable.ipynb)
### [gensim.ipynb](https://github.com/hoihui/tutorial/blob/master/gensim.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/gensim.ipynb)
  * Preprocess
    * Tokenize
    * Vectorize Corpora
  * Transformations
    * Tfidf (term frequency-inverse document frequency)
    * <a href="https://radimrehurek.com/gensim/models/lsimodel.html#module-gensim.models.lsimodel">lsi (Latent Semantic Indexing)</a>
    * <a href="https://radimrehurek.com/gensim/models/ldamodel.html">lda (Latent Dirichlet Allocation)</a>
    * <a href="https://radimrehurek.com/gensim/models/hdpmodel.html">hdp (Hierarchical Dirichlet Process)</a>
  * Similarities
  * <a href="https://radimrehurek.com/gensim/models/phrases.html">Phrase Modelling</a>
  * Word2Vec
    * pretrained by nltk
    * pretrained by Google
    * pretrained GloVe by Stanford
    * self-trained Word2Vec
  * Topic Modelling
    * Internal LDA
    * umass Mallet
### [gym.ipynb](https://github.com/hoihui/tutorial/blob/master/gym.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/gym.ipynb)
  * Basics
    * [Environment](http://gym.openai.com/envs/)-specific
  * Wrapper
  * Custom environment
### [nltk.ipynb](https://github.com/hoihui/tutorial/blob/master/nltk.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/nltk.ipynb)
  * Corpora
    * Samples
    * Special Corpora ("Lexical Resources")
    * WordNet
    * Own Corpus
  * Text exploration statistics
    * Frequencies
    * n-grams
  * Preprocessing
    * Tokenization
    * Sentence Segmentation
    * Stemming
    * Lemmatization
  * ngram Language Modelling
    * MLE $P(w|v)=\frac{P(v,w)}{P(w)}$
    * Laplace smoothing $P(w|v)=\frac{P(v,w)+1}{P(w)+V}$
    * Kneser-Ney Interpolated
  * Sequence Tagging
    * Tagged Corpora
    * Custom Tagger
    * N-gram Tagging
    * Probabilistic Graphical Model
  * Sentiment
    * VADER
### [othernlp.ipynb](https://github.com/hoihui/tutorial/blob/master/othernlp.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/othernlp.ipynb)
  * Stanford NLP
  * TextBlob for sentiment
  * Spacy for preprocessing
  * scikit-learn
### [pytorch.ipynb](https://github.com/hoihui/tutorial/blob/master/pytorch.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/pytorch.ipynb)
  * Tensors
    * Numbers
    * Linear Alegra
    * Functions
    * Autograd
    * To/from Numpy
  * Datasets
    * Images
    * MNIST
    * Cats/Dogs
  * Constructing Network
    * Instantiate
    * Initialization
    * Forward
    * Using nn. without class
  * Training Network
    * Defining loss function
    * backprop
    * Optimizer
    * On GPU
  * Inference
  * Save / Load
    * Pretrained Models
### [stable-baseline.ipynb](https://github.com/hoihui/tutorial/blob/master/stable-baseline.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/stable-baseline.ipynb)
  * Common interface
    * multiprocessing
    * callbacks/Tensorboard
    * [Atari Game interface](https://stable-baselines.readthedocs.io/en/master/guide/examples.html#id1)
    * Normalize input (env wrapper)
    * [Custom "policy" network](https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html)
  * Examples
    * [Proximal Policy Optimization](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
    * [Deep Q Network](https://stable-baselines.readthedocs.io/en/master/modules/dqn.html)
    * Recurrent policies
### [tensorflow_keras.ipynb](https://github.com/hoihui/tutorial/blob/master/tensorflow_keras.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoihui/tutorial/blob/master/tensorflow_keras.ipynb)
  * Tensors
    * Numbers
    * Linear Alegra
    * Functions
    * Autograd
    * To/from Numpy
    * [TF function and AutoGraph](https://www.tensorflow.org/alpha/tutorials/eager/tf_function)
  * Datasets
    * [Keras.datasets](https://keras.io/datasets/)
    * [tensorflow_datasets](https://www.tensorflow.org/datasets/datasets)
    * Transformation (Augmentation)
    * Custom Dataset
    * images Loader
    * Download csv
  * Constructing Network
    * Prebuilt <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers">Layers</a>
    * Custom Layer
    * Sequential Building
    * Functional API build step by step as walkthrough
    * Common Model Properties
    * Common Layer Properties
    * Composite model specifying inputs/outputs
  * Training Network
    * Custom Training
    * Custom grad + [Keras Optimizer](https://keras.io/optimizers/)
    * Keras Loss & Optimizer
  * Save / load
    * Pretrained models from Built-in
    * Pretrained models from TFHub
  * Custom Evaluation
  * sklearn-style interface ?
