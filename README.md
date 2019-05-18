* files starting with lower-case letters are tutorials for packages
* files starting with capitalized letters are sample codes for respective topics
# TOC
1. Autoencoder.ipynb
  * MNIST w/ torch
    * Load Data
    * Linear Autoencoder
      * Train
      * Evaluate
    * Convolutional Autoencoder
      * Train
      * Evaluate
    * Denoising Convolutional Autoencoder
      * Train
      * Evaluate
2. CNN.ipynb
  * MNIST
    * Torch
      * Load Data
      * Model
      * Train
      * Evaluate
    * Keras
      * Load data
      * Model
      * Train
      * Evaluate
  * CIFAR-10
    * Torch
      * Load Data
      * Model
      * Train
      * Evaluate
  * Transfer DenseNet to cats/dogs
    * Load data
    * Load Model
    * Train
  * Transfer VGG16 to Flowers
    * Load data
    * Load VGG19 Model
    * Train
    * Evaluate
3. MLP.ipynb
  * MNIST
    * Torch
      * Load Data
      * Model
      * Train
      * Evaluate
    * Keras
      * Load Data
      * Model
      * Train
      * Evaluate
4. NLP.ipynb
  * Logistic Regression on imdb for sentiment
    * BOW - 1-gram
    * 2-gram?
  * Classification by Neural Network
  * Language modelling by MLP
  * Sequence generation/tagging by RNN
5. RNN.ipynb
  * Sine Curve
    * Torch
      * Generate Data
      * Constructing 1-layer Model
      * Train
      * Generate
  * char-rnn
    * Torch
      * Data
      * # Preprocess
      * Model
      * Train
      * Prediction / Text generation
    * Keras
      * Data
      * Model
      * Train
      * Test
  * HRNN (https://github.com/fchollet/keras/blob/master/examples/mnist_hierarchical_rnn.py)
    * Data
    * Model
6. Reinforcement.ipynb
  * Catch (https://gist.github.com/EderSantana/c7222daa328f0e885093)
      * Evaluation
      * Visualization
  * Flappy
7. StyleTransfer.ipynb
  * Torch
    * Load VGG19 and Images
    * Extracting style/content features from vgg
    * Defining and Optimizing wrt Loss function
    * Final Display
8. gensim.ipynb
  * Preprocess
    * Tokenize
    * Vectorize Corpora
  * Transformations
    * Tfidf (term frequency–inverse document frequency)
    * <a href=\"https://radimrehurek.com/gensim/models/lsimodel.html#module-gensim.models.lsimodel\">lsi (Latent Semantic Indexing)</a>
    * <a href=\"https://radimrehurek.com/gensim/models/ldamodel.html\">lda (Latent Dirichlet Allocation)</a>
    * <a href=\"https://radimrehurek.com/gensim/models/hdpmodel.html\">hdp (Hierarchical Dirichlet Process)</a>
  * Similarities
  * <a href=\"https://radimrehurek.com/gensim/models/phrases.html\">Phrase Modelling</a> 
  * Word2Vec
    * pretrained by nltk
    * pretrained by Google
    * pretrained GloVe by Stanford
    * self-trained Word2Vec
  * Topic Modelling
    * Internal LDA
    * umass Mallet
9. keras.ipynb
  * Data Preparation
  * Model Building
    * Sequential Building
    * Functional API ~nngraph
    * Common Model Properties
    * Common Layer Properties
  * Training
  * Custom Evaluation
10. nltk.ipynb
  * Corpora
    * Samples
    * Special Corpora (\"Lexical Resources\")
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
    * MLE $P(w|v)=\\frac{P(v,w)}{P(w)}$
    * Laplace smoothing $P(w|v)=\\frac{P(v,w)+1}{P(w)+V}$
    * Kneser-Ney Interpolated
  * Sequence Tagging
    * Tagged Corpora
    * Custom Tagger
    * N-gram Tagging
    * Probabilistic Graphical Model
      * Traing HMM
  * Sentiment
    * VADER
11. othernlp.ipynb
  * Stanford NLP
  * TextBlob for sentiment
  * Spacy for preprocessing
  * scikit-learn
12. pytorch.ipynb
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
13. tensorflow.ipynb
  * Keras from tf
  * sklearn interface
