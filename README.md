# Awesome-Rust-MachineLearning

Awesome Rust Machine Learning crate list that writing this with the idea of migrating from Python.

And reference of Machine Learning using Rust (blog, book, movie, discussion, ...and more).

It's a list with the major items at the top.

# ToC

- [Undercarriage](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#undercarriage)
    - [Jupyter Notebook](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#jupyter-notebook)
    - [Plot](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#plot)
    - [Vector](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#vector)
    - [Dataframe](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#dataframe)
    - [Image Processing](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#image-processing)
    - [Natural Language Processing (preprocessing)](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#natural-language-processing-preprocessing)
    - [Graph](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#graph)
    - [Interface](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#interface)
    - [Workflow](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#workflow)
- [Comprehensive (like sklearn)](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#comprehensive-like-sklearn)
- [Comprehensive (statistics)](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#comprehensive-statistics)
- [Gradient Boosting](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#gradient-boosting)
- [Deep Neural Network](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#deep-neural-network)
- [Graph Model](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#graph-model)
- [Natural Language Processing (model)](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#natural-language-processing-model)
- [Recommendation](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#recommendation)
- [Information Retrieval](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#information-retrieval)
    - [Full Text Search](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#full-text-search)
    - [Nearest Neighbor Search](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#nearest-neighbor-search)
- [Reinforcement Learning](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#reinforcement-learning)
- [Supervised Learning](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#supervised-learning-model)
- [Unsupervised Learning & Clustering Model](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#unsupervised-learning--clustering-model)
- [Statistical Model](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#statistical-model)
- [Evolutionary Algorithm](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#evolutionary-algorithm)
- [Reference](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#reference)
    - [Nearby Projects](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#nearby-projects)
    - [Blogs](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#blogs)
        - [Introduction](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#introduction)
        - [Tutorial](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#tutorial)
        - [Apply](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#apply)
        - [Case Study](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#case-study)
    - [Discussion](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#discussion)
    - [Books](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#books)
    - [Movie](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#movie)
    - [Paper](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#paper)
- [Thanks](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#thanks)



# Undercarriage

## Jupyter Notebook

- [google/evcxr](https://github.com/google/evcxr) - An evaluation context for Rust.
- [emakryo/rustdef](https://github.com/emakryo/rustdef) - Jupyter extension for rust.


## Plot

- [38/plotters](https://github.com/38/plotters) - A rust drawing library for high quality data plotting for both WASM and native, statically and realtimely 🦀 📈🚀
- [igiagkiozis/plotly](https://github.com/igiagkiozis/plotly) - Plotly for Rust
- [SiegeLord/RustGnuplot](https://github.com/SiegeLord/RustGnuplot) - A Rust library for drawing plots, powered by Gnuplot.


## Vector

Most things use `ndarray` or `std::vec`. Also look at `nalgebra`.

ndarray vs nalgebra - reddit https://www.reddit.com/r/rust/comments/btn1cz/ndarray_vs_nalgebra/

- [dimforge/nalgebra](https://github.com/dimforge/nalgebra) - Linear algebra library for Rust.
- [rust-ndarray/ndarray](https://github.com/rust-ndarray/ndarray) - ndarray: an N-dimensional array with array views, multidimensional slicing, and efficient operations
- [vbarrielle/sprs](https://github.com/vbarrielle/sprs) - sparse linear algebra library for rust
- [liborty/rstats](https://github.com/liborty/rstats) - Rust Statistics and Vector Algebra Library
- [PyO3/rust-numpy](https://github.com/PyO3/rust-numpy) - PyO3-based Rust binding of NumPy C-API


## Dataframe

Mainstream is `polars` using arrow.
`datafusion` looks good too.

- [ritchie46/polars](https://github.com/ritchie46/polars) - Rust DataFrame library
- [apache/arrow](https://github.com/apache/arrow/tree/master/rust) - In-memory columnar format, in Rust.
- [apache/arrow-datafusion](https://github.com/apache/arrow-datafusion) - Apache Arrow DataFusion and Ballista query engines
- [milesgranger/black-jack](https://github.com/milesgranger/black-jack) - DataFrame / Series data processing in Rust
- [nevi-me/rust-dataframe](https://github.com/nevi-me/rust-dataframe) - A Rust DataFrame implementation, built on Apache Arrow
- [kernelmachine/utah](https://github.com/kernelmachine/utah) - Dataframe structure and operations in Rust


## Image Processing

- [image-rs/image](https://github.com/image-rs/image) - Encoding and decoding images in Rust
    - [image-rs/imageproc](https://github.com/image-rs/imageproc) - Image processing operations
- [rust-cv/ndarray-image](https://github.com/rust-cv/ndarray-image) - Allows conversion between ndarray's types and image's types
- [twistedfall/opencv-rust](https://github.com/twistedfall/opencv-rust) - Rust bindings for OpenCV 3 & 4
- [rustgd/cgmath](https://github.com/rustgd/cgmath) - A linear algebra and mathematics library for computer graphics.
- [atomashpolskiy/rustface](https://github.com/atomashpolskiy/rustface) - Face detection library for the Rust programming language


## Natural Language Processing (preprocessing)

There is also the familiar `huggingface/tokenizers` in Python. 
In `rs-natural`, addition to Tokenize, Distance, NGrams, Naive-Bayes and TF-IDF can also be used.
Machine learning models may be used to process NLP.


- [huggingface/tokenizers](https://github.com/huggingface/tokenizers/tree/master/tokenizers) - The core of tokenizers, written in Rust. Provides an implementation of today's most used tokenizers, with a focus on performance and versatility.
- [guillaume-be/rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers) - Rust-tokenizer offers high-performance tokenizers for modern language models, including WordPiece, Byte-Pair Encoding (BPE) and Unigram (SentencePiece) models
- [christophertrml/rs-natural](https://github.com/christophertrml/rs-natural) - Natural Language Processing for Rust
- [bminixhofer/nnsplit](https://github.com/bminixhofer/nnsplit) - Semantic text segmentation. For sentence boundary detection, compound splitting and more.
- [greyblake/whatlang-rs](https://github.com/greyblake/whatlang-rs) - Natural language detection library for Rust.
- [finalfusion/finalfrontier](https://github.com/finalfusion/finalfrontier) - Context-sensitive word embeddings with subwords. In Rust.
- [stickeritis/sticker](https://github.com/stickeritis/sticker) - A LSTM/Transformer/dilated convolution sequence labeler
- [pemistahl/lingua-rs](https://github.com/pemistahl/lingua-rs) - 👄 The most accurate natural language detection library in the Rust ecosystem, suitable for long and short text alike
- [usamec/cntk-rs](https://github.com/usamec/cntk-rs) - Wrapper around Microsoft CNTK library
- [bminixhofer/nlprule](https://github.com/bminixhofer/nlprule) - A fast, low-resource Natural Language Processing and Error Correction library written in Rust.
- [rth/vtext](https://github.com/rth/vtext) - Simple NLP in Rust with Python bindings
- [tamuhey/tokenizations](https://github.com/tamuhey/tokenizations) - Robust and Fast tokenizations alignment library for Rust and Python
- [vgel/treebender](https://github.com/vgel/treebender) - A HDPSG-inspired symbolic natural language parser written in Rust
- [reinfer/blingfire-rs](https://github.com/reinfer/blingfire-rs) - Rust wrapper for the BlingFire tokenization library
- [CurrySoftware/rust-stemmers](https://github.com/CurrySoftware/rust-stemmers) - Common stop words in a variety of languages
- [cmccomb/rust-stop-words](https://github.com/cmccomb/rust-stop-words) - Common stop words in a variety of languages
- [Freyskeyd/nlp](https://github.com/Freyskeyd/nlp) - Rust-nlp is a library to use Natural Language Processing algorithm with RUST
- for japanese
    - [lindera-morphology/lindera](https://github.com/lindera-morphology/lindera) - A morphological analysis library.
    - [sorami/sudachi.rs](https://github.com/sorami/sudachi.rs) - An unofficial Sudachi clone in Rust (incomplete) 🦀
    - [agatan/yoin](https://github.com/agatan/yoin) - A Japanese Morphological Analyzer written in pure Rust
    - [nakagami/awabi](https://github.com/nakagami/awabi) - A morphological analyzer using mecab dictionary



## Graph

- [alibaba/GraphScope](https://github.com/alibaba/GraphScope) - GraphScope: A One-Stop Large-Scale Graph Computing System from Alibaba
- [petgraph/petgraph](https://github.com/petgraph/petgraph) - Graph data structure library for Rust.
- [rs-graph/rs-graph](https://chiselapp.com/user/fifr/repository/rs-graph/doc/release/README.md) - rs-graph is a library for graph algorithms and combinatorial optimization
- [metamolecular/gamma](https://github.com/metamolecular/gamma) - A graph library for Rust.
- [purpleprotocol/graphlib](https://github.com/purpleprotocol/graphlib) - Simple but powerful graph library for Rust


## Interface

- [mstallmo/tensorrt-rs](https://github.com/mstallmo/tensorrt-rs) - Rust library for running TensorRT accelerated deep learning models
- [ehsanmok/tvm-rust](https://github.com/ehsanmok/tvm-rust) - Rust bindings for TVM runtime
- [vertexclique/orkhon](https://github.com/vertexclique/orkhon) - Orkhon: ML Inference Framework and Server Runtime
- [xaynetwork/xaynet](https://github.com/xaynetwork/xaynet) - Xaynet represents an agnostic Federated Machine Learning framework to build privacy-preserving AI applications
- [sonos/tract](https://github.com/sonos/tract) - Tiny, no-nonsense, self-contained, Tensorflow and ONNX inference


## Workflow

- [substantic/rain](https://github.com/substantic/rain) - Framework for large distributed pipelines
- [timberio/vector](https://github.com/timberio/vector) - A high-performance, highly reliable, observability data pipeline

# Comprehensive (like sklearn)

Many libraries support the following.

- Linear Regression
- Logistic Regression
- K-Means Clustering
- Neural Networks
- Gaussian Process Regression
- Support Vector Machines
- kGaussian Mixture Models
- Naive Bayes Classifiers
- DBSCAN
- k-Nearest Neighbor Classifiers
- Principal Component Analysis
- Decision Tree
- Support Vector Machines
- Naive Bayes
- Elastic Net

It writes down the features.

- [smartcorelib/smartcore](https://github.com/smartcorelib/smartcore) - SmartCore is a comprehensive library for machine learning and numerical computing. The library provides a set of tools for linear algebra, numerical computing, optimization, and enables a generic, powerful yet still efficient approach to machine learning.
    - LASSO, Ridge, Random Forest, LU, QR, SVD, EVD, and more metrics
    - https://smartcorelib.org/user_guide/quick_start.html
- [rust-ml/linfa](https://github.com/rust-ml/linfa) - A Rust machine learning framework.
    - Gaussian Mixture Model Clustering, Agglomerative Hierarchical Clustering, ICA
    - https://github.com/rust-ml/linfa#current-state
- [maciejkula/rustlearn](https://github.com/maciejkula/rustlearn) - Machine learning crate for Rust
    - factorization machines, k-fold cross-validation, ndcg
    - https://github.com/maciejkula/rustlearn#features
- [AtheMathmo/rusty-machine](https://github.com/AtheMathmo/rusty-machine) - Machine Learning library for Rust
    - Confusion Matrix, Cross Varidation, Accuracy, F1 Score, MSE
    - https://github.com/AtheMathmo/rusty-machine#machine-learning
- [benjarison/eval-metrics](https://github.com/benjarison/eval-metrics) - Evaluation metrics for machine learning
    - Many evaluation functions
- [blue-yonder/vikos](https://github.com/blue-yonder/vikos) - A machine learning library for supervised training of parametrized models


# Comprehensive (Statistics)

Optimize, Automatic Differentiation, Numerical Analysis, Statistics, ... and more

- [Axect/Peroxide](https://github.com/Axect/Peroxide) - Rust numeric library with R, MATLAB & Python syntax
    - Linear Algebra, Functional Programming, Automatic Differentiation, Numerical Analysis, Statistics, Special functions, Plotting, Dataframe


# Gradient Boosting

catboost is for predict only.

- [mesalock-linux/gbdt-rs](https://github.com/mesalock-linux/gbdt-rs) - MesaTEE GBDT-RS : a fast and secure GBDT library, supporting TEEs such as Intel SGX and ARM TrustZone
- [davechallis/rust-xgboost](https://github.com/davechallis/rust-xgboost) - Rust bindings for XGBoost.
- [vaaaaanquish/lightgbm-rs](https://github.com/vaaaaanquish/lightgbm-rs) - LightGBM Rust binding
- [catboost/catboost](https://github.com/catboost/catboost/tree/master/catboost/rust-package) - A fast, scalable, high performance Gradient Boosting on Decision Trees library, used for ranking, classification, regression and other machine learning tasks
- [Entscheider/stamm](https://github.com/entscheider/stamm) - Generic decision trees for rust


# Deep Neural Network

Tensorflow or Pythorch are the most common.
`tch-rs` also includes torch vision.

- [tensorflow/rust](https://github.com/tensorflow/rust) - Rust language bindings for TensorFlow
- [LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs) - Rust bindings for the C++ api of PyTorch.
- [spearow/juice](https://github.com/spearow/juice) - The Hacker's Machine Learning Engine
- [neuronika/neuronika](https://github.com/neuronika/neuronika) - Tensors and dynamic neural networks in pure Rust.
- [bilal2vec/L2](https://github.com/bilal2vec/L2) - l2 is a fast, Pytorch-style Tensor+Autograd library written in Rust
- [raskr/rust-autograd](https://github.com/raskr/rust-autograd) - Tensors and differentiable operations (like TensorFlow) in Rust
- [charles-r-earp/autograph](https://github.com/charles-r-earp/autograph) - Machine Learning Library for Rust
- [patricksongzy/corgi](https://github.com/patricksongzy/corgi) - A neural network, and tensor dynamic automatic differentiation implementation for Rust.
- [JonathanWoollett-Light/cogent](https://github.com/JonathanWoollett-Light/cogent) - Simple neural network library for classification written in Rust.
- [oliverfunk/darknet-rs](https://github.com/oliverfunk/darknet-rs) - Rust bindings for darknet
- [jakelee8/mxnet-rs](https://github.com/jakelee8/mxnet-rs) - mxnet for Rust
- [jramapuram/hal](https://github.com/jramapuram/hal) - Rust based Cross-GPU Machine Learning
- [primitiv/primitiv-rust](https://github.com/primitiv/primitiv-rust) - Rust binding of primitiv
- [chantera/dynet-rs](https://github.com/chantera/dynet-rs) - The Rust Language Bindings for DyNet
- [millardjn/alumina](https://github.com/millardjn/alumina) - A deep learning library for rust
- [jramapuram/hal](https://github.com/jramapuram/hal) - Rust based Cross-GPU Machine Learning
- [afck/fann-rs](https://github.com/afck/fann-rs) - Rust wrapper for the Fast Artificial Neural Network library
- [autumnai/leaf](https://github.com/autumnai/leaf) - Open Machine Intelligence Framework for Hackers. (GPU/CPU)
- [c0dearm/mushin](https://github.com/c0dearm/mushin) - Compile-time creation of neural networks
- [tedsta/deeplearn-rs](https://github.com/tedsta/deeplearn-rs) - Neural networks in Rust


# Graph Model

- [Synerise/cleora](https://github.com/Synerise/cleora) - Cleora AI is a general-purpose model for efficient, scalable learning of stable and inductive entity embeddings for heterogeneous relational data.


# Natural Language Processing (model)

- [guillaume-be/rust-bert](https://github.com/guillaume-be/rust-bert) - Rust native ready-to-use NLP pipelines and transformer-based models (BERT, DistilBERT, GPT2,...)
- [proycon/deepfrog](https://github.com/proycon/deepfrog) - An NLP-suite powered by deep learning
- [ferristseng/rust-tfidf](https://github.com/ferristseng/rust-tfidf) - Library to calculate TF-IDF
- [messense/fasttext-rs](https://github.com/messense/fasttext-rs) - fastText Rust binding
- [mklf/word2vec-rs](https://github.com/mklf/word2vec-rs) - pure rust implementation of word2vec
- [DimaKudosh/word2vec](https://github.com/DimaKudosh/word2vec) - Rust interface to word2vec.
- [lloydmeta/sloword2vec-rs](https://github.com/lloydmeta/sloword2vec-rs) - A naive (read: slow) implementation of Word2Vec. Uses BLAS behind the scenes for speed.


# Recommendation

For Matrix Factorization, you can use rustlearn.

- [jackgerrits/vowpalwabbit-rs](https://github.com/jackgerrits/vowpalwabbit-rs) - 🦀🐇 Rusty VowpalWabbit
- [outbrain/fwumious_wabbit](https://github.com/outbrain/fwumious_wabbit) - Fwumious Wabbit, fast on-line machine learning toolkit written in Rust
- [hja22/rucommender](https://github.com/hja22/rucommender) - Rust implementation of user-based collaborative filtering
- [maciejkula/sbr-rs](https://github.com/maciejkula/sbr-rs) - Deep recommender systems for Rust
- [chrisvittal/quackin](https://github.com/chrisvittal/quackin) - A recommender systems framework for Rust
- [snd/onmf](https://github.com/snd/onmf) - fast rust implementation of online nonnegative matrix factorization as laid out in the paper "detect and track latent factors with online nonnegative matrix factorization"
- [rhysnewell/nymph](https://github.com/rhysnewell/nymph) - Non-Negative Matrix Factorization in Rust


# Information Retrieval

## Full Text Search

- [bayard-search/bayard](https://github.com/bayard-search/bayard) - A full-text search and indexing server written in Rust.
- [neuml/txtai.rs](https://github.com/neuml/txtai.rs) - AI-powered search engine for Rust
- [meilisearch/MeiliSearch](https://github.com/meilisearch/MeiliSearch) - Lightning Fast, Ultra Relevant, and Typo-Tolerant Search Engine
- [toshi-search/Toshi](https://github.com/toshi-search/Toshi) - A full-text search engine in rust
- [BurntSushi/fst](https://github.com/BurntSushi/fst) - Represent large sets and maps compactly with finite state transducers.
- [tantivy-search/tantivy](https://github.com/tantivy-search/tantivy) - Tantivy is a full-text search engine library inspired by Apache Lucene and written in Rust
- [tinysearch/tinysearch](https://github.com/tinysearch/tinysearch) - 🔍 Tiny, full-text search engine for static websites built with Rust and Wasm
- [https://github.com/andylokandy/simsearch-rs](https://github.com/andylokandy/simsearch-rs) - A simple and lightweight fuzzy search engine that works in memory, searching for similar strings
- [jameslittle230/stork](https://github.com/jameslittle230/stork) - 🔎 Impossibly fast web search, made for static sites.
- [elastic/elasticsearch-rs](https://github.com/elastic/elasticsearch-rs) - Official Elasticsearch Rust Client


## Nearest Neighbor Search

ANN search (approximate nearest neighbor), hashing, dimensional tree, etc...

- [Enet4/faiss-rs](https://github.com/Enet4/faiss-rs) - Rust language bindings for Faiss
- [rust-cv/hnsw](https://github.com/rust-cv/hnsw) - HNSW ANN from the paper "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
- [InstantDomain/instant-distance](https://github.com/InstantDomain/instant-distance) - Fast approximate nearest neighbor searching in Rust, based on HNSW index
- [lerouxrgd/ngt-rs](https://github.com/lerouxrgd/ngt-rs) - Rust wrappers for NGT approximate nearest neighbor search
- [granne/granne](https://github.com/granne/granne) - Graph-based Approximate Nearest Neighbor Search
- [qdrant/qdrant](https://github.com/qdrant/qdrant) - Qdrant - vector similarity search engine with extended filtering support
- [rust-cv/hwt](https://github.com/rust-cv/hwt) - Hamming Weight Tree from the paper "Online Nearest Neighbor Search in Hamming Space"
- [kornelski/vpsearch](https://github.com/kornelski/vpsearch) - C library for finding nearest (most similar) element in a set
- [mrhooray/kdtree-rs](https://github.com/mrhooray/kdtree-rs) - K-dimensional tree in Rust for fast geospatial indexing and lookup
- [petabi/petal-neighbors](https://github.com/petabi/petal-neighbors) - Nearest neighbor search algorithms including a ball tree and a vantage point tree.
- [ritchie46/lsh-rs](https://github.com/ritchie46/lsh-rs) - Locality Sensitive Hashing in Rust with Python bindings
- [kampersanda/mih-rs](https://github.com/kampersanda/mih-rs) - Rust implementation of multi-index hashing for neighbor searches on 64-bit codes in the Hamming space


# Reinforcement Learning

Looks good `border`.

- [taku-y/border](https://github.com/taku-y/border) - Border is a reinforcement learning library in Rust.
- [NivenT/REnforce](https://github.com/NivenT/REnforce) - Reinforcement learning library written in Rust
- [tspooner/rsrl](https://github.com/tspooner/rsrl) - A fast, safe and easy to use reinforcement learning framework in Rust.
- [milanboers/rurel](https://github.com/milanboers/rurel) - Flexible, reusable reinforcement learning (Q learning) implementation in Rust
- [MrRobb/gym-rs](https://github.com/mrrobb/gym-rs) - OpenAI Gym bindings for Rust


# Supervised Learning Model

- [shadeMe/liblinear-rs](https://github.com/shademe/liblinear-rs) - Rust language bindings for the LIBLINEAR C/C++ library.
- [messense/crfsuite-rs](https://github.com/messense/crfsuite-rs) - Rust binding to crfsuite
- [ralfbiedert/ffsvm-rust](https://github.com/ralfbiedert/ffsvm-rust) - FFSVM stands for "Really Fast Support Vector Machine"
- [zenoxygen/bayespam](https://github.com/zenoxygen/bayespam) - A simple bayesian spam classifier written in Rust.
- [Rui_Vieira/naive-bayesnaive-bayes](https://gitlab.com/ruivieira/naive-bayes) - A Naive Bayes classifier written in Rust.
- [sile/randomforest](https://github.com/sile/randomforest) - A random forest implementation in Rust
- [tomtung/craftml-rs](https://github.com/tomtung/craftml-rs) - A Rust🦀 implementation of CRAFTML, an Efficient Clustering-based Random Forest for Extreme Multi-label Learning


# Unsupervised Learning & Clustering Model

- [frjnn/bhtsne](https://github.com/frjnn/bhtsne) - Barnes-Hut t-SNE implementation written in Rust.
- [avinashshenoy97/RusticSOM](https://github.com/avinashshenoy97/RusticSOM) - Rust library for Self Organising Maps (SOM).
- [diffeo/kodama](https://github.com/diffeo/kodama) - Fast hierarchical agglomerative clustering in Rust.
- [kno10/rust-kmedoids](https://github.com/kno10/rust-kmedoids) - k-Medoids clustering in Rust with the FasterPAM algorithm
- [petabi/petal-clustering](https://github.com/petabi/petal-clustering) - DBSCAN and OPTICS clustering algorithms.
- [milesgranger/gap_statistic](https://github.com/milesgranger/gap_statistic) - Dynamically get the suggested clusters in the data for unsupervised learning.
- [genbattle/rkm](https://github.com/genbattle/rkm) - Generic k-means implementation written in Rust


# Statistical Model

- [Redpoll/changepoint](https://gitlab.com/Redpoll/changepoint) - Includes the following change point detection algorithms: Bocpd -- Online Bayesian Change Point Detection Reference. BocpdTruncated -- Same as Bocpd but truncated the run-length distribution when those lengths are unlikely.
- [krfricke/arima](https://github.com/krfricke/arima) - ARIMA modelling for Rust
- [Daingun/automatica](https://gitlab.com/daingun/automatica) - Automatic Control Systems Library
- [rbagd/rust-linearkalman](https://github.com/rbagd/rust-linearkalman) - Kalman filtering and smoothing in Rust
- [sanity/pair_adjacent_violators](https://github.com/sanity/pair_adjacent_violators) - An implementation of the Pair Adjacent Violators algorithm for isotonic regression in Rust


# Evolutionary Algorithm

- [martinus/differential-evolution-rs](https://github.com/martinus/differential-evolution-rs) - Generic Differential Evolution for Rust
- [innoave/genevo](https://github.com/innoave/genevo) - Execute genetic algorithm (GA) simulations in a customizable and extensible way.
- [Jeffail/spiril](https://github.com/Jeffail/spiril) - Rust library for genetic algorithms
- [sotrh/rust-genetic-algorithm](https://github.com/sotrh/rust-genetic-algorithm) - Example of a genetic algorithm in Rust and Python
- [willi-kappler/darwin-rs](https://github.com/willi-kappler/darwin-rs) - darwin-rs, evolutionary algorithms with rust


# Reference

## Nearby Projects

- [Are we learning yet?](http://www.arewelearningyet.com/), A work-in-progress to catalog the state of machine learning in Rust
- [e-tony/best-of-ml-rust](https://github.com/e-tony/best-of-ml-rust), A ranked list of awesome machine learning Rust libraries
- [The Best 51 Rust Machine learning Libraries](https://rustrepo.com/catalog/rust-machine-learning_newest_1), RustRepo
- [rust-unofficial/awesome-rust](https://github.com/rust-unofficial/awesome-rust), A curated list of Rust code and resources
- [Top 16 Rust Machine learning Projects](https://www.libhunt.com/l/rust/t/machine-learning), Open-source Rust projects categorized as Machine learning
- [39+ Best Rust Machine learning frameworks, libraries, software and resourcese](https://reposhub.com/rust/machine-learning), ReposHub


## Blogs

### Introduction

- [About Rust’s Machine Learning Community](https://medium.com/@autumn_eng/about-rust-s-machine-learning-community-4cda5ec8a790#.hvkp56j3f), Medium, 2016/1/6, Autumn Engineering
- [Rust vs Python: Technology And Business Comparison](https://www.ideamotive.co/blog/rust-vs-python-technology-and-business-comparison), 2021/3/4, Miłosz Kaczorowski
- [I wrote one of the fastest DataFrame libraries](https://www.ritchievink.com/blog/2021/02/28/i-wrote-one-of-the-fastest-dataframe-libraries), 2021/2/28, Ritchie Vink 
- [Polars: The fastest DataFrame library you've never heard of](https://www.analyticsvidhya.com/blog/2021/06/polars-the-fastest-dataframe-library-youve-never-heard-of) 2021/1/19, Analytics Vidhya 
- [Data Manipulation: Polars vs Rust](https://able.bio/haixuanTao/data-manipulation-polars-vs-rust--3def44c8), 2021/3/13, Xavier Tao
- [State of Machine Learning in Rust – Ehsan's Blog](https://ehsanmkermani.com/2019/05/13/state-of-machine-learning-in-rust/), 2019/5/13, Published by Ehsan
- [Ritchie Vink, Machine Learning Engineer, writes Polars, one of the fastest DataFrame libraries in Python and Rust](https://www.xomnia.com/post/ritchie-vink-writes-polars-one-of-the-fastest-dataframe-libraries-in-python-and-rust/), Xomnia, 2021/5/11
- Japanese
    - [WebAssemblyでの機械学習モデルデプロイの動向](https://tkat0.github.io/posts/deploy-ml-as-wasm), 2020/12/2, tkat0
    - [Rustで扱える機械学習関連のクレート2021](https://vaaaaaanquish.hatenablog.com/entry/2021/01/23/233113), 2021/1/23, vaaaaaanquish


### Tutorial

- [Rust Machine Learning Book](https://rust-ml.github.io/book/chapter_1.html), Examples of KMeans and DBSCAN with linfa-clustering
- [Artificial Intelligence and Machine Learning – Practical Rust Projects: Building Game, Physical Computing, and Machine Learning Applications – Dev Guis ](http://devguis.com/6-artificial-intelligence-and-machine-learning-practical-rust-projects-building-game-physical-computing-and-machine-learning-applications.html), 2021/5/19
- [Machine learning in Rust using Linfa](https://blog.logrocket.com/machine-learning-in-rust-using-linfa/), LogRocket Blog, 2021/4/30, Timeular, Mario Zupan, Examples of LogisticRegression
- [Machine Learning in Rust, Smartcore](https://medium.com/swlh/machine-learning-in-rust-smartcore-2f472d1ce83), Medium, The Startup, 2021/1/15, [Vlad Orlov](https://volodymyr-orlov.medium.com/), Examples of LinerRegression, Random Forest Regressor, and K-Fold
- [Machine Learning in Rust, Logistic Regression](https://medium.com/swlh/machine-learning-in-rust-logistic-regression-74d6743df161), Medium, The Startup, 2021/1/6, [Vlad Orlov](https://volodymyr-orlov.medium.com/)
- [Machine Learning in Rust, Linear Regression](https://medium.com/swlh/machine-learning-in-rust-linear-regression-edef3fb65f93), Medium, The Startup, 2020/12/16, [Vlad Orlov](https://volodymyr-orlov.medium.com/)
- [Machine Learning in Rust](https://athemathmo.github.io/2016/03/07/rusty-machine.html), 2016/3/7, James, Examples of LogisticRegressor
- [Machine Learning and Rust (Part 1): Getting Started!](https://levelup.gitconnected.com/machine-learning-and-rust-part-1-getting-started-745885771bc2), Level Up Coding, 2021/1/9, Stefano Bosisio 
- [Machine Learning and Rust (Part 2): Linear Regression](https://levelup.gitconnected.com/machine-learning-and-rust-part-2-linear-regression-d3b820ed28f9), Level Up Coding, 2021/6/15, Stefano Bosisio 
- [Machine Learning and Rust (Part 3): Smartcore, Dataframe, and Linear Regression](https://levelup.gitconnected.com/machine-learning-and-rust-part-3-smartcore-dataframe-and-linear-regression-10451fdc2e60), Level Up Coding, 2021/7/1, Stefano Bosisio 
- [Tensorflow Rust Practical Part 1](https://www.programmersought.com/article/18696273900/), Programmer Sought, 2018
- [A Machine Learning introduction to ndarray](https://barcelona.rustfest.eu/sessions/machine-learning-ndarray), RustFest 2019, 2019/11/12, [Luca Palmieri](https://github.com/LukeMathWalker)


### Apply

- [Deep Learning in Rust: baby steps](https://medium.com/@tedsta/deep-learning-in-rust-7e228107cccc), Medium,  2016/2/2, Theodore DeRego
- [A Rust SentencePiece implementation](https://guillaume-be.github.io/2020-05-30/sentence_piece), Rust NLP tales, 2020/5/30
- [Accelerating text generation with Rust](https://guillaume-be.github.io/2020-11-21/generation_benchmarks), Rust NLP tales, 2020/11/21
- [A Simple Text Summarizer written in Rust](https://towardsdatascience.com/a-simple-text-summarizer-written-in-rust-4df05f9327a5), Towards Data Science, 2020/11/24, [Charles Chan](https://chancharles.medium.com/), Examples of Text Sentence Vector, Cosine Distance and PageRank
- [Extracting deep learning image embeddings in Rust](https://logicai.io/blog/extracting-image-embeddings/), RecoAI, 2021/6/1, Paweł Jankiewic, Examples of ONNX


### Case study

- [Production users - Rust Programming Language](https://www.rust-lang.org/production/users), by rust-lang.org
- [Taking ML to production with Rust: a 25x speedup](https://www.lpalmieri.com/posts/2019-12-01-taking-ml-to-production-with-rust-a-25x-speedup/), A LEARNING JOURNAL, 2019/12/1, [@algo_luca](https://twitter.com/algo_luca)
- [9 Companies That Use Rust in Production](https://serokell.io/blog/rust-companies), serokell, 2020/11/18, Gints Dreimanis
- Japanese
    - [エッジMLシステムをC/C++からRustへ移行した事例](https://docs.google.com/presentation/d/1HOL9jheJnKkh2q7w3hU_px-je1qL7lxrSXV-0P1hces/edit?usp=sharing), Rust.Tokyo 2019, 2019/10/26, DeNA, tkat0
    - [Rustで作る機械学習を用いた画像クロッピングシステム](https://ml-loft.connpass.com/event/157785/), ML@Loft #9, 2019/12/19, Cookpad, johshisha
    - [fnwiya/japanese-rust-companies: 日本で Rust を利用している会社一覧](https://github.com/fnwiya/japanese-rust-companies)

## Discussion

- [Natural Language Processing in Rust : rust](https://www.reddit.com/r/rust/comments/5jj8vr/natural_language_processing_in_rust), 2016/12/6
- [Future prospect of Machine Learning in Rust Programming Language : MachineLearning](https://www.reddit.com/r/MachineLearning/comments/7iz51p/d_future_prospect_of_machine_learning_in_rust/), 2017/11/11
- [Interest for NLP in Rust? - The Rust Programming Language Forum](https://users.rust-lang.org/t/interest-for-nlp-in-rust/15331), 2018/1/19
- [Is Rust good for deep learning and artificial intelligence? - The Rust Programming Language Forum](https://users.rust-lang.org/t/is-rust-good-for-deep-learning-and-artificial-intelligence/22866), 2018/11/18
- [ndarray vs nalgebra : rust](https://www.reddit.com/r/rust/comments/btn1cz/ndarray_vs_nalgebra/), 2019/5/28
- [Taking ML to production with Rust | Hacker News](https://news.ycombinator.com/item?id=21680965), 2019/12/2
- [Who is using Rust for Machine learning in production/research? : rust](https://www.reddit.com/r/rust/comments/fvehyq/d_who_is_using_rust_for_machine_learning_in/), 2020/4/5
- [SmartCore, fast and comprehensive machine learning library for Rust! : rust](https://www.reddit.com/r/rust/comments/j1mj1g/smartcore_fast_and_comprehensive_machine_learning/), 2020/9/29


## Books

- [Practical Machine Learning with Rust: Creating Intelligent Applications in Rust (English Edition)](https://amzn.to/3h7JV8U), 2019/12/10, Joydeep Bhattacharjee
    - Write machine learning algorithms in Rust
    - Use Rust libraries for different tasks in machine learning
    - Create concise Rust packages for your machine learning applications
    - Implement NLP and computer vision in Rust
    - Deploy your code in the cloud and on bare metal servers
    - source code: [Apress/practical-machine-learning-w-rust](https://github.com/Apress/practical-machine-learning-w-rust)


## Movie

- [The /r/playrust Classifier: Real World Rust Data Science](https://www.youtube.com/watch?v=lY10kTcM8ek), RustConf 2016, 2016/10/05, Suchin Gururangan & Colin O'Brien
- [Machine Learning is changing - is Rust the right tool for the job?](https://www.youtube.com/watch?v=odI_LY8AIqo), RustLab 2019, 2019/10/31, Luca Palmieri
- [Using TensorFlow in Embedded Rust](https://www.youtube.com/watch?v=DUVE86yTfKU), 2020/09/29, Ferrous Systems GmbH, Richard Meadows
- [Building AI Units in Rust](https://www.youtube.com/watch?v=UHFlKAmANJg), FOSSASIA 2018, 2018/3/25, Vigneshwer Dhinakaran 
- Japanese
    - [Full use of Rust on edge and cloud: AI and IoT use cases エッジとクラウドでRustを使いこなす ～AI/IoTでの事例～](https://rustfest.global/session/10-%E3%82%A8%E3%83%83%E3%82%B8%E3%81%A8%E3%82%AF%E3%83%A9%E3%82%A6%E3%83%89%E3%81%A7rust%E3%82%92%E4%BD%BF%E3%81%84%E3%81%93%E3%81%AA%E3%81%99-%EF%BD%9Eai-iot%E3%81%A7%E3%81%AE%E4%BA%8B%E4%BE%8B%EF%BD%9E/), RUSTFEST, 2020/11/7, Mobility Technologies, tkat0


## Paper

- [End-to-end NLP Pipelines in Rust](https://www.aclweb.org/anthology/2020.nlposs-1.4.pdf), Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS), pages 20–25 Virtual Conference, 2020/11/19, Guillaume Becquin


# Thanks

Thanks for all the projects.

Please PR. Thanks.

