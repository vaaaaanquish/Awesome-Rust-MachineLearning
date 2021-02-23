# Awasome-Rust-MachineLearning

Awasome Rust Machine Learning crate list that writing this with the idea of migrating from Python.

It's a list with the major items at the top.




# Undercarriage

## Jupter Notebook

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


## Dataframe

Mainstream is `polars` using arrow.

- [ritchie46/polars](https://github.com/ritchie46/polars) - Rust DataFrame library
- [apache/arrow](https://github.com/apache/arrow/tree/master/rust) - In-memory columnar format, in Rust.
- [milesgranger/black-jack](https://github.com/milesgranger/black-jack) - DataFrame / Series data processing in Rust
- [nevi-me/rust-dataframe](https://github.com/nevi-me/rust-dataframe) - A Rust DataFrame implementation, built on Apache Arrow
- [kernelmachine/utah](https://github.com/kernelmachine/utah) - Dataframe structure and operations in Rust


## Image Processing

- [image-rs/image](https://github.com/image-rs/image) - Encoding and decoding images in Rust
    - [image-rs/imageproc](https://github.com/image-rs/imageproc) - Image processing operations
- [rust-cv/ndarray-image](https://github.com/rust-cv/ndarray-image) - Allows conversion between ndarray's types and image's types
- [twistedfall/opencv-rust](https://github.com/twistedfall/opencv-rust) - Rust bindings for OpenCV 3 & 4
- [rustgd/cgmath](https://github.com/rustgd/cgmath) - A linear algebra and mathematics library for computer graphics.


## Natural Language Processing (preprocessing)

There is also the familiar `huggingface/tokenizers` in Python. 
In `rs-natural`, addition to Tokenize, Distance, NGrams, Naive-Bayes and TF-IDF can also be used.

- [huggingface/tokenizers](https://github.com/huggingface/tokenizers/tree/master/tokenizers) - The core of tokenizers, written in Rust. Provides an implementation of today's most used tokenizers, with a focus on performance and versatility.
- [guillaume-be/rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers) - Rust-tokenizer offers high-performance tokenizers for modern language models, including WordPiece, Byte-Pair Encoding (BPE) and Unigram (SentencePiece) models
- [christophertrml/rs-natural](https://github.com/christophertrml/rs-natural) - Natural Language Processing for Rust
- [greyblake/whatlang-rs](https://github.com/greyblake/whatlang-rs) - Natural language detection library for Rust.
- [finalfusion/finalfrontier](https://github.com/finalfusion/finalfrontier) - Context-sensitive word embeddings with subwords. In Rust.
- [stickeritis/sticker](https://github.com/stickeritis/sticker) - A LSTM/Transformer/dilated convolution sequence labeler
- [pemistahl/lingua-rs](https://github.com/pemistahl/lingua-rs) - 👄 The most accurate natural language detection library in the Rust ecosystem, suitable for long and short text alike
- [usamec/cntk-rs](https://github.com/usamec/cntk-rs) - Wrapper around Microsoft CNTK library
- [bminixhofer/nlprule](https://github.com/bminixhofer/nlprule) - A fast, low-resource Natural Language Processing and Error Correction library written in Rust.
- [rth/vtext](https://github.com/rth/vtext) - Simple NLP in Rust with Python bindings
- [tamuhey/tokenizations](https://github.com/tamuhey/tokenizations) - Robust and Fast tokenizations alignment library for Rust and Python
- [reinfer/blingfire-rs](https://github.com/reinfer/blingfire-rs) - Rust wrapper for the BlingFire tokenization library
- [CurrySoftware/rust-stemmers](https://github.com/CurrySoftware/rust-stemmers) - Common stop words in a variety of languages
- [cmccomb/rust-stop-words](https://github.com/cmccomb/rust-stop-words) - Common stop words in a variety of languages
- [Freyskeyd/nlp](https://github.com/Freyskeyd/nlp) - Rust-nlp is a library to use Natural Language Processing algorithm with RUST
- for japanese
    - [lindera-morphology/lindera](https://github.com/lindera-morphology/lindera) - A morphological analysis library.
    - [sorami/sudachi.rs](https://github.com/sorami/sudachi.rs) - An unofficial Sudachi clone in Rust (incomplete) 🦀
    - [agatan/yoin](https://github.com/agatan/yoin) - A Japanese Morphological Analyzer written in pure Rust
    - [nakagami/awabi](https://github.com/nakagami/awabi) - A morphological analyzer using mecab dictionary


## ref

- https://users.rust-lang.org/t/interest-for-nlp-in-rust/15331
- https://www.reddit.com/r/rust/comments/5jj8vr/natural_language_processing_in_rust/


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

It write down the features.

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


# Gradient Boosting

catboost is for predict only.

- [mesalock-linux/gbdt-rs](https://github.com/mesalock-linux/gbdt-rs) - MesaTEE GBDT-RS : a fast and secure GBDT library, supporting TEEs such as Intel SGX and ARM TrustZone
- [davechallis/rust-xgboost](https://github.com/davechallis/rust-xgboost) - Rust bindings for XGBoost.
- [vaaaaanquish/lightgbm-rs](https://github.com/vaaaaanquish/lightgbm-rs) - LightGBM Rust binding
- [catboost/catboost](https://github.com/catboost/catboost/tree/master/catboost/rust-package) - A fast, scalable, high performance Gradient Boosting on Decision Trees library, used for ranking, classification, regression and other machine learning tasks


# Deep Neaural Network

Tensorflow or Pythorch are the most common.
`tch-rs` also includes torch vision.

- [tensorflow/rust](https://github.com/tensorflow/rust) - Rust language bindings for TensorFlow
- [LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs) - Rust bindings for the C++ api of PyTorch.
- [raskr/rust-autograd](https://github.com/raskr/rust-autograd) - Tensors and differentiable operations (like TensorFlow) in Rust
- [oliverfunk/darknet-rs](https://github.com/oliverfunk/darknet-rs) - Rust bindings for darknet
- [primitiv/primitiv-rust](https://github.com/primitiv/primitiv-rust) - Rust binding of primitiv
- [chantera/dynet-rs](https://github.com/chantera/dynet-rs) - The Rust Language Bindings for DyNet


# Natural Language Processing (model)

- [guillaume-be/rust-bert](https://github.com/guillaume-be/rust-bert) - Rust native ready-to-use NLP pipelines and transformer-based models (BERT, DistilBERT, GPT2,...)
- [ferristseng/rust-tfidf](https://github.com/ferristseng/rust-tfidf) - Library to calculate TF-IDF
- [messense/fasttext-rs](https://github.com/messense/fasttext-rs) - fastText Rust binding
- [mklf/word2vec-rs](https://github.com/mklf/word2vec-rs) - pure rust implemention of word2vec
- [DimaKudosh/word2vec](https://github.com/DimaKudosh/word2vec) - Rust interface to word2vec.


# Recommendation

For Matrix Factorization, you can use rustlearn.

- [jackgerrits/vowpalwabbit-rs](https://github.com/jackgerrits/vowpalwabbit-rs) - 🦀🐇 Rusty VowpalWabbit
- [hja22/rucommender](https://github.com/hja22/rucommender) - Rust implementation of user-based collaborative filtering
- [chrisvittal/quackin](https://github.com/chrisvittal/quackin) - A recommender systems framework for Rust
- [snd/onmf](https://github.com/snd/onmf) - fast rust implementation of online nonnegative matrix factorization as laid out in the paper "detect and track latent factors with online nonnegative matrix factorization"


# Information Retrieval

## Full Text Search

- [bayard-search/bayard](https://github.com/bayard-search/bayard) - A full-text search and indexing server written in Rust.
- [tinysearch/tinysearch](https://github.com/tinysearch/tinysearch) - 🔍 Tiny, full-text search engine for static websites built with Rust and Wasm
- [jameslittle230/stork](https://github.com/jameslittle230/stork) - 🔎 Impossibly fast web search, made for static sites.
- [elastic/elasticsearch-rs](https://github.com/elastic/elasticsearch-rs) - Official Elasticsearch Rust Client


## Nearest Neighbor Search

- [Enet4/faiss-rs](https://github.com/Enet4/faiss-rs) - Rust language bindings for Faiss
- [granne/granne](https://github.com/granne/granne) - Graph-based Approximate Nearest Neighbor Search
- [kornelski/vpsearch](https://github.com/kornelski/vpsearch) - C library for finding nearest (most similar) element in a set
- [mrhooray/kdtree-rs](https://github.com/mrhooray/kdtree-rs) - K-dimensional tree in Rust for fast geospatial indexing and lookup



# Reinforcement Learning

- [tspooner/rsrl](https://github.com/tspooner/rsrl) - A fast, safe and easy to use reinforcement learning framework in Rust.
- [milanboers/rurel](https://github.com/milanboers/rurel) - Flexible, reusable reinforcement learning (Q learning) implementation in Rust
- [MrRobb/gym-rs](https://github.com/mrrobb/gym-rs) - OpenAI Gym bindings for Rust


# Unsupervised learning

- [avinashshenoy97/RusticSOM](https://github.com/avinashshenoy97/RusticSOM) - Rust library for Self Organising Maps (SOM).



# Thanks

Thanks for all the projects.

This document is based on a blog in Japanese [Rustで扱える機械学習関連のクレート2021](https://vaaaaaanquish.hatenablog.com/entry/2021/01/23/233113?_ga=2.159361534.596250292.1613160100-1206398386.1613160100#--Natural-Language-Processing--).

Please PR. Thanks.



# Reference & Nearby Projects
- https://lib.rs/
- https://crates.io/keywords/
- https://github.com/rust-unofficial/awesome-rust
- https://medium.com/@autumn_eng/about-rust-s-machine-learning-community-4cda5ec8a790#.hvkp56j3f
- http://www.arewelearningyet.com/
