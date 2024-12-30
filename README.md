![arml](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning/blob/main/public/img/arml.png?raw=true)

This repository is a list of machine learning libraries written in Rust.
It's a compilation of GitHub repositories, blogs, books, movies, discussions, papers.
This repository is targeted at people who are thinking of migrating from Python. 🦀🐍

It is divided into several basic library and algorithm categories.
And it also contains libraries that are no longer maintained and small libraries.
It has commented on the helpful parts of the code.
It also commented on good libraries within each category.

We can find a better way to use Rust for Machine Learning.


- [Website (en)](https://vaaaaanquish.github.io/Awesome-Rust-MachineLearning)
- [GitHub (en)](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning/blob/main/README.md)
- [GitHub (ja)](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning/blob/main/README.ja.md)


# ToC

- [Support Tools](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#support-tools)
    - [Jupyter Notebook](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#jupyter-notebook)
    - [Graph Plot](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#graph-plot)
    - [Vector](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#vector)
    - [Dataframe](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#dataframe)
    - [Image Processing](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#image-processing)
    - [Natural Language Processing (preprocessing)](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#natural-language-processing-preprocessing)
    - [Graphical Modeling](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#graphical-modeling)
    - [Interface & Pipeline & AutoML](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#interface--pipeline--automl)
    - [Workflow](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#workflow)
    - [GPU](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#gpu)
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
    - [PodCast](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#podcast)
    - [Paper](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#paper)
- [Thanks](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning#thanks)



# Support Tools


## Jupyter Notebook

`evcxr` can be handled as Jupyter Kernel or REPL. It is helpful for learning and validation.

- [google/evcxr](https://github.com/google/evcxr) - An evaluation context for Rust.
- [emakryo/rustdef](https://github.com/emakryo/rustdef) - Jupyter extension for rust.
- [murarth/rusti](https://github.com/murarth/rusti) - REPL for the Rust programming language


## Graph Plot

It might want to try `plotters` for now.

- [38/plotters](https://github.com/38/plotters) - A rust drawing library for high quality data plotting for both WASM and native, statically and realtimely 🦀 📈🚀
- [igiagkiozis/plotly](https://github.com/igiagkiozis/plotly) - Plotly for Rust
- [milliams/plotlib](https://github.com/milliams/plotlib) - Data plotting library for Rust
- [tiby312/poloto](https://github.com/tiby312/poloto) - A simple 2D plotting library that outputs graphs to SVG that can be styled using CSS.
- [askanium/rustplotlib](https://github.com/askanium/rustplotlib) - A pure Rust visualization library inspired by D3.js
- [SiegeLord/RustGnuplot](https://github.com/SiegeLord/RustGnuplot) - A Rust library for drawing plots, powered by Gnuplot.
- [saona-raimundo/preexplorer](https://github.com/saona-raimundo/preexplorer) - Externalize easily the plotting process from Rust to gnuplot.
- [procyon-rs/vega_lite_4.rs](https://github.com/procyon-rs/vega_lite_4.rs) - rust api for vega-lite v4
    - [procyon-rs/showata](https://github.com/procyon-rs/showata) - A library of to show data (in browser, evcxr_jupyter) as table, chart...
- [coder543/dataplotlib](https://github.com/coder543/dataplotlib) - Scientific plotting library for Rust
- [shahinrostami/chord_rs](https://github.com/shahinrostami/chord_rs) - Rust crate for creating beautiful interactive Chord Diagrams. Pro version available at https://m8.fyi/chord


ASCII line graph:

- [loony-bean/textplots-rs](https://github.com/loony-bean/textplots-rs) Terminal plotting library for Rust
- [orhanbalci/rasciigraph](https://github.com/orhanbalci/rasciigraph) Zero dependency Rust crate to make lightweight ASCII line graph ╭┈╯ in command line apps with no other dependencies.
- [jakobhellermann/piechart](https://github.com/jakobhellermann/piechart) a rust crate for drawing fancy pie charts in the terminal
- [milliams/plot](https://github.com/milliams/plot) Command-line plotting tool written in Rust


Examples:

- Plotters Developer's Guide - Plotter Developer's Guide [https://plotters-rs.github.io/book/intro/introduction.html](https://plotters-rs.github.io/book/intro/introduction.html)
- Plotly.rs - Plotly.rs Book [https://igiagkiozis.github.io/plotly/content/plotly_rs.html](https://igiagkiozis.github.io/plotly/content/plotly_rs.html)
- petgraph_review [https://timothy.hobbs.cz/rust-play/petgraph_review.html](https://timothy.hobbs.cz/rust-play/petgraph_review.html)
- evcxr-jupyter-integration [https://plotters-rs.github.io/plotters-doc-data/evcxr-jupyter-integration.html](https://plotters-rs.github.io/plotters-doc-data/evcxr-jupyter-integration.html)
- Rust for Data Science: Tutorial 1 - DEV Community [https://dev.to/davidedelpapa/rust-for-data-science-tutorial-1-4g5j](https://dev.to/davidedelpapa/rust-for-data-science-tutorial-1-4g5j)
- Preface | Data Crayon [https://datacrayon.com/posts/programming/rust-notebooks/preface/](https://datacrayon.com/posts/programming/rust-notebooks/preface/)
- Drawing SVG Graphs with Rust [https://cetra3.github.io/blog/drawing-svg-graphs-rust/](Drawing SVG Graphs with Rust https://cetra3.github.io/blog/drawing-svg-graphs-rust/)


## Vector

Most things use `ndarray` or `std::vec`. 

Also, look at `nalgebra`. When the size of the matrix is known, it is valid.
See also: [ndarray vs nalgebra - reddit](https://www.reddit.com/r/rust/comments/btn1cz/ndarray_vs_nalgebra/)

- [dimforge/nalgebra](https://github.com/dimforge/nalgebra) - Linear algebra library for Rust.
- [rust-ndarray/ndarray](https://github.com/rust-ndarray/ndarray) - ndarray: an N-dimensional array with array views, multidimensional slicing, and efficient operations
- [AtheMathmo/rulinalg](https://github.com/AtheMathmo/rulinalg) - A linear algebra library written in Rust
- [sarah-quinones/faer-rs](https://github.com/sarah-quinones/faer-rs) - a SIMD accelerated linear algebra library written in Rust
- [arrayfire/arrayfire-rust](https://github.com/arrayfire/arrayfire-rust) - Rust wrapper for ArrayFire
- [bluss/arrayvec](https://github.com/bluss/arrayvec) - A vector with a fixed capacity. (Rust)
- [vbarrielle/sprs](https://github.com/vbarrielle/sprs) - sparse linear algebra library for rust
- [liborty/rstats](https://github.com/liborty/rstats) - Rust Statistics and Vector Algebra Library
- [PyO3/rust-numpy](https://github.com/PyO3/rust-numpy) - PyO3-based Rust binding of NumPy C-API
- [rust-cv/nshare](https://github.com/rust-cv/nshare) - Conversion between ndarray and nalgebra types


## Dataframe

It might want to try `polars` for now. `datafusion` looks good too.

- [ritchie46/polars](https://github.com/ritchie46/polars) - Rust DataFrame library
- [apache/arrow](https://github.com/apache/arrow-rs) - In-memory columnar format, in Rust.
- [apache/arrow-datafusion](https://github.com/apache/arrow-datafusion) - Apache Arrow DataFusion and Ballista query engines
- [milesgranger/black-jack](https://github.com/milesgranger/black-jack) - DataFrame / Series data processing in Rust
- [nevi-me/rust-dataframe](https://github.com/nevi-me/rust-dataframe) - A Rust DataFrame implementation, built on Apache Arrow
- [kernelmachine/utah](https://github.com/kernelmachine/utah) - Dataframe structure and operations in Rust
- [sinhrks/brassfibre](https://github.com/sinhrks/brassfibre) - Provides multiple-dtype columner storage, known as DataFrame in pandas/R


## Image Processing

It might want to try `image-rs` for now. Algorithms such as linear transformations are implemented in other libraries as well.

- [image-rs/image](https://github.com/image-rs/image) - Encoding and decoding images in Rust
    - [image-rs/imageproc](https://github.com/image-rs/imageproc) - Image processing operations
- [rust-cv/ndarray-image](https://github.com/rust-cv/ndarray-image) - Allows conversion between ndarray's types and image's types
- [rust-cv/cv](https://github.com/rust-cv/cv) - Rust CV mono-repo. Contains pure-Rust dependencies which attempt to encapsulate the capability of OpenCV, OpenMVG, and vSLAM frameworks in a cohesive set of APIs.
- [twistedfall/opencv-rust](https://github.com/twistedfall/opencv-rust) - Rust bindings for OpenCV 3 & 4
- [rustgd/cgmath](https://github.com/rustgd/cgmath) - A linear algebra and mathematics library for computer graphics.
- [atomashpolskiy/rustface](https://github.com/atomashpolskiy/rustface) - Face detection library for the Rust programming language


## Natural Language Processing (preprocessing)

- [google-research/deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) - This repository contains code to deduplicate language model datasets as descrbed in the paper "Deduplicating Training Data Makes Language Models Better" by Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch and Nicholas Carlini. This repository contains both the ExactSubstr deduplication implementation (written in Rust) along with the scripts we used in the paper to perform deduplication and inspect the results (written in Python). In an upcoming update, we will add files to reproduce the NearDup-deduplicated versions of the C4, RealNews, LM1B, and Wiki-40B-en datasets.
- [pemistahl/lingua-rs](https://github.com/pemistahl/lingua-rs) - 👄 The most accurate natural language detection library in the Rust ecosystem, suitable for long and short text alike
- [usamec/cntk-rs](https://github.com/usamec/cntk-rs) - Wrapper around Microsoft CNTK library
- [stickeritis/sticker](https://github.com/stickeritis/sticker) - A LSTM/Transformer/dilated convolution sequence labeler
- [tensordot/syntaxdot](https://github.com/tensordot/syntaxdot) - Neural syntax annotator, supporting sequence labeling, lemmatization, and dependency parsing.
- [christophertrml/rs-natural](https://github.com/christophertrml/rs-natural) - Natural Language Processing for Rust
- [bminixhofer/nnsplit](https://github.com/bminixhofer/nnsplit) - Semantic text segmentation. For sentence boundary detection, compound splitting and more.
- [greyblake/whatlang-rs](https://github.com/greyblake/whatlang-rs) - Natural language detection library for Rust.
- [finalfusion/finalfrontier](https://github.com/finalfusion/finalfrontier) - Context-sensitive word embeddings with subwords. In Rust.
- [bminixhofer/nlprule](https://github.com/bminixhofer/nlprule) - A fast, low-resource Natural Language Processing and Error Correction library written in Rust.
- [rth/vtext](https://github.com/rth/vtext) - Simple NLP in Rust with Python bindings
- [tamuhey/tokenizations](https://github.com/tamuhey/tokenizations) - Robust and Fast tokenizations alignment library for Rust and Python
- [vgel/treebender](https://github.com/vgel/treebender) - A HDPSG-inspired symbolic natural language parser written in Rust
- [reinfer/blingfire-rs](https://github.com/reinfer/blingfire-rs) - Rust wrapper for the BlingFire tokenization library
- [CurrySoftware/rust-stemmers](https://github.com/CurrySoftware/rust-stemmers) - Common stop words in a variety of languages
- [cmccomb/rust-stop-words](https://github.com/cmccomb/rust-stop-words) - Common stop words in a variety of languages
- [Freyskeyd/nlp](https://github.com/Freyskeyd/nlp) - Rust-nlp is a library to use Natural Language Processing algorithm with RUST
- [Daniel-Liu-c0deb0t/uwu](https://github.com/Daniel-Liu-c0deb0t/uwu) - fastest text uwuifier in the west


## Graphical Modeling

- [alibaba/GraphScope](https://github.com/alibaba/GraphScope) - GraphScope: A One-Stop Large-Scale Graph Computing System from Alibaba
- [petgraph/petgraph](https://github.com/petgraph/petgraph) - Graph data structure library for Rust.
- [rs-graph/rs-graph](https://chiselapp.com/user/fifr/repository/rs-graph/doc/release/README.md) - rs-graph is a library for graph algorithms and combinatorial optimization
- [metamolecular/gamma](https://github.com/metamolecular/gamma) - A graph library for Rust.
- [purpleprotocol/graphlib](https://github.com/purpleprotocol/graphlib) - Simple but powerful graph library for Rust
- [yamafaktory/hypergraph](https://github.com/yamafaktory/hypergraph) - Hypergraph is a data structure library to generate directed hypergraphs

## Interface & Pipeline & AutoML

- [modelfoxdotdev/modelfox](https://github.com/modelfoxdotdev/modelfox) - Modelfox is an all-in-one automated machine learning framework. https://github.com/modelfoxdotdev/modelfox
- [datafuselabs/datafuse](https://github.com/datafuselabs/datafuse) - A Modern Real-Time Data Processing & Analytics DBMS with Cloud-Native Architecture, written in Rust
- [mstallmo/tensorrt-rs](https://github.com/mstallmo/tensorrt-rs) - Rust library for running TensorRT accelerated deep learning models
- [pipehappy1/tensorboard-rs](https://github.com/pipehappy1/tensorboard-rs) - Write TensorBoard events in Rust.
- [ehsanmok/tvm-rust](https://github.com/ehsanmok/tvm-rust) - Rust bindings for TVM runtime
- [vertexclique/orkhon](https://github.com/vertexclique/orkhon) - Orkhon: ML Inference Framework and Server Runtime
- [xaynetwork/xaynet](https://github.com/xaynetwork/xaynet) - Xaynet represents an agnostic Federated Machine Learning framework to build privacy-preserving AI applications
- [pykeio/ort](https://github.com/pykeio/ort) - Idiomatic bindings to the C++ onnx runtime library
- [webonnx/wonnx](https://github.com/webonnx/wonnx) - A GPU-accelerated ONNX inference run-time written 100% in Rust, ready for the web
- [sonos/tract](https://github.com/sonos/tract) - Tiny, no-nonsense, self-contained, Tensorflow and ONNX inference
- [robertknight/rten](https://github.com/robertknight/rten) - Small and efficient ONNX inference engine
- [MegEngine/MegFlow](https://github.com/MegEngine/MegFlow) - Efficient ML solutions for long-tailed demands.


## Workflow

- [substantic/rain](https://github.com/substantic/rain) - Framework for large distributed pipelines
- [timberio/vector](https://github.com/timberio/vector) - A high-performance, highly reliable, observability data pipeline


## GPU

- [Rust-GPU/Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA) - Ecosystem of libraries and tools for writing and executing extremely fast GPU code fully in Rust.
- [EmbarkStudios/rust-gpu](https://github.com/EmbarkStudios/rust-gpu) - 🐉 Making Rust a first-class language and ecosystem for GPU code 🚧
- [termoshtt/accel](https://github.com/termoshtt/accel) - GPGPU Framework for Rust
- [kmcallister/glassful](https://github.com/kmcallister/glassful) - Rust-like syntax for OpenGL Shading Language
- [MaikKlein/rlsl](https://github.com/MaikKlein/rlsl) - Rust to SPIR-V compiler
- [japaric-archived/nvptx](https://github.com/japaric-archived/nvptx) - How to: Run Rust code on your NVIDIA GPU
- [msiglreith/inspirv-rust](https://github.com/msiglreith/inspirv-rust) - Rust (MIR) → SPIR-V (Shader) compiler



# Comprehensive (like sklearn)

All libraries support the following algorithms.

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


It might want to try `smartcore` or `linfa` for now.

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
- [mbillingr/openml-rust](https://github.com/mbillingr/openml-rust) - A rust interface to http://openml.org/


# Comprehensive (Statistics)

- [statrs-dev/statrs](https://github.com/statrs-dev/statrs) - Statistical computation library for Rust
- [rust-ndarray/ndarray-stats](https://github.com/rust-ndarray/ndarray-stats) - Statistical routines for ndarray
- [Axect/Peroxide](https://github.com/Axect/Peroxide) - Rust numeric library with R, MATLAB & Python syntax
    - Linear Algebra, Functional Programming, Automatic Differentiation, Numerical Analysis, Statistics, Special functions, Plotting, Dataframe
- [tarcieri/micromath](https://github.com/tarcieri/micromath) - Embedded Rust arithmetic, 2D/3D vector, and statistics library


# Gradient Boosting

- [mesalock-linux/gbdt-rs](https://github.com/mesalock-linux/gbdt-rs) - MesaTEE GBDT-RS : a fast and secure GBDT library, supporting TEEs such as Intel SGX and ARM TrustZone
- [davechallis/rust-xgboost](https://github.com/davechallis/rust-xgboost) - Rust bindings for XGBoost.
- [vaaaaanquish/lightgbm-rs](https://github.com/vaaaaanquish/lightgbm-rs) - LightGBM Rust binding
- [catboost/catboost](https://github.com/catboost/catboost/tree/master/catboost/rust-package) - A fast, scalable, high performance Gradient Boosting on Decision Trees library, used for ranking, classification, regression and other machine learning tasks (predict only)
- [Entscheider/stamm](https://github.com/entscheider/stamm) - Generic decision trees for rust


# Deep Neural Network

`Tensorflow bindings` and `PyTorch bindings` are the most common.
`tch-rs` also has torch vision, which is useful.

- [tensorflow/rust](https://github.com/tensorflow/rust) - Rust language bindings for TensorFlow
- [LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs) - Rust bindings for the C++ api of PyTorch.
- [VasanthakumarV/einops](https://github.com/vasanthakumarv/einops) - Simplistic API for deep learning tensor operations
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
- [sakex/neat-gru-rust](https://github.com/sakex/neat-gru-rust) - neat-gru
- [nerosnm/n2](https://github.com/nerosnm/n2) - (Work-in-progress) library implementation of a feedforward, backpropagation artificial neural network
- [Wuelle/deep_thought](https://github.com/Wuelle/deep_thought) - Neural Networks in Rust
- [MikhailKravets/NeuroFlow](https://github.com/MikhailKravets/NeuroFlow) - Awesome deep learning crate
- [dvigneshwer/deeprust](https://github.com/dvigneshwer/deeprust) - Machine learning crate in Rust
- [millardjn/rusty_sr](https://github.com/millardjn/rusty_sr) - Deep learning superresolution in pure rust
- [coreylowman/dfdx](https://github.com/coreylowman/dfdx) - Strongly typed Deep Learning in Rust
- [tracel-ai/burn](https://github.com/tracel-ai/burn) - Flexible and comprehensive Deep Learning in Rust
- [huggingface/candle](https://github.com/huggingface/candle) - Huggingface created Deep Learning framework in Rust

# Graph Model

- [Synerise/cleora](https://github.com/Synerise/cleora) - Cleora AI is a general-purpose model for efficient, scalable learning of stable and inductive entity embeddings for heterogeneous relational data.
- [Pardoxa/net_ensembles](https://github.com/Pardoxa/net_ensembles) - Rust library for random graph ensembles


# Natural Language Processing (model)

- [huggingface/tokenizers](https://github.com/huggingface/tokenizers/tree/master/tokenizers) - The core of tokenizers, written in Rust. Provides an implementation of today's most used tokenizers, with a focus on performance and versatility.
- [guillaume-be/rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers) - Rust-tokenizer offers high-performance tokenizers for modern language models, including WordPiece, Byte-Pair Encoding (BPE) and Unigram (SentencePiece) models
- [guillaume-be/rust-bert](https://github.com/guillaume-be/rust-bert) - Rust native ready-to-use NLP pipelines and transformer-based models (BERT, DistilBERT, GPT2,...)
- [sno2/bertml](https://github.com/sno2/bertml) - Use common pre-trained ML models in Deno!
- [cpcdoy/rust-sbert](https://github.com/cpcdoy/rust-sbert) - Rust port of sentence-transformers (https://github.com/UKPLab/sentence-transformers)
- [vongaisberg/gpt3_macro](https://github.com/vongaisberg/gpt3_macro) - Rust macro that uses GPT3 codex to generate code at compiletime
- [proycon/deepfrog](https://github.com/proycon/deepfrog) - An NLP-suite powered by deep learning
- [ferristseng/rust-tfidf](https://github.com/ferristseng/rust-tfidf) - Library to calculate TF-IDF
- [messense/fasttext-rs](https://github.com/messense/fasttext-rs) - fastText Rust binding
- [mklf/word2vec-rs](https://github.com/mklf/word2vec-rs) - pure rust implementation of word2vec
- [DimaKudosh/word2vec](https://github.com/DimaKudosh/word2vec) - Rust interface to word2vec.
- [lloydmeta/sloword2vec-rs](https://github.com/lloydmeta/sloword2vec-rs) - A naive (read: slow) implementation of Word2Vec. Uses BLAS behind the scenes for speed.


# Recommendation

- [PersiaML/PERSIA](https://github.com/PersiaML/PERSIA) - High performance distributed framework for training deep learning recommendation models based on PyTorch.
- [jackgerrits/vowpalwabbit-rs](https://github.com/jackgerrits/vowpalwabbit-rs) - 🦀🐇 Rusty VowpalWabbit
- [outbrain/fwumious_wabbit](https://github.com/outbrain/fwumious_wabbit) - Fwumious Wabbit, fast on-line machine learning toolkit written in Rust
- [hja22/rucommender](https://github.com/hja22/rucommender) - Rust implementation of user-based collaborative filtering
- [maciejkula/sbr-rs](https://github.com/maciejkula/sbr-rs) - Deep recommender systems for Rust
- [chrisvittal/quackin](https://github.com/chrisvittal/quackin) - A recommender systems framework for Rust
- [snd/onmf](https://github.com/snd/onmf) - fast rust implementation of online nonnegative matrix factorization as laid out in the paper "detect and track latent factors with online nonnegative matrix factorization"
- [rhysnewell/nymph](https://github.com/rhysnewell/nymph) - Non-Negative Matrix Factorization in Rust


# Information Retrieval

## Full Text Search

- [quickwit-inc/quickwit](https://github.com/quickwit-inc/quickwit) - Quickwit is a big data search engine.
- [bayard-search/bayard](https://github.com/bayard-search/bayard) - A full-text search and indexing server written in Rust.
- [neuml/txtai.rs](https://github.com/neuml/txtai.rs) - AI-powered search engine for Rust
- [meilisearch/MeiliSearch](https://github.com/meilisearch/MeiliSearch) - Lightning Fast, Ultra Relevant, and Typo-Tolerant Search Engine
- [toshi-search/Toshi](https://github.com/toshi-search/Toshi) - A full-text search engine in rust
- [BurntSushi/fst](https://github.com/BurntSushi/fst) - Represent large sets and maps compactly with finite state transducers.
- [tantivy-search/tantivy](https://github.com/tantivy-search/tantivy) - Tantivy is a full-text search engine library inspired by Apache Lucene and written in Rust
- [tinysearch/tinysearch](https://github.com/tinysearch/tinysearch) - 🔍 Tiny, full-text search engine for static websites built with Rust and Wasm
- [quantleaf/probly-search](https://github.com/quantleaf/probly-search) - A lightweight full-text search library that provides full control over the scoring calculations
- [https://github.com/andylokandy/simsearch-rs](https://github.com/andylokandy/simsearch-rs) - A simple and lightweight fuzzy search engine that works in memory, searching for similar strings
- [jameslittle230/stork](https://github.com/jameslittle230/stork) - 🔎 Impossibly fast web search, made for static sites.
- [elastic/elasticsearch-rs](https://github.com/elastic/elasticsearch-rs) - Official Elasticsearch Rust Client


## Nearest Neighbor Search

- [Enet4/faiss-rs](https://github.com/Enet4/faiss-rs) - Rust language bindings for Faiss
- [rust-cv/hnsw](https://github.com/rust-cv/hnsw) - HNSW ANN from the paper "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
- [rust-cv/hgg](https://github.com/rust-cv/hgg) - HNSW inspired algorithm "Hierarchical Greedy Graphs" - fully deterministic and optimised for areas of computer vision where HNSW would be used
- [hora-search/hora](https://github.com/hora-search/hora) - 🚀 efficient approximate nearest neighbor search algorithm collections library, which implemented with Rust 🦀. horasearch.com
- [InstantDomain/instant-distance](https://github.com/InstantDomain/instant-distance) - Fast approximate nearest neighbor searching in Rust, based on HNSW index
- [lerouxrgd/ngt-rs](https://github.com/lerouxrgd/ngt-rs) - Rust wrappers for NGT approximate nearest neighbor search
- [granne/granne](https://github.com/granne/granne) - Graph-based Approximate Nearest Neighbor Search
- [u1roh/kd-tree](https://github.com/u1roh/kd-tree) - k-dimensional tree in Rust. Fast, simple, and easy to use.
- [qdrant/qdrant](https://github.com/qdrant/qdrant) - Qdrant - vector similarity search engine with extended filtering support
- [rust-cv/hwt](https://github.com/rust-cv/hwt) - Hamming Weight Tree from the paper "Online Nearest Neighbor Search in Hamming Space"
- [fulara/kdtree-rust](https://github.com/fulara/kdtree-rust) - kdtree implementation for rust.
- [mrhooray/kdtree-rs](https://github.com/mrhooray/kdtree-rs) - K-dimensional tree in Rust for fast geospatial indexing and lookup
- [kornelski/vpsearch](https://github.com/kornelski/vpsearch) - C library for finding nearest (most similar) element in a set
- [petabi/petal-neighbors](https://github.com/petabi/petal-neighbors) - Nearest neighbor search algorithms including a ball tree and a vantage point tree.
- [ritchie46/lsh-rs](https://github.com/ritchie46/lsh-rs) - Locality Sensitive Hashing in Rust with Python bindings
- [kampersanda/mih-rs](https://github.com/kampersanda/mih-rs) - Rust implementation of multi-index hashing for neighbor searches on 64-bit codes in the Hamming space


# Reinforcement Learning

- [taku-y/border](https://github.com/taku-y/border) - Border is a reinforcement learning library in Rust.
- [NivenT/REnforce](https://github.com/NivenT/REnforce) - Reinforcement learning library written in Rust
- [edlanglois/relearn](https://github.com/edlanglois/relearn) - Reinforcement learning with Rust
- [tspooner/rsrl](https://github.com/tspooner/rsrl) - A fast, safe and easy to use reinforcement learning framework in Rust.
- [milanboers/rurel](https://github.com/milanboers/rurel) - Flexible, reusable reinforcement learning (Q learning) implementation in Rust
- [Ragnaroek/bandit](https://github.com/Ragnaroek/bandit) - Bandit Algorithms in Rust
- [MrRobb/gym-rs](https://github.com/mrrobb/gym-rs) - OpenAI Gym bindings for Rust


# Supervised Learning Model

- [tomtung/omikuji](https://github.com/tomtung/omikuji) - An efficient implementation of Partitioned Label Trees & its variations for extreme multi-label classification
- [shadeMe/liblinear-rs](https://github.com/shademe/liblinear-rs) - Rust language bindings for the LIBLINEAR C/C++ library.
- [messense/crfsuite-rs](https://github.com/messense/crfsuite-rs) - Rust binding to crfsuite
- [ralfbiedert/ffsvm-rust](https://github.com/ralfbiedert/ffsvm-rust) - FFSVM stands for "Really Fast Support Vector Machine"
- [zenoxygen/bayespam](https://github.com/zenoxygen/bayespam) - A simple bayesian spam classifier written in Rust.
- [Rui_Vieira/naive-bayesnaive-bayes](https://gitlab.com/ruivieira/naive-bayes) - A Naive Bayes classifier written in Rust.
- [Rui_Vieira/random-forests](https://gitlab.com/ruivieira/random-forests) - A Rust library for Random Forests.
- [sile/randomforest](https://github.com/sile/randomforest) - A random forest implementation in Rust
- [tomtung/craftml-rs](https://github.com/tomtung/craftml-rs) - A Rust🦀 implementation of CRAFTML, an Efficient Clustering-based Random Forest for Extreme Multi-label Learning
- [nkaush/naive-bayes-rs](https://github.com/nkaush/naive-bayes-rs) - A Rust library with homemade machine learning models to classify the MNIST dataset. Built in an attempt to get familiar with advanced Rust concepts.


# Unsupervised Learning & Clustering Model

- [frjnn/bhtsne](https://github.com/frjnn/bhtsne) - Barnes-Hut t-SNE implementation written in Rust.
- [vaaaaanquish/label-propagation-rs](https://github.com/vaaaaanquish/label-propagation-rs) - Label Propagation Algorithm by Rust. Label propagation (LP) is graph-based semi-supervised learning (SSL). LGC and CAMLP have been implemented.
- [nmandery/extended-isolation-forest](https://github.com/nmandery/extended-isolation-forest) - Rust port of the extended isolation forest algorithm for anomaly detection
- [avinashshenoy97/RusticSOM](https://github.com/avinashshenoy97/RusticSOM) - Rust library for Self Organising Maps (SOM).
- [diffeo/kodama](https://github.com/diffeo/kodama) - Fast hierarchical agglomerative clustering in Rust.
- [kno10/rust-kmedoids](https://github.com/kno10/rust-kmedoids) - k-Medoids clustering in Rust with the FasterPAM algorithm
- [petabi/petal-clustering](https://github.com/petabi/petal-clustering) - DBSCAN and OPTICS clustering algorithms.
- [savish/dbscan](https://github.com/savish/dbscan) - A naive DBSCAN implementation in Rust
- [gu18168/DBSCANSD](https://github.com/gu18168/DBSCANSD) - Rust implementation for DBSCANSD, a trajectory clustering algorithm.
- [lazear/dbscan](https://github.com/lazear/dbscan) - Dependency free implementation of DBSCAN clustering in Rust
- [whizsid/kddbscan-rs](https://github.com/whizsid/kddbscan-rs) - A rust library inspired by kDDBSCAN clustering algorithm
- [Sauro98/appr_dbscan_rust](https://github.com/Sauro98/appr_dbscan_rust) - Program implementing the approximate version of DBSCAN introduced by Gan and Tao
- [quietlychris/density_clusters](https://github.com/quietlychris/density_clusters) - A naive density-based clustering algorithm written in Rust
- [milesgranger/gap_statistic](https://github.com/milesgranger/gap_statistic) - Dynamically get the suggested clusters in the data for unsupervised learning.
- [genbattle/rkm](https://github.com/genbattle/rkm) - Generic k-means implementation written in Rust
- [selforgmap/som-rust](https://github.com/selforgmap/som-rust) - Self Organizing Map (SOM) is a type of Artificial Neural Network (ANN) that is trained using an unsupervised, competitive learning to produce a low dimensional, discretized representation (feature map) of higher dimensional data.


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
- [Quickwit: A highly cost-efficient search engine in Rust](https://quickwit.io/blog/quickwit-first-release/), 2021/7/13, quickwit, PAUL MASUREL
- [Data Manipulation: Polars vs Rust](https://able.bio/haixuanTao/data-manipulation-polars-vs-rust--3def44c8), 2021/3/13, Xavier Tao
- [Check out Rust in Production](https://serokell.io/blog/rust-in-production-qovery), 2021/8/10, Qovery, @serokell
- [Why I started Rust instead of stick to Python](https://medium.com/geekculture/why-i-started-rust-instead-of-stick-to-python-626bab07479a), 2021/9/26, Medium, Geek Culture, Marshal SHI


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
- [Simple Linear Regression from scratch in Rust](https://cheesyprogrammer.com/2018/12/13/simple-linear-regression-from-scratch-in-rust/), Web Development, Software Architecture, Algorithms and more, 2018/12/13, philipp
- [Interactive Rust in a REPL and Jupyter Notebook with EVCXR](https://depth-first.com/articles/2020/09/21/interactive-rust-in-a-repl-and-jupyter-notebook-with-evcxr/), Depth-First, 2020/9/21, Richard L. Apodaca
- [Rust for Data Science: Tutorial 1](https://dev.to/davidedelpapa/rust-for-data-science-tutorial-1-4g5j), dev, 2021/8/25, Davide Del Papa
- [petgraph_review](https://timothy.hobbs.cz/rust-play/petgraph_review.html), 2019/10/11, Timothy Hobbs
- [Rust for ML. Rust](https://medium.com/tempus-ex/rust-for-ml-fba0421b0959), Medium, Tempus Ex, 2021/8/1, Michael Naquin
- [Adventures in Drone Photogrammetry Using Rust and Machine Learning (Image Segmentation with linfa and DBSCAN)](http://cmoran.xyz/writing/adventures_in_photogrammetry), 2021/11/14, CHRISTOPHER MORAN


### Apply

- [Deep Learning in Rust: baby steps](https://medium.com/@tedsta/deep-learning-in-rust-7e228107cccc), Medium,  2016/2/2, Theodore DeRego
- [A Rust SentencePiece implementation](https://guillaume-be.github.io/2020-05-30/sentence_piece), Rust NLP tales, 2020/5/30
- [Accelerating text generation with Rust](https://guillaume-be.github.io/2020-11-21/generation_benchmarks), Rust NLP tales, 2020/11/21
- [A Simple Text Summarizer written in Rust](https://towardsdatascience.com/a-simple-text-summarizer-written-in-rust-4df05f9327a5), Towards Data Science, 2020/11/24, [Charles Chan](https://chancharles.medium.com/), Examples of Text Sentence Vector, Cosine Distance and PageRank
- [Extracting deep learning image embeddings in Rust](https://logicai.io/blog/extracting-image-embeddings/), RecoAI, 2021/6/1, Paweł Jankiewic, Examples of ONNX
- [Deep Learning in Rust with GPU](https://able.bio/haixuanTao/deep-learning-in-rust-with-gpu--26c53a7f), 2021/7/30, Xavier Tao
- [tch-rs pretrain example - Docker for PyTorch rust bindings tch-rs. Example of pretrain model](https://github.com/vaaaaanquish/tch-rs-pretrain-example-docker), 2021/8/15, vaaaaanquish
- [Rust ANN search Example - Image search example by approximate nearest-neighbor library In Rust](https://github.com/vaaaaanquish/rust-ann-search-example), 2021/8/15, vaaaaanquish
- [dzamkov/deep-learning-test - Implementing deep learning in Rust using just a linear algebra library (nalgebra)](https://github.com/dzamkov/deep-learning-test), 2021/8/30, dzamkov
- [vaaaaanquish/rust-machine-learning-api-example - The axum example that uses resnet224 to infer images received in base64 and returns the results.](https://github.com/vaaaaanquish/rust-machine-learning-api-example), 2021/9/7, vaaaaanquish
- [Rust for Machine Learning: Benchmarking Performance in One-shot - A Rust implementation of Siamese Neural Networks for One-shot Image Recognition for benchmarking performance and results](https://utmist.gitlab.io/projects/rust-ml-oneshot/), UofT Machine Intelligence Student Team
- [Why Wallaroo Moved From Pony To Rust](https://wallarooai.medium.com/why-wallaroo-moved-from-pony-to-rust-292e7339fc34), 2021/8/19, Wallaroo.ai
- [epwalsh/rust-dl-webserver - Example of serving deep learning models in Rust with batched prediction](https://github.com/epwalsh/rust-dl-webserver), 2021/11/16, epwalsh


### Case study

- [Production users - Rust Programming Language](https://www.rust-lang.org/production/users), by rust-lang.org
- [Taking ML to production with Rust: a 25x speedup](https://www.lpalmieri.com/posts/2019-12-01-taking-ml-to-production-with-rust-a-25x-speedup/), A LEARNING JOURNAL, 2019/12/1, [@algo_luca](https://twitter.com/algo_luca)
- [9 Companies That Use Rust in Production](https://serokell.io/blog/rust-companies), serokell, 2020/11/18, Gints Dreimanis
- [Masked Language Model on Wasm, BERT on flontend examples](https://github.com/optim-corp/masked-lm-wasm/), optim-corp/masked-lm-wasm, 2021/8/27, Optim
- [Serving TensorFlow with Actix-Web](https://github.com/kykosic/actix-tensorflow-example), kykosic/actix-tensorflow-example
- [Serving PyTorch with Actix-Web](https://github.com/kykosic/actix-pytorch-example), kykosic/actix-pytorch-example


## Discussion

- [Natural Language Processing in Rust : rust](https://www.reddit.com/r/rust/comments/5jj8vr/natural_language_processing_in_rust), 2016/12/6
- [Future prospect of Machine Learning in Rust Programming Language : MachineLearning](https://www.reddit.com/r/MachineLearning/comments/7iz51p/d_future_prospect_of_machine_learning_in_rust/), 2017/11/11
- [Interest for NLP in Rust? - The Rust Programming Language Forum](https://users.rust-lang.org/t/interest-for-nlp-in-rust/15331), 2018/1/19
- [Is Rust good for deep learning and artificial intelligence? - The Rust Programming Language Forum](https://users.rust-lang.org/t/is-rust-good-for-deep-learning-and-artificial-intelligence/22866), 2018/11/18
- [ndarray vs nalgebra : rust](https://www.reddit.com/r/rust/comments/btn1cz/ndarray_vs_nalgebra/), 2019/5/28
- [Taking ML to production with Rust | Hacker News](https://news.ycombinator.com/item?id=21680965), 2019/12/2
- [Who is using Rust for Machine learning in production/research? : rust](https://www.reddit.com/r/rust/comments/fvehyq/d_who_is_using_rust_for_machine_learning_in/), 2020/4/5
- [Deep Learning in Rust](https://www.reddit.com/r/rust/comments/igz8iv/deep_learning_in_rust/), 2020/8/26
- [SmartCore, fast and comprehensive machine learning library for Rust! : rust](https://www.reddit.com/r/rust/comments/j1mj1g/smartcore_fast_and_comprehensive_machine_learning/), 2020/9/29
- [Deep Learning in Rust with GPU on ONNX](https://www.reddit.com/r/MachineLearning/comments/ouul33/d_p_deep_learning_in_rust_with_gpu_on_onnx/), 2021/7/31
- [Rust vs. C++ the main differences between these popular programming languages](https://codilime.com/blog/rust-vs-cpp-the-main-differences-between-these-popular-programming-languages/), 2021/8/25
- [I wanted to share my experience of Rust as a deep learning researcher](https://www.reddit.com/r/rust/comments/pft9n9/i_wanted_to_share_my_experience_of_rust_as_a_deep/), 2021/9/2
- [How far along is the ML ecosystem with Rust?](https://www.reddit.com/r/rust/comments/poglgg/how_far_along_is_the_ml_ecosystem_with_rust/), 2021/9/15


## Books

- [Practical Machine Learning with Rust: Creating Intelligent Applications in Rust (English Edition)](https://amzn.to/3h7JV8U), 2019/12/10, Joydeep Bhattacharjee
    - Write machine learning algorithms in Rust
    - Use Rust libraries for different tasks in machine learning
    - Create concise Rust packages for your machine learning applications
    - Implement NLP and computer vision in Rust
    - Deploy your code in the cloud and on bare metal servers
    - source code: [Apress/practical-machine-learning-w-rust](https://github.com/Apress/practical-machine-learning-w-rust)
- [DATA ANALYSIS WITH RUST NOTEBOOKS](https://datacrayon.com/shop/product/data-analysis-with-rust-notebooks/), 2021/9/3, Shahin Rostami
    - Plotting with Plotters and Plotly
    - Operations with ndarray
    - Descriptive Statistics
    - Interactive Diagram
    - Visualisation of Co-occurring Types
    - download source code and dataset
    - full text
        - [https://datacrayon.com/posts/programming/rust-notebooks/preface/](https://datacrayon.com/posts/programming/rust-notebooks/preface/)


## Movie

- [The /r/playrust Classifier: Real World Rust Data Science](https://www.youtube.com/watch?v=lY10kTcM8ek), RustConf 2016, 2016/10/05, Suchin Gururangan & Colin O'Brien
- [Building AI Units in Rust](https://www.youtube.com/watch?v=UHFlKAmANJg), FOSSASIA 2018, 2018/3/25, Vigneshwer Dhinakaran 
- [Python vs Rust for Simulation](https://www.youtube.com/watch?v=kytvDxxedWY), EuroPython 2019, 2019/7/10, Alisa Dammer
- [Machine Learning is changing - is Rust the right tool for the job?](https://www.youtube.com/watch?v=odI_LY8AIqo), RustLab 2019, 2019/10/31, Luca Palmieri
- [Using TensorFlow in Embedded Rust](https://www.youtube.com/watch?v=DUVE86yTfKU), 2020/09/29, Ferrous Systems GmbH, Richard Meadows
- [Rust London - Data science takeover](https://www.youtube.com/watch?v=EGlilM0Dtu8), 2020/11/26, 4 talks at Rust London, Andraz Tori, Paul Korebitz, Lorenz Schmidt, Daniel McKenna
- [Writing the Fastest GBDT Library in Rust](https://www.youtube.com/watch?v=D1NAREuicNs), 2021/09/16, RustConf 2021, Isabella Tromba
- [Creating a Text-To-Speech System in Rust](https://www.youtube.com/watch?v=HiqId_9pysM), 2024/03/28, Rustnation 2024, Daniel McKenna


## PodCast

- DATA SCIENCE AT HOME
    - [Rust and machine learning #1 (Ep. 107)](https://datascienceathome.com/rust-and-machine-learning-1-ep-107/)
    - [Rust and machine learning #2 with Luca Palmieri (Ep. 108)](https://datascienceathome.com/rust-and-machine-learning-2-with-luca-palmieri-ep-108/)
    - [Rust and machine learning #3 with Alec Mocatta (Ep. 109)](https://datascienceathome.com/rust-and-machine-learning-3-with-alec-mocatta-ep-109/)
    - [Rust and machine learning #4: practical tools (Ep. 110)](https://datascienceathome.com/rust-and-machine-learning-4-practical-tools-ep-110/)
    - [Machine Learning in Rust: Amadeus with Alec Mocatta (Ep. 127)](https://datascienceathome.com/machine-learning-in-rust-amadeus-with-alec-mocatta-rb-ep-127/)
    - [Rust and deep learning with Daniel McKenna (Ep. 135)](https://datascienceathome.com/rust-and-deep-learning/)
    - [Is Rust flexible enough for a flexible data model? (Ep. 137)](https://datascienceathome.com/is-rust-flexible-enough-for-a-flexible-data-model-ep-137/)
    - [Pandas vs Rust (Ep. 144)](https://datascienceathome.com/pandas-vs-rust-ep-144/)
    - [Apache Arrow, Ballista and Big Data in Rust with Andy Grove (Ep. 145)](https://datascienceathome.com/apache-arrow-ballista-and-big-data-in-rust-with-andy-grove-ep-145/)
    - [Polars: the fastest dataframe crate in Rust (Ep. 146)](https://datascienceathome.com/polars-the-fastest-dataframe-crate-in-rust-ep-146/)
    - [Apache Arrow, Ballista and Big Data in Rust with Andy Grove RB (Ep. 160)](https://datascienceathome.com/apache-arrow-ballista-and-big-data-in-rust-with-andy-grove-rb-ep-160/)


## Paper

- [End-to-end NLP Pipelines in Rust](https://www.aclweb.org/anthology/2020.nlposs-1.4.pdf), Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS), pages 20–25 Virtual Conference, 2020/11/19, Guillaume Becquin


# How to contribute

Please just update the README.md.

If you update this README.md, CI will be executed automatically.
And the website will also be updated.


# Thanks

Thanks for all the projects.

[https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning)
