# Awesome-Rust-MachineLearning

日本語向けのrustクレートや記事等をまとめたもの

- For English: [README.md](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning/blob/main/README.md)
- For Japanese: [README.ja.md](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning/blob/main/README.ja.md)

# Undercarriage

## Natural Language Processing (preprocessing)

lindraはneologdなどの辞書資産が利用できる。sudachi.rsはワークスが巻き取る形になったので今後に期待。Unofficialとなったがsudachiclone-rsというものも存在する。vaporettoはKyTeaの早いやつ。

- [lindera-morphology/lindera](https://github.com/lindera-morphology/lindera) - A morphological analysis library.
- [legalforce-research/vaporetto](https://github.com/legalforce-research/vaporetto) - Vaporetto: Pointwise prediction based tokenizer
- [sorami/sudachi.rs](https://github.com/sorami/sudachi.rs) - An unofficial Sudachi clone in Rust (incomplete) 🦀
- [Yasu-umi/sudachiclone-rs](https://github.com/Yasu-umi/sudachiclone-rs) - sudachiclone-rs is a Rust version of Sudachi, a Japanese morphological analyzer.
- [wareya/notmecab-rs](https://github.com/wareya/notmecab-rs) - notmecab-rs is a very basic mecab clone, designed only to do parsing, not training.
- [agatan/yoin](https://github.com/agatan/yoin) - A Japanese Morphological Analyzer written in pure Rust
- [nakagami/awabi](https://github.com/nakagami/awabi) - A morphological analyzer using mecab dictionary


# Reference

## Blog


### 解説やチュートリアル、やってみた

- [Rust で重回帰](https://sinhrks.hatenablog.com/entry/2015/11/14/230721), 2015/11/14, StatsFragments
- [Rust で主成分分析](https://sinhrks.hatenablog.com/entry/2015/11/15/232129), 2015/11/15, StatsFragments
- [Rust で階層的クラスタリング](https://sinhrks.hatenablog.com/entry/2015/11/21/000845), 2015/11/21, StatsFragments
- [Rust で k-means クラスタリング](https://sinhrks.hatenablog.com/entry/2015/12/07/003842), 2015/12/7, StatsFragments
- [RustでDeepLearning入門](https://qiita.com/ta_to_co/items/c5686bc702b9255b9c06), 2019/9/1, ta_to_co
- [sudachi.rsを使って遊んでみる](https://clutte.red/blog/2020/04/play-with-sudachi-rs/), 2020/4/16, Cluttered Room
- [RustのドローイングライブラリPlottersの紹介](https://lab.mo-t.com/blog/rust-plotters), MoT Lab, 2020/8/4, _tkato_
- [WebAssemblyでの機械学習モデルデプロイの動向](https://tkat0.github.io/posts/deploy-ml-as-wasm), 2020/12/2, tkat0
- [Rustによるlindera、neologd、fasttext、XGBoostを用いたテキスト分類](https://vaaaaaanquish.hatenablog.com/entry/2020/12/14/192246), 2020/12/14, vaaaaaanquish
- [『ゼロから作る Deep Learning』を読んで Rust で実装した話](https://qiita.com/Surpris/items/823b60caf554ecd36d20), 2020/12/15, Surpris2021
- [Rustで扱える機械学習関連のクレート2021](https://vaaaaaanquish.hatenablog.com/entry/2021/01/23/233113), 2021/1/23, vaaaaaanquish
- [Rust の機械学習ライブラリ smartcore に入門してみた](https://zenn.dev/mattn/articles/3290149a6fc18c), 2021/7/10, mattn
- [Rustによる機械学習概覧を技術書典11に寄稿するまでの軌跡](https://vaaaaaanquish.hatenablog.com/entry/2021/07/10/110000), 2021/7/10, vaaaaaanquish
- [バンディッドアルゴリズム(Epsilon-greedy)の実装](https://dev.classmethod.jp/articles/bandit/), 2021/7/16, DevelopersIO, 中村 修太
- [SmartCoreでペンギンの分類をやってみる](https://dev.classmethod.jp/articles/smartcore-palmer/), 2021/7/27, DevelopersIO, 中村 修太
- [Pure Rustな近似最近傍探索ライブラリhoraを用いた画像検索を実装する](https://vaaaaaanquish.hatenablog.com/entry/2021/08/10/065117), 2021/8/10, vaaaaaanquish
- [WebAssemblyを用いてBERTモデルをフロントエンドで動かす](https://tech-blog.optim.co.jp/entry/2021/08/13/100000), 2021/8/13, OPTiM
- [Rustでlabel propagationを実装した](https://vaaaaaanquish.hatenablog.com/entry/2021/08/27/062345), 2021/8/27, vaaaaaanquish
- [Rust×WASMに入門する(Linderaでブラウザから形態素解析)](https://shine-bal.hatenablog.com/entry/2021/08/15/210915), 2021/8/15



### 実装紹介

- [Rust初心者がRustで全文検索サーバを作ってみた](https://qiita.com/mosuka/items/3517ef4a1eb07fa2661f), 2020/1/27, mosuka
- [Rust初心者がRust製の日本語形態素解析器の開発を引き継いでみた](https://qiita.com/mosuka/items/0fdaaf91f5530d427dc7), 2020/9/11, mosuka
- [LinderaをTantivyで使えるようにした](https://qiita.com/mosuka/items/5261c90c8cdc0be90f91), 2020/3/3, mosuka
- [日本語形態素解析器 SudachiPy の 現状と今後について(Sudachi.rs開発がワークスに譲渡された事が公開)](https://speakerdeck.com/waptech/ri-ben-yu-xing-tai-su-jie-xi-qi-sudachipy-false-xian-zhuang-tojin-hou-nituite?slide=28), 2021/7/6, WAP
- [Rustによる自然言語処理ツールの実装: 形態素解析器「sudachi.rs」](https://qiita.com/sorami/items/7934fec2074c493c0f7d), 2021/7/7, sorami


### 事例

- [エッジMLシステムをC/C++からRustへ移行した事例](https://docs.google.com/presentation/d/1HOL9jheJnKkh2q7w3hU_px-je1qL7lxrSXV-0P1hces/edit?usp=sharing), Rust.Tokyo 2019, 2019/10/26, DeNA, tkat0
- [Rustで作る機械学習を用いた画像クロッピングシステム](https://ml-loft.connpass.com/event/157785/), ML@Loft #9, 2019/12/19, Cookpad, johshisha
- [fnwiya/japanese-rust-companies: 日本で Rust を利用している会社一覧](https://github.com/fnwiya/japanese-rust-companies)
- [WebAssemblyで機械学習Webアプリ「俺か俺以外か」をつくった](https://vaaaaaanquish.hatenablog.com/entry/2020/12/26/120837), 2020/12/26, vaaaaaanquish


### 動画

- [Full use of Rust on edge and cloud: AI and IoT use cases エッジとクラウドでRustを使いこなす ～AI/IoTでの事例～](https://rustfest.global/session/10-%E3%82%A8%E3%83%83%E3%82%B8%E3%81%A8%E3%82%AF%E3%83%A9%E3%82%A6%E3%83%89%E3%81%A7rust%E3%82%92%E4%BD%BF%E3%81%84%E3%81%93%E3%81%AA%E3%81%99-%EF%BD%9Eai-iot%E3%81%A7%E3%81%AE%E4%BA%8B%E4%BE%8B%EF%BD%9E/), RUSTFEST, 2020/11/7, Mobility Technologies, tkat0


### repo

- [vaaaaanquish/wasm_lindera_example](https://github.com/vaaaaanquish/wasm_lindera_example) - rust + lindera + webassembly + next.js + typescriptで形態素解析するサンプル
- [vaaaaanquish/rust-text-analysis](https://github.com/vaaaaanquish/rust-text-analysis) - Rustによるlindera、neologd、fasttext、XGBoostを用いたテキスト分類のお試し
