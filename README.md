# abeam_presentation_analysis

## Description
[WINGS-FMSP](https://www.ms.u-tokyo.ac.jp/wings-fmsp/)プログラムにおける
2024年度社会数理実践研究にて、ABeam Consulting社より課題/データ提供をいただき、
以下の分析を実行しました。

- ABeam Consulting社の社員の方々のプレゼンテーション動画から音声特徴量を抽出し、
どのような特徴がプレゼンの上手さに寄与するのかをロジスティック回帰により分析しました。
- Separationの問題に対処するため、Firthの方法によるペナルティ付き最尤法でパラメータを推定しています。
- 詳細は近日中に[数理科学実践研究レター](https://www.ms.u-tokyo.ac.jp/lmsr/)に掲載される予定です。
- `praat-parselmouth`や`rpy2`などのGPL-v2+ライセンスのライブラリを含むため、本プロジェクトもGPL-v3ライセンスの下で公開しています。

## Notebooks
### notebooks/preprocess.ipynb
生のプレゼン音声データに前処理を施しています。

### notebooks/calc_features.ipynb
前処理済のプレゼンデータから、各種特徴量を計算しています。

### notebooks/logistic_firth.ipynb
加工済のデータセットに基づいて、Firthの方法を用いたロジスティック回帰を実行します。
Pythonの`firthlogist`パッケージの実装を利用しています。

### notebooks/logistic_firth_r.ipynb
Pythonの `firthlogist`パッケージの信頼性を確認するために、Rの`logistf`パッケージを用いて同様の分析を行い、整合性を確認しています。
Rの`logistf`パッケージは、"A solution to the problem of separation in logistic regression" (<https://doi.org/10.1002/sim.1047>) の著者であるG. Heinze氏により開発されているようです。

### notebooks/sim_estimation.ipynb
シミュレーションにより、実データに近い設定でFirthの方法がうまく機能することを調べています。
- Separationの解決策として単純にはL2 penaltyを課すことが思いつきますが、それに比べてFirthの方法によるパラメータ推定値の方が、バイアスも分散も小さくなることを確認しました。
  - 特徴量が多変量正規分布していると仮定し、実データから推定されたパラメータに基づくロジット構造から人工データを生成します。
  - L2 penaltyの係数はLOO CVでチューニングしました。

### notebooks/aiueo_formant.ipynb
私が「あ」「い」「う」「え」「お」と発声した録音データに対して、Praatによるフォルマント推定を試しています。


## Description(EN)
As part of the [WINGS-FMSP](https://www.ms.u-tokyo.ac.jp/wings-fmsp/) program of Practical Social Mathematics Research in 2024, we received tasks and data from ABeam Consulting, and conducted the following analysis:

- We extracted audio features from presentation videos of ABeam Consulting employees and analyzed which features contribute to presentation quality using logistic regression.
- To address the problem of separation, we estimated the parameters using the penalized maximum likelihood method with Firth's correction.
- More details will be published soon in the [Practical Mathematical Science Research Letter](https://www.ms.u-tokyo.ac.jp/lmsr/).
- Since libraries such as `praat-parselmouth` and `rpy2` are licensed under GPL-v3, this project is also released under the GPL-v3 license.

## Notebooks(EN)
### notebooks/preprocess.ipynb
This notebook applies preprocessing to the raw presentation audio data.

### notebooks/calc_features.ipynb
This notebook calculates various features from the preprocessed presentation data.

### notebooks/logistic_firth.ipynb
This notebook runs logistic regression using Firth's method based on the processed dataset. It utilizes the `firthlogist` package in Python.

### notebooks/logistic_firth_r.ipynb
To verify the reliability of the Python `firthlogist` package, we performed the same analysis using the `logistf` package in R and confirmed the consistency. The `logistf` package in R was developed by G. Heinze, the author of "A solution to the problem of separation in logistic regression" (<https://doi.org/10.1002/sim.1047>).

### notebooks/sim_estimation.ipynb
This notebook explores the effectiveness of Firth's method in a setting close to real data through simulations.
- As a simple solution to separation, one might apply an L2 penalty, but we confirmed that Firth's method produces parameter estimates with lower bias and variance compared to L2-penalized estimates.
  - Assuming that the features follow a multivariate normal distribution, we generated synthetic data based on the logistic structure estimated from real data.
  - The L2 penalty coefficient was tuned using leave-one-out cross-validation (LOO CV).

### notebooks/aiueo_formant.ipynb
This notebook tests formant estimation using Praat on my recordings of the vowels "a", "i", "u", "e", and "o".
