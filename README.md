# sign-language-classification
手話単語分類

LSA64(http://facundoq.github.io/unlp/lsa64/)のデータセットをもとに構築した3次元の畳み込みモデルです。

学習データ:2880
検証データ:320

入力：30フレームの動画
出力：ラベル値

※検証データに対して、約88%

<h2>必要なもの</h2>
1.python3
2.tensorflow(version2)

<h2>使い方</h2>
python main.py
