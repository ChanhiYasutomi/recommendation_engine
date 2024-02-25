# sklearn.metrics.pairwiseモジュールからcosine_similarity関数を使用して、2つのベクトルの余弦類似度を計算するサンプルコードです。
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 2つのサンプルベクトルを定義
vector1 = np.array([1, 2, 3, 4, 5])
vector2 = np.array([5, 4, 3, 2, 1])

# ベクトルを2次元配列に変換（1行目にvector1、2行目にvector2が来るようにする）
vectors = np.array([vector1, vector2])

# cosine_similarity関数を使って余弦類似度を計算
similarity_matrix = cosine_similarity(vectors)

# 結果を表示
print("余弦類似度行列:")
print(similarity_matrix)

# このコードでは、まず2つのサンプルベクトル vector1 と vector2 を定義し、それらを2次元配列 vectors にまとめます。
# 次に cosine_similarity 関数を使って vectors の各行のベクトル間の余弦類似度を計算し、結果を similarity_matrix に格納します。
# 最後に、計算された余弦類似度行列を表示します。
