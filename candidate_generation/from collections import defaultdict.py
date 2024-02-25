from collections import defaultdict

def generate_cooccurrence_matrix(data):
    cooccurrence_matrix = defaultdict(int) # デフォルトで0を返す辞書 cooccurrence_matrix を初期化(共起行列を表現するためのデータ構造)

    for items in data:
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                item_pair = tuple(sorted([items[i], items[j]])) # 2つのアイテムを取り出し、アイテムのペアをソートして順序付けし、タプルとして item_pair に保存。これにより、同じ組み合わせの順序が異なっても同じキーが生成されるようになる。
                cooccurrence_matrix[item_pair] += 1

    return cooccurrence_matrix

# データのサンプル
data = [
    ['A', 'B', 'C'],
    ['A', 'B', 'D'],
    ['B', 'C', 'D'],
    ['A', 'C', 'D']
]

# 共起行列の生成
cooccurrence_matrix = generate_cooccurrence_matrix(data)

# 結果の表示
for key, value in cooccurrence_matrix.items():
    print(key, ':', value)
