from collections import defaultdict

# データセット内のアイテム間の共起関係を効率的に計算し、アイテム間の関係性や類似性を分析
# 各顧客の購買履歴を取得し、その購買履歴から商品のペアの共起数を直接計算
# 共起数を直接計算するため、データフレーム全体ではなく、各顧客の購買履歴を処理する必要がある
# そのため、データ量が多い場合でも、メモリ使用量を低く抑えることができるが、計算量は比較的高くなる

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



from collections import defaultdict

def generate_cooccurrence_matrix(data):
    cooccurrence_matrix = defaultdict(int) #デフォルトで0を返す辞書 cooccurrence_matrix を初期化(共起行列を表現するためのデータ構造)

    for items in data:
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                item_pair = tuple(sorted([items[i], items[j]])) # 2つのアイテムを取り出し、アイテムのペアをソートして順序付けし、タプルとして item_pair に保存。これにより、同じ組み合わせの順序が異なっても同じキーが生成されるようになる。
                cooccurrence_matrix[item_pair] += 1

    return cooccurrence_matrix

# データフレームから必要なデータを取得
cooccurrence_per_product = df_customer.groupby('customer_id')['product_code'].apply(list).reset_index(name='product_list')

# 共起行列の生成
cooccurrence_matrix = generate_cooccurrence_matrix(cooccurrence_per_product['product_list'])

# 共起行列をデータフレームに変換
df_cooccurrence = pd.DataFrame(cooccurrence_matrix.items(), columns=['item_pair', 'co_count'])
df_cooccurrence[['item1', 'item2']] = pd.DataFrame(df_cooccurrence['item_pair'].tolist(), index=df_cooccurrence.index)
df_cooccurrence.drop('item_pair', axis=1, inplace=True)
df_cooccurrence = df_cooccurrence[(df_cooccurrence['item1'] != df_cooccurrence['item2'])]
display(df_cooccurrence.sort_values(by = 'co_count', ascending = False))

# 候補生成
df_cooccurrence_candidate = df_cooccurrence[df_cooccurrence['co_count'] > 350]
df_cooccurrence_unique_candidate = set(df_cooccurrence_candidate['item1']).union(set(df_cooccurrence_candidate['item2']))
len(df_cooccurrence_unique_candidate)
