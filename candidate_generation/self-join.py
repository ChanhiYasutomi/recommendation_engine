# ログデータを結合し、同じ商品同士のレコードを除外し、製品AとBの出現頻度を集計するプロセスを示しています。

# 共起行列の実装例(データ量によってはメモリクラッシュする)
# 共起行列は、アイテム間の共起関係を表す行列であり、行と列の各要素は対応するアイテムの共起回数を示す

# データフレームを事前に処理してから、商品のペアの共起数を計算(共起数を集計するアプローチ)
# データフレーム内の顧客ごとの商品リストを取得し、その後、商品のペアごとに共起数をカウント
# データフレーム全体を処理するため、メモリ使用量が増加する可能性があるが、計算量は比較的少なくなる
import pandas as pd

# ログデータのサンプルデータを作成
data_customer = {
    'customer_id': [1, 1, 2, 2, 3],
    'product_code': ['A', 'B', 'A', 'C', 'B'],
    'order_id': [101, 102, 103, 104, 105]
}
df_customer = pd.DataFrame(data_customer)

# ログデータ同士を結合
df_customer_cooccurrence = df_customer.merge(df_customer[['customer_id','product_code']], on='customer_id')

# 同じ商品同士のレコードは除外
df_customer_cooccurrence = df_customer_cooccurrence[df_customer_cooccurrence['product_code_x'] != df_customer_cooccurrence['product_code_y']]

# 製品A, Bの出現頻度を集計
df_customer_cooccurrence = df_customer_cooccurrence.groupby(['product_code_x', 'product_code_y'])['order_id'].count().reset_index()
df_customer_cooccurrence.columns = ['product', 'candidate_product', 'co_count']
df_customer_cooccurrence = df_customer_cooccurrence.sort_values(by='co_count', ascending=False)
df_customer_cooccurrence

# 候補生成
# df_customer_cooccurrence_candidate = df_customer_cooccurrence[df_customer_cooccurrence['co_count'] > 300]
# df_customer_cooccurrence_candidate = set(df_customer_cooccurrence_candidate['candidate_product'])
# df_customer_cooccurrence_candidate = set(df_customer_cooccurrence_candidate['product']).union(set(df_customer_cooccurrence_candidate['candidate_product']))
# len(df_customer_cooccurrence_candidate)

# このコードでは、ログデータのサンプルデータを作成し、提供された手順に従ってデータを処理しています。
# 最後に、製品AとBの出現頻度を示すデータフレームを表示しています。

