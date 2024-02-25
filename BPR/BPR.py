### BPR
# ・Bayesian Personalized Ranking (BPR) は、ユーザーにとってアイテムの順位付けを学習するための推薦システムのためのモデル
# ・Collaborative Filtering (CF) の一種であり、ユーザーが過去に評価したアイテムの情報を利用して、未評価のアイテムの順位付けを行う

# BPRは、以下のような特徴を持つ
# └ 隠れ変数モデル: BPRは、ユーザーとアイテムの間の相互作用を表現するための隠れ変数を使用する
# これにより、モデルはユーザーの好みやアイテムの特徴を表現し、それらの間の関係を学習する

# └ ランキング学習: モデルはユーザーが好むアイテムを他のアイテムよりも高いランクに配置するように学習する
# これにより、ユーザーにとって最も好ましいアイテムを推薦できる

# └ ベイジアンアプローチ: 事後分布を推定し、その分布からランキングを推定する
# これにより、不確実性を考慮して推薦を行うことができる

pip install implicit

# BPRの実装
df_customer['customer_label'], customer_idx = pd.factorize(df_customer['customer_id'])
df_customer['product_code_label'], product_idx = pd.factorize(df_customer['product_code'])

# 疎行列（sparse matrix）を作成(疎行列は、行列のほとんどの要素が0であるような行列を効率的に表現するためのデータ構造)
from scipy.sparse import csr_matrix
sparse_df = csr_matrix((np.ones(len(df_customer)), (df_customer['customer_label'], df_customer['product_code_label'])))

import implicit

# モデルの埋め込みの次元数、反復回数、学習率、正則化を設定
factors = 64  # 埋め込みの次元数を指定。モデルがユーザーとアイテムの特徴を表現するために使用するベクトルの次元数。値が大きいほど、モデルの表現力が高くなるが、計算コストも増加する。
iterations = 500  # モデルの学習を行う反復回数を指定。反復回数が多いほど、モデルはより多くのデータを使用して学習するが、計算時間も増加する。
learning_rate = 0.1  # 学習率は、モデルのパラメーターを更新する際に使用されるステップサイズを制御する。値が小さいほど、更新されるパラメーターの変化が小さくなる。
regularization = 0.1  # 正則化は、モデルの過学習を防ぐために使用される。正則化項は、学習時のコスト関数に加算され、大きなパラメーター値にペナルティを課すことで、パラメーターの大きな変動を抑制する。

# ベイジアン個人化ランキングモデルの初期化
model = implicit.bpr.BayesianPersonalizedRanking(factors=factors, iterations=iterations, learning_rate=learning_rate, regularization=regularization, random_state=42)

# モデルの学習
model.fit(sparse_df)

# ユーザーと製品の情報をベクトル化
customer2emb = dict(zip(customer_idx, model.user_factors))
product2emb = dict(zip(product_idx, model.item_factors))

# import pickle
# 2つの辞書をそれぞれバイナリファイルに書き込み
# with open('bpr_customer2emb.pkl', 'wb') as fp:
#     pickle.dump(customer2emb, fp)

# with open('bpr_product2emb.pkl', 'wb') as fp:
#     pickle.dump(product2emb, fp)

# # ファイルからデータを読み込み
# with open('bpr_customer2emb.pkl', 'rb') as fp:
#     loaded_customer2emb = pickle.load(fp)

# with open('bpr_product2emb.pkl', 'rb') as fp:
#     loaded_product2emb = pickle.load(fp)

# コサイン類似度での商品での絞り込み
# テスト
# 類似度を計算したい商品
target_product = '0-0-0-468'

# コサイン類似度を計算
similarities = {}
for product_id, feature_vector in product2emb.items():
    if product_id != target_product:
        similarity = cosine_similarity([product2emb[target_product]], [feature_vector])
        similarities[product_id] = similarity[0][0]

# 類似度が高い順にソート
sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

# 上位N件の類似商品を取得
top_n = 5
top_similar_products = sorted_similarities[:top_n]

# 結果をリストに格納
similar_products_list = []
for product_id, similarity in top_similar_products:
    similar_products_list.append({'target_product': target_product,
                                  'similar_product': product_id,
                                  'similarity': similarity})

# リストをデータフレームに変換
similar_products_df = pd.DataFrame(similar_products_list)
similar_products_df



# 類似度を計算したい商品のリスト
target_products = pd.DataFrame(df_cooccurrence_unique_candidate)[0].tolist()

# 上位N件の類似商品を格納するデータフレーム
similar_products_df = pd.DataFrame(columns=['target_product', 'similar_product', 'similarity'])

# コサイン類似度を計算
for target_product in target_products:
    similarities = {}
    for product_id, feature_vector in product2emb.items():
        if product_id != target_product:
            similarity = cosine_similarity([product2emb[target_product]], [feature_vector])
            similarities[product_id] = similarity[0][0]

    # 類似度が高い順にソート
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # 上位N件の類似商品を取得
    top_n = 5
    top_similar_products = sorted_similarities[:top_n]

    # 結果をデータフレームに追加
    rows = [{'target_product': target_product,
             'similar_product': similar_product,
             'similarity': similarity} for similar_product, similarity in top_similar_products]
    similar_products_df = pd.concat([similar_products_df, pd.DataFrame(rows)], ignore_index=True)

similar_products_df
# aaa = similar_products_df['target_product'].unique()

## ベクトルとの内積
# ・ユーザーのベクトルと各製品のベクトルの内積を計算することにより、ユーザーが各製品に対してどれだけ興味を持っているかを評価することができる
# ・内積の値が高いほど、ユーザーと製品の間の類似度が高いことを意味する
# ・内積が高い製品は、ユーザーにとって関連性が高いと見なされ、ユーザーの好みや興味に合った製品である可能性が高いことを示す
# ・そのため、レコメンドでは、ユーザーのベクトルと製品のベクトルの内積を計算して、ユーザーに最適な製品を見つけるための重要な手法として使用される

# 内積の値を正規化する関数
def normalize_dot_product(dot_product):
    # 内積が1を超える場合、1にクリップする
    if dot_product > 1.0:
        return 1.0
    # 内積が-1を下回る場合、-1にクリップする
    elif dot_product < -1.0:
        return -1.0
    # それ以外の場合、そのままの値を返す
    else:
        return dot_product

# 指定されたユーザーIDに対して推薦商品を生成するための関数
def recommend_products(user_id, customer2emb, product2emb, candidate_items, top_n=10):
    # 指定されたユーザーIDに対応するユーザーのベクトルを取得
    user_vector = customer2emb.get(user_id)
    if user_vector is None:
        return "User IDが見つかりません"

    # ユーザーのベクトルと候補商品のベクトルの内積を計算し、正規化する
    scores = {}
    for product_id in candidate_items:
        product_vector = product2emb.get(product_id)
        if product_vector is not None:
            dot_product = np.dot(user_vector, product_vector)
            normalized_dot_product = normalize_dot_product(dot_product)
            scores[product_id] = normalized_dot_product

    # 正規化された内積の降順で商品をソート
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 上位n個の商品を取得
    top_products = sorted_scores[:top_n]

    return top_products

# 推薦された商品のリストからユーザーIDと商品ID、スコアを持つデータフレームを作成する関数
def create_recommendation_dataframe(user_id, recommended_products):
    # 推薦された商品IDとスコアを格納するリストを初期化
    recommended_product_ids = []
    scores = []

    # 推薦された商品IDとスコアをリストに追加
    for product_id, score in recommended_products:
        recommended_product_ids.append(product_id)
        scores.append(score)

    # ユーザーIDをリピートしてリストに追加
    user_ids = [user_id] * len(recommended_products)

    # ユーザーID、商品ID、スコアを持つデータフレームを作成
    recommendation_df = pd.DataFrame({
        'customer_id': user_ids,
        'product_code': recommended_product_ids,
        'score': scores
    })

    return recommendation_df
def get_recommendation_metrics(df, unique_items):
    # データフレームのproduct_code列に含まれる値のセットを取得
    recommended_items_set = set(df['product_code'])

    # unique_itemに含まれる商品コードをリストアップ
    candidate_items = unique_items

    # 推薦された商品の中で実際に適切なもの（候補アイテムに含まれるもの）の割合を計算
    true_count = sum(item in candidate_items for item in recommended_items_set)
    true_rate = true_count / len(recommended_items_set)

    return true_rate



# テスト
# ユーザーIDを指定して推薦商品を生成
user_id = '287843'
# user_ids = df_customer['customer_id'].unique()

recommended_products = recommend_products(user_id, customer2emb, product2emb, df_cooccurrence_unique_candidate)

# 推薦された商品のデータフレームを作成
df_recommendation = create_recommendation_dataframe(user_id, recommended_products)
df_recommendation
ユーザーIDの列を持つデータフレームからユーザーIDを取得
user_ids = df_customer['customer_id'].unique()



# ユーザーごとに推薦された商品を格納するリストを初期化
dfs_recommendation = []

# ユーザーごとに推薦商品を生成し、データフレームに変換
for user_id in user_ids:
    recommended_products = recommend_products(user_id, customer2emb, product2emb, df_cooccurrence_unique_candidate)
    df_recommendation = create_recommendation_dataframe(user_id, recommended_products)
    dfs_recommendation.append(df_recommendation)

# 推薦された商品のデータフレームを結合
df_combined_recommendation = pd.concat(dfs_recommendation, ignore_index=True)
df_combined_recommendation = df_combined_recommendation[['customer_id', 'product_code']]



import concurrent.futures

# ユーザーごとに推薦された商品を格納するリストを初期化
dfs_recommendation = []

# ユーザーごとに推薦商品を生成する関数
def process_user(user_id):
    recommended_products = recommend_products(user_id, customer2emb, product2emb, df_cooccurrence_unique_candidate)
    df_recommendation = create_recommendation_dataframe(user_id, recommended_products)
    return df_recommendation

# 並列処理の実行
with concurrent.futures.ThreadPoolExecutor() as executor:
    # ユーザーごとに推薦商品を生成する処理をマルチスレッドで実行
    futures = [executor.submit(process_user, user_id) for user_id in user_ids]

    # 完了した順に結果を取得してリストに追加
    for future in concurrent.futures.as_completed(futures):
        df_recommendation = future.result()
        dfs_recommendation.append(df_recommendation)

# 推薦された商品のデータフレームを結合
df_combined_recommendation = pd.concat(dfs_recommendation, ignore_index=True)
df_combined_recommendation = df_combined_recommendation[['customer_id', 'product_code']]



## コサイン類似度
from sklearn.metrics.pairwise import cosine_similarity

# ユーザーに最も興味を持ちそうな商品をtop10でを推薦するための関数
def recommend_products(user_id, customer2emb, product2emb, candidate_items, top_n=10):
    # ユーザーのベクトルを取得
    user_vector = customer2emb.get(user_id)
    if user_vector is None:
        return "User IDが見つかりません"

    # ユーザーのベクトルと候補商品のベクトルのコサイン類似度を計算
    similarities = {}
    for product_id in candidate_items: #商品を絞り込む
        product_vector = product2emb.get(product_id)
        if product_vector is not None:
            similarity = cosine_similarity([user_vector], [product_vector])[0][0]
            similarities[product_id] = similarity

    # コサイン類似度に基づいて商品をランキング
    ranked_products = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # 上位n個の商品を取得
    top_products = ranked_products[:top_n]

    return top_products

# レコメンドされた商品のリストからデータフレームを作成
def create_recommendation_dataframe(user_id, recommended_products):
    # 商品IDとコサイン類似度のリストを取得
    product_ids = [product[0] for product in recommended_products]
    similarities = [product[1] for product in recommended_products]

    # データフレームを作成
    df_recommendation = pd.DataFrame({
        'user_id': [user_id] * len(product_ids),  # ユーザーIDをリピートしてリストに追加
        'product_code': product_ids,
        'cosine_similarity': similarities
    })

    return df_recommendation
  
# ユーザーIDごとにレコメンドを行い、データフレームを作成(計算コスト高め)
recommendation_dfs = []
for user_id in df_customer['customer_id'].unique():
    recommended_products = recommend_products(user_id, customer2emb, product2emb, top20_per_age_candidate_items) # 商品を絞り込み。
    recommendation_df = create_recommendation_dataframe(user_id, recommended_products)
    recommendation_dfs.append(recommendation_df)

# すべてのデータフレームを結合
final_recommendation_df = pd.concat(recommendation_dfs, ignore_index=True)
final_submission_df = final_recommendation_df[['user_id', 'product_code']]



# 並列処理
import concurrent.futures

# dfs_recommendation リストを初期化
dfs_recommendation = []

# ユーザーごとのレコメンデーションを計算する関数
def process_user(user_id):
    recommended_products = recommend_products(user_id, customer2emb, product2emb, df_cooccurrence_unique_candidate)
    df_recommendation = create_recommendation_dataframe(user_id, recommended_products)
    return df_recommendation

# 並列処理の実行
with concurrent.futures.ThreadPoolExecutor() as executor:
    # df_customer['customer_id'].unique() からユーザーIDを取得してマルチスレッドで処理
    futures = [executor.submit(process_user, user_id) for user_id in df_customer['customer_id'].unique()]
    # 完了した順に結果を取得して dfs_recommendation に追加
    for future in concurrent.futures.as_completed(futures):
        dfs_recommendation.append(future.result())

# recommendation_dfs を1つの DataFrame に結合
df_final_recommendation = pd.concat(dfs_recommendation, ignore_index=True)
df_final_submission = df_final_recommendation[['user_id', 'product_code']]
