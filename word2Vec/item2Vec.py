### item2Vec
# item2vecは、Word2Vecのアイデアをアイテム（商品、アイテム、またはコンテンツ）に適用した手法
# └ Word2Vecが単語の意味を表現するために単語埋め込みを学習するのに対し、item2vecはアイテムの意味や関連性を表現するためにアイテム埋め込みを学習する。

# item2vecは、レコメンドシステムにおいて特に有用
# └ 顧客が過去に購入または閲覧したアイテムの履歴を入力データとして使用し、顧客が興味を持つであろうアイテムを推薦するための埋め込みを学習をすることで、アイテム同士の類似性や関連性を捉えることができる。

# item2vecを使用する際には、Word2Vecと同様に、入力データをセンテンス（アイテムのシーケンス）のリストとして提供する。各センテンスは、顧客ごとのアイテムの購入または閲覧履歴などのアイテムのシーケンス。その後、Word2Vecと同様の手法でアイテム埋め込みを学習する。

# item2vecを使用することで、レコメンドシステムは顧客が興味を持ちそうなアイテムを効果的に特定し、レコメンドできる。

pip install gensim

# 絞り込みはここで行う?

# 例: ユーザー毎に購入したアイテムを時系列順に並べる
sorted_purchase_history = df_customer.groupby('customer_id').apply(lambda x: x.sort_values('order_date')['product_code'].tolist())
display(sorted_purchase_history) # 各顧客の購買パターンを表している

# 学習
from gensim.models import Word2Vec

# sentences:モデルのトレーニングに使用されるテキストデータのリストまたはイテレーター, 通常、このテキストデータは単語のシーケンスであり、各シーケンスはトレーニング用の文や文章を表す
# vector_size:単語埋め込みの次元数を指定。各単語がベクトルとして表現される次元数を定義する, 一般的には、埋め込みの次元数が大きいほど、モデルがより複雑な意味を表現できるようになる
# window:コンテキストウィンドウのサイズを指定, トレーニング中にモデルが考慮する単語の前後の単語の数を定義する
# min_count:モデルの語彙に含める単語の最小出現回数を指定。出現回数がこの値未満の単語は無視され、語彙に含まれない, トレーニングにおいてレアな単語やノイズとなる単語を除外することができる。
# sg:モデルの種類を指定, sg=0 の場合はCBOW（Continuous Bag of Words）モデルが使用され、sg=1 の場合はSkip-gramモデルが使用される。CBOWは周囲の単語から中心の単語を予測するモデルであり、Skip-gramは中心の単語から周囲の単語を予測するモデル

# Word2Vecモデルのトレーニング
# 各顧客の購買のパターンに基づいて顧客間の類似性や製品間の関連性を学習する
# 購買履歴をテキストデータと見なしてWord2Vecモデルに学習させることで、顧客や製品の特徴を抽出し、その特徴を用いて顧客間の類似性や製品間の関連性を計算している
model = Word2Vec(sentences=sorted_purchase_history, vector_size=100, window=5, min_count=1, sg=1)

# 学習済みモデルを保存
# model.wv.save_word2vec_format('test_competition.bin', binary = True)

# # モデルのロード
# from gensim.models import KeyedVectors
# model = KeyedVectors.load_word2vec_format('/content/test_competition.bin', binary=True)

# レコメンドするための関数
def recommend_items(user_history):
    similar_items = model.wv.most_similar(user_history, topn=10)
    return [item[0] for item in similar_items]

# 並行処理を実現するためのモジュール
# 複数のスレッドやプロセスを使ってタスクを同時に実行し、並行性を高めることができる
import concurrent.futures

# レコメンド関数（concurrent.futuresを使用）
def recommend_items_parallel(customer_id):
    # ユーザーの購買履歴を取得
    user_history = df_customer[df_customer['customer_id'] == customer_id]['product_code'].tolist()

    # レコメンドを行い、トップ10の結果を返す
    return customer_id, recommend_items(user_history)[:10]

# レコメンド結果を保存するための辞書
recommendations_dict = {}

# 全ユーザーのcustomer_idを取得
# all_customer_ids = df_customer['customer_id'].sample(n=100, random_state=42).unique()
all_customer_ids = df_customer['customer_id'].unique()

# ThreadPoolExecutorを使用して並列処理を行う
# concurrent.futuresモジュールを使って並列処理を行う方法の1つがThreadPoolExecutor
# ThreadPoolExecutorは、スレッドを使って非同期に関数を実行し、並列処理を行う
# これにより、CPUバウンドではないI/Oバウンドのタスクを効率的に処理できる
with concurrent.futures.ThreadPoolExecutor() as executor:
    # 各ユーザーごとにrecommend_items_parallel関数を並列実行し、結果を辞書に保存
    results = executor.map(recommend_items_parallel, all_customer_ids)
    for customer_id, recommended_items in results:
        recommendations_dict[customer_id] = recommended_items

# 辞書からデータフレームを作成
df_recommendations = pd.DataFrame(list(recommendations_dict.items()), columns=['customer_id', 'recommended_items'])
df_recommendations = df_recommendations.explode('recommended_items')
df_recommendations = df_recommendations.rename(columns = {'recommended_items' : 'product_code'})
df_recommendations

# 並列処理を使用すると処理が早くなる主な理由

# リソースの効率的な利用: 複数の処理を同時に実行することで、CPUやその他のリソースをより効率的に利用できる
# 複数のスレッドやプロセスを使って複数のタスクを同時に処理することで、待ち時間を最小限に抑えることができる

# スケーラビリティ: 並列処理を使うと、処理をスケールアップしやすくなる
# 新たな処理ノードを追加することで、処理能力を簡単に増やすことができる

# タスクの分割: 大きなタスクを複数の小さなタスクに分割して並列に処理することで、全体としての処理時間を短縮できる

# I/O待ちの削減: I/Oバウンドなタスクでは、処理がI/O待ちでブロックされることがある
# 複数のタスクを並列に処理することで、I/O待ち時間を最小限に抑えることができる

# 並列処理を効果的に行うには、適切なタスクの分割や同期の管理が必要
# また、スレッドやプロセス間でのデータ共有や競合状態の回避など、注意が必要な点もある



# I/Oバウンド: プログラムがI/O（Input/Output、入出力）操作に時間を費やしている状態を指す
# ファイルの読み書き、ネットワーク通信、データベースアクセスなどの操作が該当する

# プログラムがI/Oバウンドである場合、CPUの処理速度よりもデータの読み書き速度やネットワークの応答速度などが遅くなることが一般的
# そのため、プログラムがI/O処理を待っている間、CPUはほとんど使用されずに待機状態になる

# ウェブアプリケーションでは、ユーザーからのリクエストを受け取り、データベースからデータを取得してそれを表示するという処理が一般的
# この場合、データベースアクセスやネットワーク通信などのI/O操作に時間がかかるため、プログラムはI/Oバウンド状態になる

# I/Oバウンドな処理では、プログラムのパフォーマンスを向上させるために、非同期処理や並列処理などの手法が使用されることがある
# これにより、プログラムがI/O操作を待っている間に他の処理を行うことができ、全体としての効率が向上する
