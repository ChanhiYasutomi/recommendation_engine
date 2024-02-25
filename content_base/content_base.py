# 必要なライブラリをインポート
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

warnings.simplefilter("ignore")
pd.set_option("display.max_columns", None)

# user_dfを作成
user_df = merge_train_df[["customer_id", "attribute_1"]]
# コサイン類似度算出用に、user_dfをonehotベクトル化
user_df = pd.get_dummies(user_df, columns=["attribute_1"])
# ユーザーごとに集約
user_df = user_df.groupby(["customer_id"]).sum().reset_index()

# item_dfを作成
item_df = merge_item_df[["product_code", "attribute_1"]]
# コサイン類似度算出用に、item_dfをonehotベクトル化
item_df = pd.get_dummies(item_df, columns=["attribute_1"])
# アイテムごとに集約
item_df = item_df.groupby(["product_code"]).sum().reset_index()

# user_dfとitem_dfのカラムを取得
user_col_name = list(user_df.columns)
user_col_name.remove("customer_id")

item_col_name = list(item_df.columns)
item_col_name.remove("product_code")

# user_df、item_dfのカラム数を揃えるために、それぞれの差分カラムを取得
user_diff_col = list(set(user_col_name) - set(item_col_name))
item_diff_col = list(set(item_col_name) - set(user_col_name))

# user_df、item_dfから差分カラムを削除
user_df = user_df.drop(user_diff_col, axis=1)
item_df = item_df.drop(item_diff_col, axis=1)

# コサイン類似度算出用に、ユーザーのリスト、user_dfのnumpy配列を作成
user_list = list(user_df["customer_id"])
user_array = np.array(user_df.drop(["customer_id"], axis=1))

# コサイン類似度算出用に、アイテムのリスト、item_dfのnumpy配列を作成
item_list = list(item_df["product_code"])
item_array = np.array(item_df.drop(["product_code"], axis=1))

# コサイン類似度を算出し、類似度上位100アイテムを抽出
# 100アイテム抽出しているのは、すでに購入したアイテムを除外するため
cs = cosine_similarity(user_array, item_array)
sorted_cs = np.argsort(cs)[:, ::-1][:, 0:100]

# 上位100アイテムとIDを紐づけ
item_dict = dict(zip(range(len(item_list)), item_list))
cs_list = list([[item_dict[i] for i in row] for row in sorted_cs])

# ユーザーごとに購入したアイテムのリストを作成
purchased_item_df = train_df[["customer_id", "product_code"]].sort_values(["customer_id", "product_code"]).reset_index(drop=True)
purchased_item_list = list(purchased_item_df.groupby("customer_id")["product_code"].apply(list))

# 作成したリストから上位10件のアイテムのみを抽出
for i in range(len(cs_list)):
    for e in purchased_item_list[i]:
        if e in cs_list[i]:
            cs_list[i].remove(e)
    del cs_list[i][10:]

# ユーザーごとに、上位10件のアイテムを提出用のdfに挿入
submission_df = pd.DataFrame(columns=["customer_id", "product_code"])

current_index = 0
for i in range(len(cs_list)):
        submission_df = pd.concat([submission_df, pd.DataFrame(cs_list[i], columns=["product_code"])], ignore_index=True)
        for j in range(len(cs_list[i])):
                submission_df.loc[current_index+j, "customer_id"] = user_df.iloc[i, 0]
        current_index = current_index+j+1
submission_df.head()
