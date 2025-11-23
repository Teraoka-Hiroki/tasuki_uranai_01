import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ページ設定
st.set_page_config(page_title="Moodleコース レコメンドアプリ", layout="wide")

# 日本語フォント対応（環境に合わせて適宜変更してください）
# Streamlit Cloud等ではjapanize_matplotlibが便利ですが、ない場合は英語ラベルになります
try:
    import japanize_matplotlib
    FONT_AVAILABLE = True
except ImportError:
    FONT_AVAILABLE = False

# ---------------------------------------------------------
# 1. データ読み込みと初期設定
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # CSVファイルの読み込み（副作用なし）
    csv_file = 'course_learning_path.csv'
    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("エラー: 'course_learning_path.csv' が見つかりません。同じフォルダに配置してください。")
    st.stop()

# クラスターIDとタイプ名の定義（分析結果に基づく）
# 重心計算結果から推定したタイプ名
CLUSTER_NAMES = {
    0: "AI研究者・ハッカー (理論×応用)",
    1: "データサイエンス基礎 (理論×基礎)",
    2: "ITエンジニア基礎 (Web×基礎)",
    3: "スタンダード・総合 (バランス型)",
    4: "AIアプリクリエイター (Web×応用)"
}

# 各クラスターの解説文
CLUSTER_DESC = {
    0: "高度な数理モデルと最新の生成AI技術の両方を深く探究したい、研究志向のあなたにおすすめです。",
    1: "データサイエンスや数学的背景をしっかり固めたい、理論重視のあなたにおすすめです。",
    2: "プログラミングやWebの仕組みなど、ITの基礎体力をつけたいあなたにおすすめです。",
    3: "まずは偏りなく、AI・情報の基礎から応用までをバランスよく学びたいあなたにおすすめです。",
    4: "理屈よりもまずは動くものを！最新の生成AIやWeb技術を使ってアプリを作りたいあなたにおすすめです。"
}

# ---------------------------------------------------------
# 2. サイドバー：アンケート入力
# ---------------------------------------------------------
st.sidebar.header("🔍 あなたの興味・関心")
st.sidebar.write("以下の質問に答えて、あなたにぴったりの学習コースを見つけましょう。")

st.sidebar.markdown("---")

# 質問1: Factor 1 (Web vs 理論)
# 負の値: Web/インフラ, 正の値: 理論/DS
q1 = st.sidebar.slider(
    "Q1. 興味があるのはどっち？",
    min_value=-3.0,
    max_value=3.0,
    value=0.0,
    step=0.5,
    format="%f",
    help="左に行くほど「Web・システム開発」、右に行くほど「数学・理論分析」です。"
)
st.sidebar.caption("Web・アプリ開発 ⟵ 　 ⟶ 数学・データ分析")

st.sidebar.markdown("---")

# 質問2: Factor 2 (生成AI vs 基礎)
# 負の値: 生成AI/応用, 正の値: 基礎/教科情報
q2 = st.sidebar.slider(
    "Q2. 学習スタイルの好みは？",
    min_value=-3.0,
    max_value=3.0,
    value=0.0,
    step=0.5,
    format="%f",
    help="左に行くほど「最新AI活用・実践」、右に行くほど「教科書・基礎理解」です。"
)
st.sidebar.caption("生成AI・実践 ⟵ 　 ⟶ 教科書・基礎")

# ユーザーの座標ベクトル
user_vector = np.array([q1, q2])

# ---------------------------------------------------------
# 3. メインロジック：マッチング
# ---------------------------------------------------------
# 各クラスターの重心（Centroid）を計算
centroids = df.groupby('Cluster')[['Factor1_Score', 'Factor2_Score']].mean()

# ユーザー座標と各重心との距離を計算 (ユークリッド距離)
distances = {}
for cluster_id, row in centroids.iterrows():
    centroid_vector = np.array([row['Factor1_Score'], row['Factor2_Score']])
    dist = np.linalg.norm(user_vector - centroid_vector)
    distances[cluster_id] = float(dist)

# 最も距離が近いクラスターを選択
best_cluster_id = min(distances, key=distances.get)

# CLUSTER_NAMES のキーは int の想定なので変換を試みる
try:
    best_cluster_key = int(best_cluster_id)
except (ValueError, TypeError):
    best_cluster_key = best_cluster_id

best_cluster_name = CLUSTER_NAMES.get(best_cluster_key, f"Cluster {best_cluster_id}")

# ---------------------------------------------------------
# 4. 結果表示画面
# ---------------------------------------------------------
st.title("🎓 レコメンド結果")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"あなたは... **「{best_cluster_name}」** タイプです！")
    st.info(CLUSTER_DESC.get(best_cluster_key, ""))

    # 推奨ルートの表示
    st.markdown("### 🚀 推奨学習ルート")
    st.write("以下の順序で学ぶと、知識を効率よく積み上げることができます。")

    # 選択されたクラスターのデータを抽出してソート
    # Cluster カラムの型差でマッチしないケースを防ぐため文字列比較を使用
    target_courses = df[df['Cluster'].astype(str) == str(best_cluster_id)].sort_values('Recommended_Order')

    if target_courses.empty:
        st.write("該当する推奨コースが見つかりませんでした。データを確認してください。")
    else:
        # リスト表示
        for i, (idx, row) in enumerate(target_courses.iterrows(), 1):
            with st.expander(f"{i}. {row['コース名（短縮）']}"):
                st.write(f"**内容:** {row.get('評価の根拠と特記事項', '詳細なし')}")
                st.write(f"**分野スコア:** 理論度 {row['Factor1_Score']:.2f} / 基礎度 {row['Factor2_Score']:.2f}")

with col2:
    st.markdown("### 🗺️ コースマップ")

    # 散布図の描画
    fig, ax = plt.subplots(figsize=(8, 8))

    # 全コースのプロット
    sns.scatterplot(
        data=df,
        x='Factor1_Score',
        y='Factor2_Score',
        hue='Cluster',
        palette='bright',
        alpha=0.4,
        s=100,
        ax=ax,
        legend=False
    )

    # 選ばれたクラスターを強調（存在する場合のみ）
    if not target_courses.empty:
        sns.scatterplot(
            data=target_courses,
            x='Factor1_Score',
            y='Factor2_Score',
            color='red',
            s=150,
            marker='o',
            label='推奨コース',
            ax=ax
        )

    # ユーザーの位置をプロット（★マーク）
    ax.scatter(
        user_vector[0],
        user_vector[1],
        color='gold',
        s=400,
        marker='*',
        edgecolor='black',
        label='あなた',
        zorder=10
    )

    # 軸とラベル
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_xlabel("Web・システム <---> 理論・数学")
    ax.set_ylabel("生成AI・応用 <---> 基礎・教科書")
    ax.set_title("あなたの立ち位置")
    ax.legend()

    # 日本語フォントがない場合の文字化け対策
    if not FONT_AVAILABLE:
        ax.set_xlabel("Web <---> Theory")
        ax.set_ylabel("GenAI <---> Basic")
        ax.set_title("Your Position")

    st.pyplot(fig)
