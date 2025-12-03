import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm

# ---------------------------------------------------------
# ページ設定
# ---------------------------------------------------------
st.set_page_config(page_title="Moodleコース レコメンドアプリ", layout="wide")

# ---------------------------------------------------------
# 日本語フォント設定（修正版）
# ---------------------------------------------------------
def configure_japanese_font():
    """
    日本語フォントを適用し、成功したかどうかを返す関数。
    Seabornのテーマ設定にもフォントを明示的に渡すことで文字化けを防ぐ。
    """
    # 1. japanize_matplotlib (最優先)
    try:
        import japanize_matplotlib
        japanize_matplotlib.japanize()
        # 【重要】Seabornのテーマ設定にフォント名を明示的に渡す
        sns.set_theme(style="whitegrid", font="IPAexGothic")
        return True, "IPAexGothic"
    except ImportError:
        pass

    # 2. システムフォント探索（japanize_matplotlibがない場合のバックアップ）
    target_fonts = [
        'Noto Sans CJK JP', 'Noto Sans JP', 'Hiragino Sans', 
        'Hiragino Kaku Gothic ProN', 'Yu Gothic', 'Meiryo', 
        'TakaoGothic', 'IPAGothic', 'IPAexGothic', 'VL Gothic', 'MS Gothic'
    ]
    
    # システムにインストールされているフォント一覧を取得
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    
    found_font = None
    for font in target_fonts:
        if font in available_fonts:
            found_font = font
            break
            
    if found_font:
        # Matplotlibのデフォルト設定を更新
        plt.rcParams['font.family'] = found_font
        # Seabornのテーマ設定にもフォントを適用
        sns.set_theme(style="whitegrid", font=found_font)
        return True, found_font

    # 3. フォントが見つからない場合
    # 英語設定のままにするが、最低限Seabornのスタイルは適用
    sns.set_theme(style="whitegrid")
    return False, None

# 設定を実行
FONT_AVAILABLE, USED_FONT = configure_japanese_font()

# デバッグ用メッセージ（不要ならコメントアウト）
if FONT_AVAILABLE:
    # st.toast(f"日本語フォント適用中: {USED_FONT}", icon="✅")
    pass
else:
    st.toast("日本語フォントが見つかりませんでした。英語モードで表示します。", icon="⚠️")

# マイナス記号の文字化け対策
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 0. ダミーデータ生成（CSVがない場合のフォールバック）
# ---------------------------------------------------------
def create_dummy_csv():
    data = {
        'Cluster': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        'Factor1_Score': [2.5, 2.0, 1.5, 1.8, -2.0, -1.5, 0.0, 0.2, -2.5, -1.8],
        'Factor2_Score': [-1.5, -2.0, 2.0, 1.5, 1.0, 1.5, 0.1, -0.1, -2.0, -1.5],
        'コース名（短縮）': [
            '高度AI理論', '生成AI実装特論', '統計数学基礎', 'データ分析入門',
            'Web開発基礎', 'Linuxサーバー構築', '情報リテラシー', 'ITパスポート対策',
            'Reactアプリ開発', '最新API活用'
        ],
        '評価の根拠と特記事項': [
            '最新論文の輪読を行います', 'LLMのファインチューニング', '確率統計の基礎から', 'Pythonでのデータ操作',
            'HTML/CSS/JSの基礎', 'コマンドライン操作', 'PCの基本操作', '資格取得向け',
            'モダンフロントエンド', '生成AI APIの活用'
        ],
        'Recommended_Order': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    }
    df = pd.DataFrame(data)
    df.to_csv('course_learning_path.csv', index=False)
    return df

# ---------------------------------------------------------
# 1. データ読み込みと初期設定
# ---------------------------------------------------------
@st.cache_data
def load_data():
    csv_file = 'course_learning_path.csv'
    if not os.path.exists(csv_file):
        return create_dummy_csv()
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("データの読み込みに失敗しました。")
    st.stop()

# クラスターIDとタイプ名の定義
CLUSTER_NAMES = {
    0: "AI研究者・ハッカー (理論×応用)",
    1: "データサイエンス基礎 (理論×基礎)",
    2: "ITエンジニア基礎 (Web×基礎)",
    3: "スタンダード・総合 (バランス型)",
    4: "AIアプリクリエイター (Web×応用)"
}

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

q1 = st.sidebar.slider(
    "Q1. 興味があるのはどっち？",
    min_value=-3.0, max_value=3.0, value=0.0, step=0.5,
    help="左：Web・システム開発 ／ 右：数学・データ分析"
)
st.sidebar.caption("Web・アプリ開発 ⟵ 　 ⟶ 数学・データ分析")

st.sidebar.markdown("---")

q2 = st.sidebar.slider(
    "Q2. 学習スタイルの好みは？",
    min_value=-3.0, max_value=3.0, value=0.0, step=0.5,
    help="左：最新AI活用・実践 ／ 右：教科書・基礎理解"
)
st.sidebar.caption("生成AI・実践 ⟵ 　 ⟶ 教科書・基礎")

user_vector = np.array([q1, q2])

# ---------------------------------------------------------
# 3. メインロジック：マッチング
# ---------------------------------------------------------
centroids = df.groupby('Cluster')[['Factor1_Score', 'Factor2_Score']].mean()

distances = {}
for cluster_id, row in centroids.iterrows():
    centroid_vector = np.array([row['Factor1_Score'], row['Factor2_Score']])
    dist = np.linalg.norm(user_vector - centroid_vector)
    distances[cluster_id] = float(dist)

best_cluster_id = min(distances, key=distances.get)

try:
    best_cluster_key = int(best_cluster_id)
except (ValueError, TypeError):
    best_cluster_key = best_cluster_id

best_cluster_name = CLUSTER_NAMES.get(best_cluster_key, f"Cluster {best_cluster_id}")

# ---------------------------------------------------------
# 4. 結果表示画面
# ---------------------------------------------------------
st.markdown("## テラオカ電子のMoodleコース レコメンドアプリ")
st.markdown("# 『タスク占い』")
st.title("🎓 レコメンド結果")

col1_container = st.container()
with col1_container:
    st.subheader(f"あなたは... **「{best_cluster_name}」** タイプです！")
    st.info(CLUSTER_DESC.get(best_cluster_key, ""))

    st.markdown("### 🚀 推奨学習ルート")
    st.write("以下の順序で学ぶと、知識を効率よく積み上げることができます。")

    target_courses = df[df['Cluster'].astype(str) == str(best_cluster_id)].sort_values('Recommended_Order')

    if target_courses.empty:
        st.write("該当する推奨コースが見つかりませんでした。")
    else:
        for i, (idx, row) in enumerate(target_courses.iterrows(), 1):
            with st.expander(f"{i}. {row['コース名（短縮）']}"):
                st.write(f"**内容:** {row.get('評価の根拠と特記事項', '詳細なし')}")
                st.write(f"**分野スコア:** 理論度 {row['Factor1_Score']:.2f} / 基礎度 {row['Factor2_Score']:.2f}")

st.markdown("---")

col2_container = st.container()
with col2_container:
    st.markdown("### 🗺️ コースマップ")

    fig, ax = plt.subplots(figsize=(8, 8))

    # 全コースのプロット
    sns.scatterplot(
        data=df, x='Factor1_Score', y='Factor2_Score',
        hue='Cluster', palette='bright', alpha=0.4, s=100,
        ax=ax, legend=False
    )

    # 推奨コース
    if not target_courses.empty:
        sns.scatterplot(
            data=target_courses, x='Factor1_Score', y='Factor2_Score',
            color='red', s=150, marker='o', label='推奨コース', ax=ax
        )

    # ユーザーの位置
    ax.scatter(
        user_vector[0], user_vector[1],
        color='gold', s=400, marker='*', edgecolor='black',
        label='あなた', zorder=10
    )

    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    
    # 軸ラベルの設定
    if FONT_AVAILABLE:
        ax.set_xlabel("Web・システム <---> 理論・数学")
        ax.set_ylabel("生成AI・応用 <---> 基礎・教科書")
        ax.set_title("あなたの立ち位置")
        # 凡例の設定（フォントプロパティを再指定してダメ押し）
        ax.legend(prop={'family': USED_FONT})
    else:
        ax.set_xlabel("Web <---> Theory")
        ax.set_ylabel("GenAI <---> Basic")
        ax.set_title("Your Position (Japanese Font Missing)")
        ax.legend()

    st.pyplot(fig)