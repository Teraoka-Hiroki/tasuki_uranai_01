import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from pathlib import Path

# ---------------------------------------------------------
# ページ設定
# ---------------------------------------------------------
st.set_page_config(page_title="Moodleコース レコメンドアプリ", layout="wide")

# ---------------------------------------------------------
# 日本語フォント設定（最も確実な方法）
# ---------------------------------------------------------
@st.cache_resource
def setup_japanese_font():
    """
    日本語フォントを設定する関数
    複数の方法を試して、確実に日本語を表示できるようにする
    """
    # Method 1: japanize-matplotlibを試す
    try:
        import japanize_matplotlib
        japanize_matplotlib.japanize()
        plt.rcParams['axes.unicode_minus'] = False
        return True, "japanize-matplotlib"
    except ImportError:
        pass
    
    # Method 2: システムフォントを探して設定
    # フォントキャッシュを再構築
    fm._load_fontmanager(try_read_cache=False)
    
    japanese_fonts = [
        'Noto Sans CJK JP',
        'Noto Sans JP', 
        'IPAexGothic',
        'IPAGothic',
        'Hiragino Sans',
        'Hiragino Kaku Gothic ProN',
        'Yu Gothic',
        'Meiryo',
        'MS Gothic',
        'TakaoGothic',
        'VL Gothic',
        'Noto Sans Mono CJK JP'
    ]
    
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    for font_name in japanese_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['axes.unicode_minus'] = False
            return True, font_name
    
    # Method 3: DejaVu Sansをフォールバックとして設定（英語のみ）
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    return False, "DejaVu Sans (英語のみ)"

FONT_SUCCESS, FONT_NAME = setup_japanese_font()

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

# フォント状態を表示
if FONT_SUCCESS:
    st.sidebar.success(f"✅ 日本語表示: {FONT_NAME}")
else:
    st.sidebar.error(f"⚠️ 日本語フォント未検出: {FONT_NAME}")
    st.sidebar.info("📝 グラフは英語表示になります")

st.sidebar.markdown("---")

q1 = st.sidebar.slider(
    "Q1. 興味があるのはどっち？",
    min_value=-5.0, max_value=3.0, value=0.0, step=0.5,
    help="左：Web・システム開発 ／ 右：数学・データ分析"
)
st.sidebar.caption("Web・アプリ開発   ⇔   数学・データ分析")

st.sidebar.markdown("---")

q2 = st.sidebar.slider(
    "Q2. 学習スタイルの好みは？",
    min_value=-2.0, max_value=1.5, value=0.0, step=0.5,
    help="左：最新AI活用・実践 ／ 右：教科書・基礎理解"
)
st.sidebar.caption("生成AI・実践   ⇔   教科書・基礎")

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

    # グラフ作成
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # カラーパレット
    colors = ['#ADD8E6', '#FFA07A', '#90EE90', '#FFB6C1', '#DDA0DD']
    
    # 全クラスターをプロット
    for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
        cluster_data = df[df['Cluster'] == cluster_id]
        ax.scatter(
            cluster_data['Factor1_Score'], 
            cluster_data['Factor2_Score'],
            alpha=1.0, s=120, color=colors[i % len(colors)],
            label=f'Cluster {cluster_id}', edgecolors='gray', linewidths=0.5
        )

    # 推奨コース（赤色で強調）
    if not target_courses.empty:
        ax.scatter(
            target_courses['Factor1_Score'], 
            target_courses['Factor2_Score'],
            color='red', s=250, marker='o', 
            label='Recommended Courses' if not FONT_SUCCESS else '推奨コース',
            zorder=5, edgecolors='darkred', linewidths=2.5
        )

    # ユーザーの位置（金色の星）
    ax.scatter(
        user_vector[0], user_vector[1],
        color='gold', s=600, marker='*', edgecolor='black', linewidths=2.5,
        label='You' if not FONT_SUCCESS else 'あなた',
        zorder=10
    )

    # 軸と補助線
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # ラベルとタイトル（日本語フォントの状態に応じて切り替え）
    if FONT_SUCCESS:
        ax.set_xlabel("Web・アプリ開発   ⇔   数学・データ分析", fontsize=13, fontweight='bold')
        ax.set_ylabel("生成AI・実践   ⇔   教科書・基礎", fontsize=13, fontweight='bold')
        ax.set_title("あなたの立ち位置とおすすめコース", fontsize=15, fontweight='bold', pad=20)
    else:
        ax.set_xlabel("Web/System   ⇔   Theory/Math", fontsize=13, fontweight='bold')
        ax.set_ylabel("GenAI/Applied   ⇔   Basic/Textbook", fontsize=13, fontweight='bold')
        ax.set_title("Your Position & Recommended Courses", fontsize=15, fontweight='bold', pad=20)
    
    # 凡例
    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11, 
              framealpha=0.9, edgecolor='black')
    
    # グリッド
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    
    # 軸の範囲を設定
    ax.set_xlim(-5, 3)
    ax.set_ylim(-2, 1.5)
    
    # 余白調整
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close()

st.markdown("---")
st.caption("💡 スライダーを調整すると、リアルタイムでおすすめが変わります！")

# デバッグ情報（開発時のみ表示）
with st.expander("🔧 デバッグ情報"):
    st.write(f"**フォント適用状況:** {FONT_SUCCESS}")
    st.write(f"**使用フォント:** {FONT_NAME}")
    st.write(f"**利用可能なフォント数:** {len(fm.fontManager.ttflist)}")
    
    # 日本語フォントのリストを表示
    jp_fonts = [f.name for f in fm.fontManager.ttflist if any(
        keyword in f.name.lower() for keyword in ['gothic', 'mincho', 'jp', 'japanese', 'cjk', 'noto', 'ipa']
    )]
    if jp_fonts:
        unique_jp_fonts = list(set(jp_fonts))[:10]
        st.write(f"**検出された日本語フォント:** {', '.join(unique_jp_fonts)}")
    else:
        st.write("**検出された日本語フォント:** なし")