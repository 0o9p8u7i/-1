import pandas as pd
import jieba
jieba.setLogLevel(jieba.logging.WARN)
import joblib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score

# ✅ 分类 code 到 名称映射
code_to_name = {
    '100': '民生', '101': '文化', '102': '娱乐', '103': '体育', '104': '财经',
    '106': '房产', '107': '汽车', '108': '教育', '109': '科技', '110': '军事',
    '112': '旅游', '113': '国际', '114': '证券', '115': '农业', '116': '电竞'
}

# ✅ 中文分词函数
def chinese_tokenizer(text):
    return jieba.lcut(text)

# ✅ 自定义文本组合类
class TextCombiner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return (X['title'].fillna('') + ' ' + X['keywords'].fillna(''))

# ✅ 读取文本数据
def deal_with_txt(filepath):
    data = pd.read_csv(filepath, sep='_!_', names=['news_id', 'category_code', 'category_name', 'title', 'keywords'], engine='python')
    data = data.dropna()
    X = data[['title', 'keywords']]
    Y = data['category_code'].astype(str)
    return X, Y

# ✅ 构建 Pipeline
def build_pipeline():
    return Pipeline([
        ('combine', TextCombiner()),
        ('tfidf', TfidfVectorizer(tokenizer=chinese_tokenizer, max_features=5000)),
        ('clf', MultinomialNB())
    ])

# ✅ 中文可视化
def plot_accuracy_per_class(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    codes = [k for k in report if k.isdigit()]
    scores = [report[k]['f1-score'] for k in codes]
    labels = [code_to_name.get(k, k) for k in codes]

    # 设置中文字体（适用于 Windows）
    font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12)  # 黑体
    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, scores, color='skyblue')

    # 添加分数标签
    for bar, score in zip(bars, scores):
        plt.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.2f}', va='center', fontproperties=font)

    plt.xlabel('F1 分数', fontproperties=font)
    plt.title('各类别F1评分', fontproperties=font)
    plt.tight_layout()
    plt.savefig('category_f1_scores.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    filepath = 'dataset.txt'
    X, Y = deal_with_txt(filepath)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    model = build_pipeline()
    model.fit(X_train, y_train)

    joblib.dump(model, 'news_classifier.pkl')
    print("✅ 模型已保存为 news_classifier.pkl")

    y_pred = model.predict(X_test)
    print("✅ 准确率：", accuracy_score(y_test, y_pred))
    print("✅ 分类报告：\n", classification_report(y_test, y_pred, zero_division=0))

    # 可视化
    plot_accuracy_per_class(y_test, y_pred)