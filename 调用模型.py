import joblib
import jieba
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# 1. 定义中文分词器（必须和训练时一致）
def chinese_tokenizer(text):
    return jieba.lcut(text)

# 2. 自定义特征组合器（必须和训练时一致）
class TextCombiner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return (X['title'].fillna('') + ' ' + X['keywords'].fillna(''))

# 3. 分类 code -> 名称 对应表
code_to_name = {
    '100': 'news_story', '101': 'news_culture', '102': 'news_entertainment',
    '103': 'news_sports', '104': 'news_finance', '106': 'news_house',
    '107': 'news_car', '108': 'news_edu', '109': 'news_tech', '110': 'news_military',
    '112': 'news_travel', '113': 'news_world', '114': 'stock',
    '115': 'news_agriculture', '116': 'news_game'
}

# 4. 加载模型
try:
    model = joblib.load('news_classifier.pkl')
except Exception as e:
    print("模型加载失败，请确保定义了 TextCombiner 和 chinese_tokenizer。")
    print("错误详情：", e)
    exit()

# 5. 用户输入并预测
def predict_news(title, keywords):
    input_df = pd.DataFrame([{'title': title, 'keywords': keywords}])
    pred_code = model.predict(input_df)[0]
    category_name = code_to_name.get(pred_code, "未知类别")
    return pred_code, category_name

if __name__ == '__main__':
    title = input("请输入新闻标题：")
    keywords = input("请输入新闻关键词（可选，用空格分隔）：")
    code, name = predict_news(title, keywords)
    print(f"\n预测类别：{code} - {name}")
