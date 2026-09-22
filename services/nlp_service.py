"""
NLP Service - text processing, sentiment analysis, keyword extraction,
text classification, entity extraction, and readability analysis.
"""
import re
import string
import math
from typing import Dict, List
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.helpers import format_response


class NLPService:
    """Natural language processing utilities."""

    CHINESE_STOP_WORDS = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
        "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
        "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
        "们", "那", "什么", "怎么", "如何", "因为", "所以", "但是", "而且",
        "可以", "这个", "那个", "我们", "他们", "它们", "已经", "还是",
        "或者", "如果", "虽然", "然后", "这样", "那样", "之", "与",
        "中", "为", "以", "而", "及", "等", "被", "把", "让", "从",
        "对", "向", "在", "于", "给", "将", "并", "所", "其", "该",
        "每", "各", "某", "哪", "谁", "几", "多", "少", "太", "更",
        "最", "才", "再", "又", "还", "就", "便", "刚", "已",
    }

    # Extended sentiment dictionaries for Chinese
    POSITIVE_WORDS = {
        "好", "棒", "优秀", "喜欢", "爱", "赞", "美好", "开心", "快乐",
        "满意", "感谢", "漂亮", "厉害", "舒服", "方便", "强大", "出色",
        "完美", "惊喜", "推荐", "进步", "成功", "希望", "信心", "支持",
        "幸福", "美好", "精彩", "温暖", "辉煌", "卓越", "杰出", "优异",
        "很棒", "真好", "太棒了", "了不起", "超赞", "赞不绝口", "好评",
        "不错", "大方", "优雅", "精致", "舒适", "便利", "高效", "稳定",
        "值得", "满意", "正能量", "阳光", "甜蜜", "可爱", "迷人", "聪慧",
        "热情", "周到", "贴心", "用心", "实惠", "划算", "健康",
        "good", "great", "excellent", "love", "wonderful", "amazing",
        "happy", "beautiful", "perfect", "best", "awesome", "fantastic",
        "outstanding", "superb", "brilliant", "delightful", "impressive",
    }

    NEGATIVE_WORDS = {
        "差", "坏", "糟糕", "讨厌", "恶心", "垃圾", "烂", "失望",
        "难过", "伤心", "生气", "愤怒", "抱怨", "失败", "痛苦",
        "无聊", "崩溃", "后悔", "差劲", "一般", "不好", "太差",
        "很差", "极差", "忽悠", "诈骗", "坑爹", "无语", "烦躁",
        "焦虑", "抑郁", "悲哀", "惨淡", "混乱", "麻烦", "糟糕透顶",
        "不值", "浪费", "失望至极", "令人失望", "不堪入目",
        "bad", "terrible", "awful", "hate", "worst", "horrible",
        "ugly", "poor", "sad", "angry", "disappointing", "dreadful",
        "pathetic", "useless", "disgusting", "frustrating",
    }

    # Text classification labels
    CLASSIFICATION_LABELS = [
        {"label": "科技/编程", "keywords": {"代码", "编程", "程序", "软件", "算法", "人工智能",
         "AI", "机器学习", "数据库", "服务器", "API", "Python", "Java", "JavaScript", "开发",
         "开源", "GitHub", "云计算", "算法", "架构", "框架", "API", "bug", "调试"}},
        {"label": "商业/财经", "keywords": {"市场", "投资", "股票", "经济", "企业", "销售",
         "利润", "收入", "金融", "商业", "管理", "营销", "创业", "公司", "财报", "融资",
         "商业模式", "股权", "上市", "并购"}},
        {"label": "教育/学术", "keywords": {"学习", "课程", "教育", "大学", "学校", "考试",
         "研究", "论文", "教授", "知识", "理论", "学术", "硕士", "博士", "学位", "图书馆",
         "教材", "教学", "实习", "毕业"}},
        {"label": "生活/娱乐", "keywords": {"电影", "音乐", "游戏", "美食", "旅游", "运动",
         "健康", "购物", "时尚", "综艺", "宠物", "健身", "摄影", "烹饪", "美容", "家居",
         "电影", "电视剧", "动漫", "旅游"}},
        {"label": "新闻/时事", "keywords": {"国家", "政府", "政策", "法规", "社会", "国际",
         "事件", "报道", "记者", "新闻", "发布会", "外交", "军事", "选举", "改革"}},
        {"label": "科学/技术", "keywords": {"科学", "研究", "实验", "物理", "化学", "生物",
         "医学", "基因", "量子", "航天", "宇宙", "数学", "统计", "生态", "环境"}},
    ]

    # Chinese entity patterns
    ENTITY_PATTERNS = [
        ("网址", re.compile(r'https?://[^\s]+', re.IGNORECASE)),
        ("邮箱", re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')),
        ("电话", re.compile(r'1[3-9]\d{9}')),
        ("日期", re.compile(r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?')),
        ("金额", re.compile(r'(\$|¥|€|£)\d+([,.]\d+)?|(\d+([,.]\d+)?)\s*(元|美元|欧元|英镑|港币|日元)')),
        ("公司", re.compile(r'(北京|上海|深圳|广州|杭州|成都|武汉|南京|天津|重庆)([^\s]{2,20})?(公司|集团|有限|科技|实业|股份)')),
        ("职务称谓", re.compile(r'(\w{2,4})(先生|女士|教授|博士|工程师|经理|总监|总裁)')),
    ]

    def __init__(self, language: str = "chinese"):
        self._language = language

    # ----------------------------------------------------------------
    def sentiment_analysis(self, text: str) -> Dict:
        """Enhanced sentiment analysis with expanded word lists."""
        if not text or not text.strip():
            return format_response(success=False, error="Cannot analyze empty text")

        tokens = self._tokenize(text.lower())
        pos_count = sum(1 for t in tokens if t in self.POSITIVE_WORDS)
        neg_count = sum(1 for t in tokens if t in self.NEGATIVE_WORDS)

        total = pos_count + neg_count or 1
        score = (pos_count - neg_count) / total

        # Also calculate text-level intensity
        intensity = "低"
        abs_score = abs(score)
        if abs_score > 0.6:
            intensity = "高"
        elif abs_score > 0.2:
            intensity = "中"

        if score > 0.15:
            label = "正面"
        elif score < -0.15:
            label = "负面"
        else:
            label = "中性"

        # Highlight positive/negative tokens found
        pos_found = [t for t in tokens if t in self.POSITIVE_WORDS][:10]
        neg_found = [t for t in tokens if t in self.NEGATIVE_WORDS][:10]

        return format_response(success=True, data={
            "情感": label,
            "得分": round(score, 4),
            "强度": intensity,
            "正面词数": pos_count,
            "负面词数": neg_count,
            "正面词": list(set(pos_found)),
            "负面词": list(set(neg_found)),
            "文本长度": len(text),
        })

    # ----------------------------------------------------------------
    def keyword_extraction(self, text: str, top_k: int = 10) -> Dict:
        """Enhanced keyword extraction using TF-IDF and frequency weighting."""
        if not text or not text.strip():
            return format_response(success=False, error="Cannot extract keywords from empty text")

        tokens = self._tokenize(text.lower())
        filtered = [t for t in tokens if t not in self.CHINESE_STOP_WORDS
                    and len(t) > 1 and t not in string.punctuation]

        freq = Counter(filtered)
        sentences = [s for s in re.split(r'[。！？.!?\n]', text) if len(s.strip()) > 3]

        # Frequency-based keywords with normalized score
        max_f = max(freq.values()) if freq else 1
        keywords_freq = [{"词": w, "出现次数": c, "分数": round(c / max_f, 4), "方法": "词频"}
                         for w, c in freq.most_common(top_k)]

        # TF-IDF keywords for multi-sentence text
        keywords_tfidf = []
        if len(sentences) > 1:
            try:
                vectorizer = TfidfVectorizer(
                    max_features=top_k,
                    token_pattern=r"(?u)\b\w+\b",
                )
                tfidf_matrix = vectorizer.fit_transform(sentences)
                scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
                terms = vectorizer.get_feature_names_out()
                ranks = np.argsort(scores)[::-1][:top_k]
                keywords_tfidf = [
                    {"词": terms[i], "分数": round(float(scores[i]), 4), "方法": "TF-IDF"}
                    for i in ranks if scores[i] > 0
                ]
            except Exception:
                pass

        return format_response(success=True, data={
            "词频结果": keywords_freq,
            "TFIDF结果": keywords_tfidf or keywords_freq,
            "总词数": len(tokens),
            "去重词数": len(freq),
            "关键词": keywords_freq[0]["词"] if keywords_freq else None,
        })

    # ----------------------------------------------------------------
    def text_summary(self, text: str, max_sentences: int = 3) -> Dict:
        """Extractive text summarization with improved sentence scoring."""
        if not text or not text.strip():
            return format_response(success=False, error="Cannot summarize empty text")

        sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if not sentences:
            return format_response(success=False, error="No meaningful sentences found")

        if len(sentences) <= max_sentences:
            return format_response(success=True, data={
                "摘要": text,
                "原句数": len(sentences),
                "摘要句数": len(sentences),
                "压缩比": 1.0,
            })

        tokens = self._tokenize(text.lower())
        filtered_tokens = [t for t in tokens if t not in self.CHINESE_STOP_WORDS and len(t) > 1]
        word_freq = Counter(filtered_tokens)
        max_freq = max(word_freq.values()) if word_freq else 1

        scored = []
        for i, s in enumerate(sentences):
            stokens = self._tokenize(s.lower())
            # TF score
            tf_score = sum(word_freq.get(t, 0) / (max_freq * len(stokens))
                          for t in stokens if t not in self.CHINESE_STOP_WORDS)
            # Position bonus: prefer earlier sentences
            position_bonus = max(0, 1 - i / len(sentences))
            score = tf_score * 0.7 + position_bonus * 0.3
            scored.append((score, i, s))

        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[:max_sentences]
        top.sort(key=lambda x: x[1])
        summary = "".join(s[2] for s in top)
        chars_original = sum(len(s) for s in sentences)
        chars_summary = sum(len(s[2]) for s in top)

        return format_response(success=True, data={
            "摘要": summary,
            "原句数": len(sentences),
            "摘要句数": len(top),
            "压缩比": round(chars_summary / max(chars_original, 1), 3),
        })

    # ----------------------------------------------------------------
    def word_count(self, text: str) -> Dict:
        """Detailed text statistics."""
        if not text:
            return format_response(success=True, data={
                "characters": 0, "words": 0, "sentences": 0,
            })

        chars = len(text)
        chars_no_space = len(text.replace(" ", "").replace("\n", "").replace("\r", ""))
        tokens = self._tokenize(text)
        lines = text.count("\n") + 1 if text else 0
        sentences_count = max(len(re.split(r'[。！？.!?\n]', text)) - 1, 1)
        paragraphs = max(len(re.split(r'\n\s*\n', text)), 1)
        avg_word_len = round(sum(len(t) for t in tokens) / max(len(tokens), 1), 1)
        avg_sentence_len = round(len(tokens) / max(sentences_count, 1), 1)
        unique_words = len(set(tokens))
        lexical_diversity = round(unique_words / max(len(tokens), 1), 4)

        return format_response(success=True, data={
            "总字符": chars,
            "去空格字符": chars_no_space,
            "字符数": chars,
            "词数": len(tokens),
            "句数": sentences_count,
            "段落数": paragraphs,
            "行数": lines,
            "去重词数": unique_words,
            "平均词长": avg_word_len,
            "平均句长": avg_sentence_len,
            "词汇丰富度": lexical_diversity,
        })

    # ----------------------------------------------------------------
    def classify_text(self, text: str) -> Dict:
        """Classify text into categories based on keyword matching."""
        if not text or not text.strip():
            return format_response(success=False, error="Cannot classify empty text")

        tokens = set(self._tokenize(text.lower()))
        results = []
        for category in self.CLASSIFICATION_LABELS:
            matches = tokens & category["keywords"]
            match_count = len(matches)
            if match_count > 0:
                score = round(match_count / max(len(tokens), 1), 4)
                results.append({
                    "分类": category["label"],
                    "匹配度": score,
                    "命中词": sorted(list(matches))[:8],
                })

        results.sort(key=lambda x: -x["score"])
        top_label = results[0]["label"] if results else "未分类"

        return format_response(success=True, data={
            "主分类": top_label,
            "分类结果": results[:5],
            "总词数": len(tokens),
        })

    # ----------------------------------------------------------------
    def extract_entities(self, text: str) -> Dict:
        """Extract basic named entities from text using regex patterns."""
        if not text or not text.strip():
            return format_response(success=False, error="Cannot extract from empty text")

        entities = {}
        for entity_type, pattern in self.ENTITY_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                # Flatten tuple matches
                flat = [m if isinstance(m, str) else "".join(m) for m in matches]
                entities[entity_type] = list(set(flat))

        # Also extract hashtag-like patterns (common in Chinese social media)
        hashtags = re.findall(r'#([^#\s]+)', text)
        if hashtags:
            entities["话题"] = list(set(hashtags))

        total_found = sum(len(v) for v in entities.values())

        return format_response(success=True, data={
            "实体": entities,
            "命中数": total_found,
            "类型": list(entities.keys()),
        })

    # ----------------------------------------------------------------
    def readability_score(self, text: str) -> Dict:
        """Calculate readability metrics for the text."""
        if not text or not text.strip():
            return format_response(success=False, error="Cannot analyze empty text")

        wc = self.word_count(text)
        tokens = self._tokenize(text)
        sentences_count = wc.get("data", {}).get("句数", 1)

        # Average words per sentence
        avg_words = len(tokens) / max(sentences_count, 1)

        # Count complex words (long words > 4 chars for Chinese, > 8 for English)
        complex_words = sum(1 for t in tokens if len(t) > 4)
        complex_ratio = complex_words / max(len(tokens), 1)

        # Simple readability level estimate
        if avg_words < 10:
            level = "简单"
        elif avg_words < 20:
            level = "中等"
        else:
            level = "复杂"

        # Estimate reading time (Chinese: ~400 chars/min, English: ~200 words/min)
        reading_time_chars = round(len(text) / 400, 1)
        reading_time_words = round(len(tokens) / 200, 1)

        return format_response(success=True, data={
            "难度": level,
            "平均句长": round(avg_words, 1),
            "复杂词占比": round(complex_ratio, 3),
            "句数": sentences_count,
            "词数": len(tokens),
            "字符数": len(text),
            "阅读分钟": max(reading_time_chars, reading_time_words),
            "阅读时间": f"约{max(reading_time_chars, reading_time_words, 0.1):.0f}分钟",
        })

    # ----------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Mixed Chinese/English tokenizer with bigram support."""
        tokens = []
        # English words
        for word in re.findall(r"[a-zA-Z0-9]+", text):
            tokens.append(word.lower())
        # Chinese characters and bigrams
        chinese_chars = re.findall(r"[\u4e00-\u9fff]+", text)
        for chunk in chinese_chars:
            for ch in chunk:
                tokens.append(ch)
            if len(chunk) >= 2:
                tokens.extend([chunk[i:i + 2] for i in range(len(chunk) - 1)])
        return tokens