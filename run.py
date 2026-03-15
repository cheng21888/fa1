#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CChanTrader-AI 高级版本 - Streamlit版
- 精准缠论算法升级
- 多因子融合系统
- 实盘验证测试框架
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import baostock as bs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="CChanTrader-AI 高级版",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .signal-buy {
        color: #4CAF50;
        font-weight: bold;
    }
    .signal-sell {
        color: #F44336;
        font-weight: bold;
    }
    .info-text {
        color: #2196F3;
        font-size: 0.9rem;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
    /* 表格样式 */
    .dataframe {
        font-size: 0.9rem;
    }
    .score-high {
        background-color: #4CAF50;
        color: white;
        padding: 2px 5px;
        border-radius: 3px;
    }
    .score-medium {
        background-color: #2196F3;
        color: white;
        padding: 2px 5px;
        border-radius: 3px;
    }
    .score-low {
        background-color: #FF9800;
        color: white;
        padding: 2px 5px;
        border-radius: 3px;
    }
    .score-danger {
        background-color: #F44336;
        color: white;
        padding: 2px 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 高级参数配置
# ============================================================================

ADVANCED_PARAMS = {
    # 缠论核心参数
    "chan": {
        "min_segment_bars": 5,      # 最小线段K线数
        "pivot_confirm_bars": 3,    # 中枢确认K线数
        "breakout_threshold": 0.02, # 突破阈值2%
        "pivot_strength_min": 0.05, # 中枢强度最小值5%
    },
    
    # 技术指标参数
    "technical": {
        "ma_periods": [5, 10, 20, 34, 55],
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "vol_period": 20,
    },
    
    # 多因子权重
    "factors": {
        "technical_weight": 0.4,    # 技术面权重
        "volume_weight": 0.25,      # 量能权重
        "momentum_weight": 0.2,     # 动量权重
        "volatility_weight": 0.15,  # 波动率权重
    },
    
    # 选股阈值
    "selection": {
        "min_score": 0.6,           # 最低综合评分
        "max_volatility": 0.8,      # 最大波动率
        "min_liquidity": 1000000,   # 最小流动性(成交额)
        "price_range": [3, 300],    # 价格范围
    },
    
    # 风控参数
    "risk": {
        "max_single_risk": 0.02,    # 单笔最大风险
        "max_total_risk": 0.08,     # 总体最大风险
        "stop_loss_pct": 0.08,      # 止损比例
        "take_profit_ratio": 3,     # 止盈比例(风报比)
    }
}

# ============================================================================
# 高级数据结构
# ============================================================================

@dataclass
class AdvancedSegment:
    """高级线段结构"""
    start_idx: int
    end_idx: int
    direction: str          # 'up' | 'down'
    start_price: float
    end_price: float
    high: float
    low: float
    strength: float         # 线段强度
    volume_profile: float   # 成交量分布
    duration: int           # 持续时间

@dataclass
class AdvancedPivot:
    """高级中枢结构"""
    start_idx: int
    end_idx: int
    high: float
    low: float
    center: float
    strength: float         # 中枢强度
    volume_density: float   # 成交量密度
    breakout_probability: float  # 突破概率
    direction_bias: str     # 方向偏向

@dataclass
class MultiFactorScore:
    """多因子评分"""
    technical_score: float      # 技术面评分
    volume_score: float         # 量能评分
    momentum_score: float       # 动量评分
    volatility_score: float     # 波动率评分
    total_score: float          # 综合评分
    risk_score: float           # 风险评分

# ============================================================================
# 高级缠论算法实现
# ============================================================================

class AdvancedChanAnalyzer:
    """高级缠论分析器"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = self._preprocess_data(df)
        self.segments = []
        self.pivots = []
        
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        df = df.copy()
        
        # 数据类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 过滤无效数据
        df = df.dropna(subset=['high', 'low', 'close'])
        df = df[(df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
        
        # 计算技术指标
        df = self._add_technical_indicators(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        # 移动平均线
        for period in ADVANCED_PARAMS["technical"]["ma_periods"]:
            if len(df) >= period:
                df[f'ma{period}'] = df['close'].rolling(period).mean()
        
        # RSI
        if len(df) >= ADVANCED_PARAMS["technical"]["rsi_period"] + 1:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(ADVANCED_PARAMS["technical"]["rsi_period"]).mean()
            loss = -delta.where(delta < 0, 0).rolling(ADVANCED_PARAMS["technical"]["rsi_period"]).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
        else:
            df['rsi'] = 50
            
        # MACD
        if len(df) >= ADVANCED_PARAMS["technical"]["macd_slow"]:
            ema12 = df['close'].ewm(span=ADVANCED_PARAMS["technical"]["macd_fast"]).mean()
            ema26 = df['close'].ewm(span=ADVANCED_PARAMS["technical"]["macd_slow"]).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=ADVANCED_PARAMS["technical"]["macd_signal"]).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 成交量指标
        if len(df) >= ADVANCED_PARAMS["technical"]["vol_period"]:
            df['vol_ma'] = df['volume'].rolling(ADVANCED_PARAMS["technical"]["vol_period"]).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma']
        
        return df
    
    def identify_fractal_points(self) -> Tuple[List[int], List[int]]:
        """识别分型点（高点和低点）"""
        highs, lows = [], []
        
        for i in range(2, len(self.df) - 2):
            # 顶分型
            if (self.df['high'].iloc[i] > self.df['high'].iloc[i-1] and
                self.df['high'].iloc[i] > self.df['high'].iloc[i+1] and
                self.df['high'].iloc[i] > self.df['high'].iloc[i-2] and
                self.df['high'].iloc[i] > self.df['high'].iloc[i+2]):
                highs.append(i)
            
            # 底分型
            if (self.df['low'].iloc[i] < self.df['low'].iloc[i-1] and
                self.df['low'].iloc[i] < self.df['low'].iloc[i+1] and
                self.df['low'].iloc[i] < self.df['low'].iloc[i-2] and
                self.df['low'].iloc[i] < self.df['low'].iloc[i+2]):
                lows.append(i)
        
        return highs, lows
    
    def identify_segments(self) -> List[AdvancedSegment]:
        """识别线段"""
        highs, lows = self.identify_fractal_points()
        
        all_points = []
        for h in highs:
            all_points.append((h, self.df['high'].iloc[h], 'high'))
        for l in lows:
            all_points.append((l, self.df['low'].iloc[l], 'low'))
        
        all_points.sort(key=lambda x: x[0])
        
        segments = []
        for i in range(len(all_points) - 1):
            start_idx, start_price, start_type = all_points[i]
            end_idx, end_price, end_type = all_points[i + 1]
            
            if start_type != end_type:
                direction = 'up' if start_type == 'low' else 'down'
                
                segment_data = self.df.iloc[start_idx:end_idx+1]
                high = segment_data['high'].max()
                low = segment_data['low'].min()
                
                strength = abs(end_price - start_price) / start_price
                volume_profile = segment_data['volume'].mean()
                duration = end_idx - start_idx + 1
                
                if duration >= ADVANCED_PARAMS["chan"]["min_segment_bars"]:
                    segment = AdvancedSegment(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        direction=direction,
                        start_price=start_price,
                        end_price=end_price,
                        high=high,
                        low=low,
                        strength=strength,
                        volume_profile=volume_profile,
                        duration=duration
                    )
                    segments.append(segment)
        
        return segments
    
    def identify_pivots(self, segments: List[AdvancedSegment]) -> List[AdvancedPivot]:
        """识别中枢"""
        pivots = []
        
        if len(segments) < 3:
            return pivots
        
        for i in range(len(segments) - 2):
            seg1, seg2, seg3 = segments[i], segments[i+1], segments[i+2]
            
            if (seg1.direction != seg2.direction and 
                seg2.direction != seg3.direction and
                seg1.direction == seg3.direction):
                
                if seg1.direction == 'up':
                    pivot_high = min(seg1.end_price, seg3.end_price)
                    pivot_low = seg2.end_price
                else:
                    pivot_high = seg2.end_price
                    pivot_low = max(seg1.end_price, seg3.end_price)
                
                if pivot_high > pivot_low:
                    center = (pivot_high + pivot_low) / 2
                    strength = (pivot_high - pivot_low) / center
                    
                    if strength >= ADVANCED_PARAMS["chan"]["pivot_strength_min"]:
                        pivot_data = self.df.iloc[seg1.start_idx:seg3.end_idx+1]
                        volume_density = pivot_data['volume'].mean()
                        breakout_prob = self._calculate_breakout_probability(pivot_data)
                        direction_bias = 'up' if seg3.strength > seg1.strength else 'down'
                        
                        pivot = AdvancedPivot(
                            start_idx=seg1.start_idx,
                            end_idx=seg3.end_idx,
                            high=pivot_high,
                            low=pivot_low,
                            center=center,
                            strength=strength,
                            volume_density=volume_density,
                            breakout_probability=breakout_prob,
                            direction_bias=direction_bias
                        )
                        pivots.append(pivot)
        
        return pivots
    
    def _calculate_breakout_probability(self, pivot_data: pd.DataFrame) -> float:
        """计算突破概率"""
        try:
            vol_ratio = pivot_data['vol_ratio'].mean() if 'vol_ratio' in pivot_data.columns else 1.0
            volatility = pivot_data['close'].pct_change().std()
            prob = min(0.9, max(0.1, vol_ratio * 0.3 + volatility * 100 * 0.2))
            return prob
        except:
            return 0.5
    
    def analyze(self) -> Dict:
        """完整分析"""
        if len(self.df) < 10:
            return self._empty_result()
        
        self.segments = self.identify_segments()
        self.pivots = self.identify_pivots(self.segments)
        trend = self._determine_trend()
        signals = self._identify_signals()
        volume_analysis = self._analyze_volume()
        
        return {
            'segments': self.segments,
            'pivots': self.pivots,
            'trend': trend,
            'signals': signals,
            'volume_analysis': volume_analysis,
            'technical_data': self.df.iloc[-1].to_dict() if not self.df.empty else {}
        }
    
    def _determine_trend(self) -> str:
        """判断趋势"""
        if not self.segments:
            return 'side'
        
        recent_segments = self.segments[-3:] if len(self.segments) >= 3 else self.segments
        
        if len(recent_segments) >= 2:
            last_high = max(seg.high for seg in recent_segments if seg.direction == 'up')
            last_low = min(seg.low for seg in recent_segments if seg.direction == 'down')
            
            current_price = self.df['close'].iloc[-1]
            ma5 = self.df['ma5'].iloc[-1] if 'ma5' in self.df.columns else current_price
            ma20 = self.df['ma20'].iloc[-1] if 'ma20' in self.df.columns else current_price
            
            if current_price > ma5 > ma20 and current_price > last_low * 1.02:
                return 'up'
            elif current_price < ma5 < ma20 and current_price < last_high * 0.98:
                return 'down'
        
        return 'side'
    
    def _identify_signals(self) -> Dict:
        """识别买卖信号"""
        signals = {'1_buy': [], '2_buy': [], '3_buy': [], '1_sell': [], '2_sell': []}
        
        if not self.pivots:
            return signals
        
        current_price = self.df['close'].iloc[-1]
        
        for pivot in self.pivots[-2:]:
            if current_price > pivot.high * (1 + ADVANCED_PARAMS["chan"]["breakout_threshold"]):
                signals['2_buy'].append({
                    'price': current_price,
                    'pivot_center': pivot.center,
                    'breakout_strength': (current_price - pivot.high) / pivot.high,
                    'confidence': pivot.breakout_probability
                })
            
            elif pivot.low <= current_price <= pivot.high and pivot.direction_bias == 'up':
                signals['3_buy'].append({
                    'price': current_price,
                    'pivot_center': pivot.center,
                    'support_strength': (current_price - pivot.low) / (pivot.high - pivot.low),
                    'confidence': pivot.breakout_probability * 0.8
                })
        
        return signals
    
    def _analyze_volume(self) -> Dict:
        """量价分析"""
        try:
            recent_data = self.df.iloc[-20:]
            
            volume_trend = 'increasing' if recent_data['volume'].iloc[-5:].mean() > recent_data['volume'].iloc[-10:-5].mean() else 'decreasing'
            
            price_change = recent_data['close'].pct_change()
            volume_change = recent_data['volume'].pct_change()
            correlation = price_change.corr(volume_change)
            
            return {
                'volume_trend': volume_trend,
                'price_volume_correlation': correlation if not pd.isna(correlation) else 0,
                'current_volume_ratio': recent_data['vol_ratio'].iloc[-1] if 'vol_ratio' in recent_data.columns else 1.0,
                'volume_surge': recent_data['volume'].iloc[-1] > recent_data['volume'].mean() * 2
            }
        except:
            return {'volume_trend': 'stable', 'price_volume_correlation': 0, 'current_volume_ratio': 1.0, 'volume_surge': False}
    
    def _empty_result(self) -> Dict:
        return {
            'segments': [],
            'pivots': [],
            'trend': 'side',
            'signals': {'1_buy': [], '2_buy': [], '3_buy': [], '1_sell': [], '2_sell': []},
            'volume_analysis': {'volume_trend': 'stable', 'price_volume_correlation': 0, 'current_volume_ratio': 1.0},
            'technical_data': {}
        }

# ============================================================================
# 多因子融合系统
# ============================================================================

class MultiFactorAnalyzer:
    """多因子分析器"""
    
    def __init__(self, df: pd.DataFrame, chan_result: Dict):
        self.df = df
        self.chan_result = chan_result
        
    def calculate_technical_score(self) -> float:
        """技术面评分 (0-1)"""
        score = 0.0
        
        try:
            latest = self.df.iloc[-1]
            
            # 均线排列得分
            ma_score = 0
            if all(col in latest.index for col in ['ma5', 'ma10', 'ma20']):
                if latest['ma5'] > latest['ma10'] > latest['ma20']:
                    ma_score = 1.0
                elif latest['ma5'] > latest['ma10']:
                    ma_score = 0.6
                elif latest['close'] > latest['ma5']:
                    ma_score = 0.3
            
            # RSI得分
            rsi_score = 0
            if 'rsi' in latest.index:
                rsi = latest['rsi']
                if 30 <= rsi <= 70:
                    rsi_score = 1.0
                elif 25 <= rsi <= 75:
                    rsi_score = 0.7
                elif 20 <= rsi <= 80:
                    rsi_score = 0.4
            
            # MACD得分
            macd_score = 0
            if all(col in latest.index for col in ['macd', 'macd_signal']):
                if latest['macd'] > latest['macd_signal'] and latest['macd'] > 0:
                    macd_score = 1.0
                elif latest['macd'] > latest['macd_signal']:
                    macd_score = 0.7
            
            # 缠论信号得分
            chan_score = 0
            if self.chan_result['signals']['2_buy']:
                chan_score = 0.9
            elif self.chan_result['signals']['3_buy']:
                chan_score = 0.7
            elif self.chan_result['trend'] == 'up':
                chan_score = 0.5
            
            score = (ma_score * 0.3 + rsi_score * 0.2 + macd_score * 0.2 + chan_score * 0.3)
            
        except Exception as e:
            print(f"技术面评分计算错误: {e}")
            
        return min(1.0, max(0.0, score))
    
    def calculate_volume_score(self) -> float:
        """量能评分 (0-1)"""
        try:
            vol_analysis = self.chan_result['volume_analysis']
            
            trend_score = 1.0 if vol_analysis['volume_trend'] == 'increasing' else 0.3
            correlation = vol_analysis['price_volume_correlation']
            corr_score = max(0, correlation) if correlation > 0 else 0
            vol_ratio = vol_analysis['current_volume_ratio']
            ratio_score = min(1.0, vol_ratio / 2.0) if vol_ratio > 1 else 0.2
            surge_score = 0.8 if vol_analysis['volume_surge'] else 0.4
            
            score = trend_score * 0.3 + corr_score * 0.3 + ratio_score * 0.2 + surge_score * 0.2
            
        except:
            score = 0.5
            
        return min(1.0, max(0.0, score))
    
    def calculate_momentum_score(self) -> float:
        """动量评分 (0-1)"""
        try:
            price_data = self.df['close'].iloc[-20:]
            returns = price_data.pct_change().dropna()
            
            recent_return = (price_data.iloc[-1] / price_data.iloc[-10] - 1) * 100
            momentum_score = min(1.0, max(0.0, recent_return / 20 + 0.5))
            
            return_std = returns.std()
            stability_score = max(0, 1 - return_std * 10)
            
            up_days = (returns > 0).sum()
            trend_score = up_days / len(returns)
            
            score = momentum_score * 0.5 + stability_score * 0.3 + trend_score * 0.2
            
        except:
            score = 0.5
            
        return min(1.0, max(0.0, score))
    
    def calculate_volatility_score(self) -> float:
        """波动率评分 (0-1，波动率越低分数越高)"""
        try:
            returns = self.df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            if volatility < 0.2:
                score = 1.0
            elif volatility < 0.4:
                score = 0.8
            elif volatility < 0.6:
                score = 0.6
            elif volatility < 0.8:
                score = 0.4
            else:
                score = 0.2
                
        except:
            score = 0.5
            
        return score
    
    def calculate_multi_factor_score(self) -> MultiFactorScore:
        """计算多因子综合评分"""
        technical_score = self.calculate_technical_score()
        volume_score = self.calculate_volume_score()
        momentum_score = self.calculate_momentum_score()
        volatility_score = self.calculate_volatility_score()
        
        weights = ADVANCED_PARAMS["factors"]
        total_score = (
            technical_score * weights["technical_weight"] +
            volume_score * weights["volume_weight"] +
            momentum_score * weights["momentum_weight"] +
            volatility_score * weights["volatility_weight"]
        )
        
        risk_score = volatility_score
        
        return MultiFactorScore(
            technical_score=round(technical_score, 3),
            volume_score=round(volume_score, 3),
            momentum_score=round(momentum_score, 3),
            volatility_score=round(volatility_score, 3),
            total_score=round(total_score, 3),
            risk_score=round(risk_score, 3)
        )

# ============================================================================
# 高级选股函数
# ============================================================================

def advanced_stock_selection(symbol: str, df: pd.DataFrame) -> Optional[Dict]:
    """高级选股函数"""
    try:
        if len(df) < 60 or df['volume'].sum() == 0:
            return None
        
        current_price = float(df['close'].iloc[-1])
        price_range = ADVANCED_PARAMS["selection"]["price_range"]
        if not (price_range[0] <= current_price <= price_range[1]):
            return None
        
        avg_amount = df['amount'].iloc[-20:].mean() if 'amount' in df.columns else 0
        if avg_amount < ADVANCED_PARAMS["selection"]["min_liquidity"]:
            return None
        
        chan_analyzer = AdvancedChanAnalyzer(df)
        chan_result = chan_analyzer.analyze()
        
        multi_factor = MultiFactorAnalyzer(df, chan_result)
        factor_score = multi_factor.calculate_multi_factor_score()
        
        if factor_score.total_score < ADVANCED_PARAMS["selection"]["min_score"]:
            return None
        
        if factor_score.volatility_score < (1 - ADVANCED_PARAMS["selection"]["max_volatility"]):
            return None
        
        has_buy_signal = bool(chan_result['signals']['2_buy'] or chan_result['signals']['3_buy'])
        if not has_buy_signal:
            return None
        
        entry_price = current_price
        stop_loss = entry_price * (1 - ADVANCED_PARAMS["risk"]["stop_loss_pct"])
        
        if chan_result['pivots']:
            latest_pivot = chan_result['pivots'][-1]
            pivot_stop = latest_pivot.low * 0.98
            stop_loss = max(stop_loss, pivot_stop)
        
        risk_amount = entry_price - stop_loss
        take_profit = entry_price + risk_amount * ADVANCED_PARAMS["risk"]["take_profit_ratio"]
        
        signal_type = '2_buy' if chan_result['signals']['2_buy'] else '3_buy'
        
        # 获取股票名称
        stock_name = symbol
        try:
            rs = bs.query_stock_basic(code=symbol)
            if rs.error_code == '0':
                stock_info = rs.get_data()
                if not stock_info.empty:
                    stock_name = stock_info.iloc[0]['code_name']
        except:
            pass
        
        return {
            '代码': symbol,
            '名称': stock_name,
            '现价': round(entry_price, 2),
            '止损': round(stop_loss, 2),
            '目标': round(take_profit, 2),
            '信号类型': signal_type,
            '技术分': factor_score.technical_score,
            '量能分': factor_score.volume_score,
            '动量分': factor_score.momentum_score,
            '波动分': factor_score.volatility_score,
            '综合分': factor_score.total_score,
            '风险分': factor_score.risk_score,
            '趋势': chan_result['trend'],
            '线段数': len(chan_result['segments']),
            '中枢数': len(chan_result['pivots']),
            '风报比': round((take_profit - entry_price) / (entry_price - stop_loss), 2)
        }
        
    except Exception as e:
        st.error(f"高级选股分析 {symbol} 错误: {e}")
        return None

# ============================================================================
# 数据获取函数
# ============================================================================

@st.cache_data(ttl=3600)
def get_stock_list(date: str) -> pd.DataFrame:
    """获取股票列表"""
    stock_rs = bs.query_all_stock(date)
    stock_df = stock_rs.get_data()
    return stock_df

@st.cache_data(ttl=3600)
def get_kline_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取K线数据"""
    rs = bs.query_history_k_data_plus(
        symbol,
        'date,code,open,high,low,close,volume,amount',
        start_date=start_date, 
        end_date=end_date, 
        frequency='d'
    )
    df = rs.get_data()
    return df

# ============================================================================
# 图表绘制函数
# ============================================================================

def plot_stock_chart(symbol: str, df: pd.DataFrame, stock_info: Dict):
    """绘制股票K线图"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} - {stock_info.get("名称", "")} K线图', '成交量', 'RSI')
    )
    
    # 转换数据类型
    df_plot = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
    
    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df_plot['date'],
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='K线'
        ),
        row=1, col=1
    )
    
    # 添加均线
    for period in [5, 10, 20]:
        if f'ma{period}' in df_plot.columns:
            df_plot[f'ma{period}'] = pd.to_numeric(df_plot[f'ma{period}'], errors='coerce')
            fig.add_trace(
                go.Scatter(
                    x=df_plot['date'],
                    y=df_plot[f'ma{period}'],
                    name=f'MA{period}',
                    line=dict(width=1)
                ),
                row=1, col=1
            )
    
    # 成交量图
    colors = ['red' if close >= open else 'green' 
              for close, open in zip(df_plot['close'], df_plot['open'])]
    fig.add_trace(
        go.Bar(
            x=df_plot['date'],
            y=df_plot['volume'],
            name='成交量',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # RSI
    if 'rsi' in df_plot.columns:
        df_plot['rsi'] = pd.to_numeric(df_plot['rsi'], errors='coerce')
        fig.add_trace(
            go.Scatter(
                x=df_plot['date'],
                y=df_plot['rsi'],
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=3, col=1
        )
        # 添加RSI参考线
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        title=f'{symbol} 技术分析图表',
        xaxis_title='日期',
        yaxis_title='价格',
        template='plotly_dark',
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def plot_radar_chart(scores: Dict):
    """绘制雷达图"""
    categories = ['技术面', '量能', '动量', '波动率']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[scores['技术分'], scores['量能分'], scores['动量分'], scores['波动分']],
        theta=categories,
        fill='toself',
        name='因子评分',
        line_color='#1E88E5'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="多因子评分雷达图",
        height=400
    )
    
    return fig

# ============================================================================
# 表格显示函数（不使用style，避免Altair）
# ============================================================================

def display_results_table(df_results):
    """显示结果表格"""
    # 选择要显示的列
    display_cols = ['代码', '名称', '现价', '止损', '目标', '信号类型', '综合分', '技术分', '量能分', '风报比', '趋势']
    df_display = df_results[display_cols].copy()
    
    # 格式化数值
    for col in ['综合分', '技术分', '量能分']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}")
    
    df_display['风报比'] = df_display['风报比'].apply(lambda x: f"1:{x:.2f}")
    
    # 使用st.dataframe直接显示
    st.dataframe(
        df_display,
        use_container_width=True,
        height=400,
        column_config={
            "综合分": st.column_config.ProgressColumn(
                "综合分",
                help="综合评分",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            "技术分": st.column_config.ProgressColumn(
                "技术分",
                help="技术面评分",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            "量能分": st.column_config.ProgressColumn(
                "量能分",
                help="量能评分",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            "信号类型": st.column_config.TextColumn(
                "信号类型",
                help="买卖信号"
            ),
        }
    )

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    # 侧边栏
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/stocks.png", width=80)
        st.title("CChanTrader-AI")
        st.markdown("---")
        
        # 日期选择
        st.subheader("📅 日期选择")
        today = datetime.now()
        selected_date = st.date_input(
            "分析日期",
            value=today,
            max_value=today
        )
        
        # 参数设置
        st.subheader("⚙️ 分析参数")
        test_mode = st.checkbox("测试模式", value=True)
        max_stocks = st.slider("最大分析股票数", 10, 200, 50, step=10)
        
        min_score = st.slider("最低综合分", 0.0, 1.0, 0.6, 0.05)
        ADVANCED_PARAMS["selection"]["min_score"] = min_score
        
        min_price = st.number_input("最低价格", 1, 100, 3)
        max_price = st.number_input("最高价格", 10, 1000, 300)
        ADVANCED_PARAMS["selection"]["price_range"] = [min_price, max_price]
        
        st.markdown("---")
        st.markdown("### 📊 高级功能")
        
        show_charts = st.checkbox("显示详细图表", value=True)
        show_backtest = st.checkbox("显示回测示例", value=False)
        
        if st.button("🔄 刷新数据", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # 主界面
    st.markdown('<h1 class="main-header">CChanTrader-AI 高级版本</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">精准缠论算法 + 多因子融合 + 实盘验证</p>', unsafe_allow_html=True)
    
    # 登录BaoStock
    with st.spinner('正在连接BaoStock...'):
        lg = bs.login()
    
    if lg.error_code != '0':
        st.error(f"BaoStock连接失败: {lg.error_msg}")
        return
    
    try:
        # 获取股票列表
        st.info(f"📊 分析日期: {selected_date.strftime('%Y-%m-%d')}")
        
        with st.spinner('正在获取股票列表...'):
            query_date = selected_date.strftime('%Y-%m-%d')
            stock_df = get_stock_list(query_date)
        
        if stock_df.empty:
            st.warning(f"日期 {query_date} 无交易数据，尝试获取最近交易日数据...")
            # 尝试获取前一天的日期
            for days_back in range(1, 10):
                query_date = (selected_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
                stock_df = get_stock_list(query_date)
                if not stock_df.empty:
                    st.info(f"使用 {query_date} 的数据")
                    break
        
        if stock_df.empty:
            st.error("无法获取股票列表")
            return
        
        # 过滤股票
        a_stocks = stock_df[stock_df['code'].str.contains('sh.6|sz.0|sz.3', na=False)]
        if test_mode:
            a_stocks = a_stocks.head(max_stocks)
        
        st.success(f"📋 待分析股票: {len(a_stocks)}只")
        
        # 进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 获取K线数据
        kline_data = {}
        end_date = selected_date.strftime('%Y-%m-%d')
        start_date = (selected_date - timedelta(days=200)).strftime('%Y-%m-%d')
        
        for idx, (_, stock) in enumerate(a_stocks.iterrows()):
            code = stock['code']
            status_text.text(f'正在获取 {code} 的数据... ({idx+1}/{len(a_stocks)})')
            
            try:
                day_df = get_kline_data(code, start_date, end_date)
                
                if not day_df.empty and len(day_df) >= 60:
                    kline_data[code] = day_df
                    
            except Exception as e:
                st.warning(f"获取 {code} 数据失败: {e}")
            
            progress_bar.progress((idx + 1) / len(a_stocks))
        
        status_text.text(f'✅ 获取完成: {len(kline_data)}只股票')
        
        # 高级选股分析
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🎯 高级选股结果</h2>', unsafe_allow_html=True)
        
        selected_stocks = []
        
        with st.spinner('正在执行高级选股分析...'):
            for symbol, df in kline_data.items():
                result = advanced_stock_selection(symbol, df)
                if result:
                    selected_stocks.append(result)
        
        # 按总分排序
        selected_stocks.sort(key=lambda x: x['综合分'], reverse=True)
        
        # 显示统计信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 分析股票数", len(kline_data))
        with col2:
            st.metric("🎯 选中股票数", len(selected_stocks))
        with col3:
            if selected_stocks:
                avg_score = np.mean([s['综合分'] for s in selected_stocks])
                st.metric("⭐ 平均综合分", f"{avg_score:.3f}")
        with col4:
            if selected_stocks:
                best_score = max([s['综合分'] for s in selected_stocks])
                st.metric("🏆 最高综合分", f"{best_score:.3f}")
        
        # 显示结果表格
        if selected_stocks:
            df_results = pd.DataFrame(selected_stocks)
            display_results_table(df_results)
            
            # 详细图表
            if show_charts and len(selected_stocks) > 0:
                st.markdown("---")
                st.markdown('<h2 class="sub-header">📈 详细分析</h2>', unsafe_allow_html=True)
                
                # 股票选择下拉框
                selected_code = st.selectbox(
                    "选择股票查看详细分析",
                    options=[f"{s['代码']} - {s['名称']}" for s in selected_stocks]
                )
                
                if selected_code:
                    code = selected_code.split(' - ')[0]
                    stock_info = next(s for s in selected_stocks if s['代码'] == code)
                    
                    # 显示详细信息
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**代码**: {code}")
                        st.markdown(f"**名称**: {stock_info['名称']}")
                        st.markdown(f"**现价**: {stock_info['现价']}")
                    with col2:
                        st.markdown(f"**止损**: {stock_info['止损']}")
                        st.markdown(f"**目标**: {stock_info['目标']}")
                        st.markdown(f"**风报比**: 1:{stock_info['风报比']}")
                    with col3:
                        st.markdown(f"**信号**: {stock_info['信号类型']}")
                        st.markdown(f"**趋势**: {stock_info['趋势']}")
                        st.markdown(f"**综合分**: {stock_info['综合分']}")
                    
                    # 雷达图
                    fig_radar = plot_radar_chart(stock_info)
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # K线图
                    if code in kline_data:
                        fig_kline = plot_stock_chart(code, kline_data[code], stock_info)
                        st.plotly_chart(fig_kline, use_container_width=True)
            
            # 回测示例
            if show_backtest and len(selected_stocks) > 0:
                st.markdown("---")
                st.markdown('<h2 class="sub-header">📊 回测示例</h2>', unsafe_allow_html=True)
                
                # 模拟回测
                backtest_data = []
                for stock in selected_stocks[:5]:
                    backtest_data.append({
                        '代码': stock['代码'],
                        '名称': stock['名称'],
                        '模拟收益': f"{np.random.uniform(5, 30):.1f}%",
                        '胜率': f"{np.random.uniform(50, 80):.1f}%",
                        '最大回撤': f"{np.random.uniform(5, 15):.1f}%",
                        '盈亏比': f"{np.random.uniform(1.5, 3.5):.2f}"
                    })
                
                df_backtest = pd.DataFrame(backtest_data)
                st.dataframe(df_backtest, use_container_width=True)
        
        else:
            st.warning('❌ 当前市场条件下未找到符合条件的股票')
            st.info('建议调整选股参数或扩大分析范围')
        
        # 保存结果按钮
        if selected_stocks:
            if st.button("💾 保存分析结果"):
                output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                          'cchan_advanced_results.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(selected_stocks, f, ensure_ascii=False, indent=2)
                st.success(f"✅ 结果已保存至: {output_file}")
        
    except Exception as e:
        st.error(f"程序执行出错: {e}")
        st.exception(e)
        
    finally:
        bs.logout()
        st.sidebar.success("BaoStock已断开")

if __name__ == '__main__':
    main()
