#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CChanTrader-AI Streamlit应用
- 精准缠论算法升级
- 多因子融合系统
- 支持日期选择
- 表格形式展示
"""

import sys
import subprocess

# 检查并安装必要的包
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 确保安装正确版本的包
required_packages = [
    "streamlit==1.28.0",
    "pandas==2.0.3",
    "numpy==1.24.3",
    "plotly==5.17.0",
    "baostock==0.8.9",
    "python-dotenv==1.0.0",
    "altair==5.0.1",
    "protobuf==3.20.3"
]

for package in required_packages:
    try:
        __import__(package.split('==')[0])
    except ImportError:
        install_package(package)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import baostock as bs
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import os
import json
from typing import Optional, Dict, List
from dataclasses import dataclass
from tqdm import tqdm

# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="CChanTrader-AI 高级版本",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0F67B3;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .signal-buy {
        color: #00C853;
        font-weight: bold;
    }
    .signal-sell {
        color: #D32F2F;
        font-weight: bold;
    }
    .info-text {
        color: #666;
        font-size: 0.9rem;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
    .buy-2 {
        background-color: #00C85320;
        padding: 2px 5px;
        border-radius: 3px;
        color: #00C853;
        font-weight: bold;
    }
    .buy-3 {
        background-color: #FFA00020;
        padding: 2px 5px;
        border-radius: 3px;
        color: #FFA000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 高级参数配置
# ============================================================================

ADVANCED_PARAMS = {
    "chan": {
        "min_segment_bars": 5,
        "pivot_confirm_bars": 3,
        "breakout_threshold": 0.02,
        "pivot_strength_min": 0.05,
    },
    "technical": {
        "ma_periods": [5, 10, 20, 34, 55],
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "vol_period": 20,
    },
    "factors": {
        "technical_weight": 0.4,
        "volume_weight": 0.25,
        "momentum_weight": 0.2,
        "volatility_weight": 0.15,
    },
    "selection": {
        "min_score": 0.6,
        "max_volatility": 0.8,
        "min_liquidity": 1000000,
        "price_range": [3, 300],
    },
    "risk": {
        "max_single_risk": 0.02,
        "max_total_risk": 0.08,
        "stop_loss_pct": 0.08,
        "take_profit_ratio": 3,
    }
}

# ============================================================================
# 高级数据结构
# ============================================================================

@dataclass
class AdvancedSegment:
    start_idx: int
    end_idx: int
    direction: str
    start_price: float
    end_price: float
    high: float
    low: float
    strength: float
    volume_profile: float
    duration: int

@dataclass
class AdvancedPivot:
    start_idx: int
    end_idx: int
    high: float
    low: float
    center: float
    strength: float
    volume_density: float
    breakout_probability: float
    direction_bias: str

@dataclass
class MultiFactorScore:
    technical_score: float
    volume_score: float
    momentum_score: float
    volatility_score: float
    total_score: float
    risk_score: float

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
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['high', 'low', 'close'])
        df = df[(df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
        df = self._add_technical_indicators(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        for period in ADVANCED_PARAMS["technical"]["ma_periods"]:
            if len(df) >= period:
                df[f'ma{period}'] = df['close'].rolling(period).mean()
        
        if len(df) >= ADVANCED_PARAMS["technical"]["rsi_period"] + 1:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(ADVANCED_PARAMS["technical"]["rsi_period"]).mean()
            loss = -delta.where(delta < 0, 0).rolling(ADVANCED_PARAMS["technical"]["rsi_period"]).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
        else:
            df['rsi'] = 50
            
        if len(df) >= ADVANCED_PARAMS["technical"]["macd_slow"]:
            ema12 = df['close'].ewm(span=ADVANCED_PARAMS["technical"]["macd_fast"]).mean()
            ema26 = df['close'].ewm(span=ADVANCED_PARAMS["technical"]["macd_slow"]).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=ADVANCED_PARAMS["technical"]["macd_signal"]).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        if len(df) >= ADVANCED_PARAMS["technical"]["vol_period"]:
            df['vol_ma'] = df['volume'].rolling(ADVANCED_PARAMS["technical"]["vol_period"]).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma']
        
        return df
    
    def identify_fractal_points(self) -> tuple[list[int], list[int]]:
        """识别分型点"""
        highs, lows = [], []
        
        for i in range(2, len(self.df) - 2):
            if (self.df['high'].iloc[i] > self.df['high'].iloc[i-1] and
                self.df['high'].iloc[i] > self.df['high'].iloc[i+1] and
                self.df['high'].iloc[i] > self.df['high'].iloc[i-2] and
                self.df['high'].iloc[i] > self.df['high'].iloc[i+2]):
                highs.append(i)
            
            if (self.df['low'].iloc[i] < self.df['low'].iloc[i-1] and
                self.df['low'].iloc[i] < self.df['low'].iloc[i+1] and
                self.df['low'].iloc[i] < self.df['low'].iloc[i-2] and
                self.df['low'].iloc[i] < self.df['low'].iloc[i+2]):
                lows.append(i)
        
        return highs, lows
    
    def identify_segments(self) -> list[AdvancedSegment]:
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
    
    def identify_pivots(self, segments: list[AdvancedSegment]) -> list[AdvancedPivot]:
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
        """空结果"""
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
        """技术面评分"""
        score = 0.0
        
        try:
            latest = self.df.iloc[-1]
            
            ma_score = 0
            if all(col in latest.index for col in ['ma5', 'ma10', 'ma20']):
                if latest['ma5'] > latest['ma10'] > latest['ma20']:
                    ma_score = 1.0
                elif latest['ma5'] > latest['ma10']:
                    ma_score = 0.6
                elif latest['close'] > latest['ma5']:
                    ma_score = 0.3
            
            rsi_score = 0
            if 'rsi' in latest.index:
                rsi = latest['rsi']
                if 30 <= rsi <= 70:
                    rsi_score = 1.0
                elif 25 <= rsi <= 75:
                    rsi_score = 0.7
                elif 20 <= rsi <= 80:
                    rsi_score = 0.4
            
            macd_score = 0
            if all(col in latest.index for col in ['macd', 'macd_signal']):
                if latest['macd'] > latest['macd_signal'] and latest['macd'] > 0:
                    macd_score = 1.0
                elif latest['macd'] > latest['macd_signal']:
                    macd_score = 0.7
            
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
        """量能评分"""
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
        """动量评分"""
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
        """波动率评分"""
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

def get_stock_name(symbol: str) -> str:
    """获取股票名称（简化版）"""
    name_map = {
        'sh.600000': '浦发银行',
        'sh.600036': '招商银行',
        'sh.600519': '贵州茅台',
        'sh.600900': '长江电力',
        'sh.601318': '中国平安',
        'sh.601166': '兴业银行',
        'sz.000001': '平安银行',
        'sz.000858': '五粮液',
        'sz.000333': '美的集团',
        'sz.002415': '海康威视',
        'sz.300750': '宁德时代',
        'sz.300059': '东方财富',
    }
    # 如果不在映射表中，返回代码的后6位
    return name_map.get(symbol, symbol.split('.')[-1])

def advanced_stock_selection(symbol: str, df: pd.DataFrame) -> Optional[Dict]:
    """高级选股函数"""
    try:
        if len(df) < 60 or df['volume'].sum() == 0:
            return None
        
        # 转换数据类型
        for col in ['close', 'volume', 'amount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        current_price = float(df['close'].iloc[-1])
        price_range = ADVANCED_PARAMS["selection"]["price_range"]
        if not (price_range[0] <= current_price <= price_range[1]):
            return None
        
        avg_amount = pd.to_numeric(df['amount'].iloc[-20:], errors='coerce').mean() if 'amount' in df.columns else 0
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
        stock_name = get_stock_name(symbol)
        
        return {
            '股票代码': symbol,
            '股票名称': stock_name,
            '最新价': round(current_price, 2),
            '信号类型': '二买' if signal_type == '2_buy' else '三买',
            '综合得分': factor_score.total_score,
            '技术面得分': factor_score.technical_score,
            '量能得分': factor_score.volume_score,
            '动量得分': factor_score.momentum_score,
            '波动率得分': factor_score.volatility_score,
            '入场价格': round(entry_price, 2),
            '止损价格': round(stop_loss, 2),
            '目标价格': round(take_profit, 2),
            '风险回报比': round((take_profit - entry_price) / (entry_price - stop_loss), 2),
            '趋势': chan_result['trend'],
            '线段数': len(chan_result['segments']),
            '中枢数': len(chan_result['pivots']),
            '分析时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        print(f"高级选股分析 {symbol} 错误: {e}")
        return None

# ============================================================================
# 数据获取函数
# ============================================================================

@st.cache_data(ttl=3600)
def get_stock_list(date: str) -> pd.DataFrame:
    """获取股票列表"""
    try:
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        return stock_df
    except Exception as e:
        print(f"获取股票列表错误: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_kline_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取K线数据"""
    try:
        rs = bs.query_history_k_data_plus(
            symbol,
            'date,code,open,high,low,close,volume,amount',
            start_date=start_date,
            end_date=end_date,
            frequency='d'
        )
        df = rs.get_data()
        return df
    except Exception as e:
        print(f"获取K线数据错误 {symbol}: {e}")
        return pd.DataFrame()

# ============================================================================
# 可视化函数
# ============================================================================

def plot_stock_chart(symbol: str, df: pd.DataFrame, analysis_result: Dict):
    """绘制股票K线图和技术指标"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} - K线图', '成交量', 'RSI')
    )
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'])
    
    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=pd.to_numeric(df['open']),
            high=pd.to_numeric(df['high']),
            low=pd.to_numeric(df['low']),
            close=pd.to_numeric(df['close']),
            name='K线'
        ),
        row=1, col=1
    )
    
    # 添加均线
    for period in [5, 10, 20]:
        if f'ma{period}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=pd.to_numeric(df[f'ma{period}']),
                    name=f'MA{period}',
                    line=dict(width=1)
                ),
                row=1, col=1
            )
    
    # 成交量
    close_prices = pd.to_numeric(df['close'])
    open_prices = pd.to_numeric(df['open'])
    colors = ['red' if close_prices.iloc[i] >= open_prices.iloc[i] else 'green' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=pd.to_numeric(df['volume']),
            name='成交量',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=pd.to_numeric(df['rsi']),
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=3, col=1
        )
        # 添加RSI参考线
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # 更新布局
    fig.update_layout(
        title=f'{symbol} 技术分析图表',
        xaxis_title='日期',
        yaxis_title='价格',
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

# ============================================================================
# 主程序
# ============================================================================

def main():
    """Streamlit主程序"""
    
    # 标题
    st.markdown('<h1 class="main-header">📈 CChanTrader-AI 高级版本</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">精准缠论算法 + 多因子融合 + 实时分析</p>', unsafe_allow_html=True)
    
    # 侧边栏配置
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ 参数配置</h2>', unsafe_allow_html=True)
        
        # 日期选择
        col1, col2 = st.columns(2)
        with col1:
            analysis_date = st.date_input(
                "分析日期",
                datetime.now(),
                max_value=datetime.now()
            )
        with col2:
            days_back = st.number_input(
                "回溯天数",
                min_value=60,
                max_value=500,
                value=200,
                step=10
            )
        
        # 市场选择
        market_type = st.multiselect(
            "市场选择",
            ["上证A股", "深证A股", "创业板"],
            default=["上证A股", "深证A股"]
        )
        
        # 选股参数
        st.markdown("---")
        st.markdown("### 🎯 选股参数")
        
        min_score = st.slider(
            "最低综合得分",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05
        )
        
        max_stocks = st.number_input(
            "最大分析数量",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )
        
        # 更新全局参数
        ADVANCED_PARAMS["selection"]["min_score"] = min_score
        
        # 分析按钮
        st.markdown("---")
        analyze_button = st.button("🚀 开始分析", type="primary", use_container_width=True)
        
        # 登录信息
        st.markdown("---")
        st.info("📊 数据来源: BaoStock")
    
    # 登录BaoStock
    lg = bs.login()
    if lg.error_code != '0':
        st.error(f"BaoStock连接失败: {lg.error_msg}")
        return
    
    try:
        # 主界面
        if analyze_button:
            with st.spinner('正在获取数据并分析...'):
                # 进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 获取股票列表
                date_str = analysis_date.strftime('%Y-%m-%d')
                status_text.text("正在获取股票列表...")
                stock_df = get_stock_list(date_str)
                
                if stock_df.empty:
                    # 如果指定日期没有数据，尝试往前找
                    for i in range(1, 10):
                        test_date = (analysis_date - timedelta(days=i)).strftime('%Y-%m-%d')
                        stock_df = get_stock_list(test_date)
                        if not stock_df.empty:
                            st.info(f"使用 {test_date} 的股票列表")
                            break
                
                if stock_df.empty:
                    st.error("无法获取股票列表")
                    return
                
                # 过滤股票
                conditions = []
                if "上证A股" in market_type:
                    conditions.append(stock_df['code'].str.contains('sh.6', na=False))
                if "深证A股" in market_type:
                    conditions.append(stock_df['code'].str.contains('sz.0', na=False))
                if "创业板" in market_type:
                    conditions.append(stock_df['code'].str.contains('sz.3', na=False))
                
                if conditions:
                    mask = conditions[0]
                    for cond in conditions[1:]:
                        mask = mask | cond
                    stock_df = stock_df[mask]
                
                # 限制数量
                stock_df = stock_df.head(max_stocks)
                
                st.info(f"📋 待分析股票: {len(stock_df)}只")
                
                # 获取K线数据
                end_date = analysis_date.strftime('%Y-%m-%d')
                start_date = (analysis_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                kline_data = {}
                for idx, (_, stock) in enumerate(stock_df.iterrows()):
                    code = stock['code']
                    progress = (idx + 1) / len(stock_df)
                    progress_bar.progress(progress)
                    status_text.text(f"正在获取 {code} 的数据... ({idx+1}/{len(stock_df)})")
                    
                    df = get_kline_data(code, start_date, end_date)
                    if not df.empty and len(df) >= 60:
                        kline_data[code] = df
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ 数据获取完成: {len(kline_data)}只")
                
                # 执行选股分析
                st.markdown("---")
                st.markdown('<h2 class="sub-header">📊 选股分析结果</h2>', unsafe_allow_html=True)
                
                results = []
                for symbol, df in kline_data.items():
                    result = advanced_stock_selection(symbol, df)
                    if result:
                        results.append(result)
                
                # 按综合得分排序
                results.sort(key=lambda x: x['综合得分'], reverse=True)
                
                if results:
                    # 转换为DataFrame
                    df_results = pd.DataFrame(results)
                    
                    # 显示统计指标
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("符合条件的股票", len(df_results))
                    with col2:
                        avg_score = df_results['综合得分'].mean()
                        st.metric("平均综合得分", f"{avg_score:.3f}")
                    with col3:
                        avg_rr = df_results['风险回报比'].mean()
                        st.metric("平均风险回报比", f"1:{avg_rr:.2f}")
                    with col4:
                        buy_signals = len(df_results[df_results['信号类型'] == '二买'])
                        st.metric("二买信号数量", buy_signals)
                    
                    # 显示表格
                    st.markdown("### 📋 详细选股列表")
                    
                    # 格式化显示
                    display_df = df_results.copy()
                    
                    # 选择显示的列
                    columns_to_show = [
                        '股票代码', '股票名称', '最新价', '信号类型', '综合得分',
                        '技术面得分', '量能得分', '动量得分', '入场价格',
                        '止损价格', '目标价格', '风险回报比', '趋势'
                    ]
                    
                    # 使用st.dataframe显示
                    st.dataframe(
                        display_df[columns_to_show],
                        use_container_width=True,
                        height=400,
                        column_config={
                            "综合得分": st.column_config.ProgressColumn(
                                "综合得分",
                                help="综合评分",
                                format="%.3f",
                                min_value=0,
                                max_value=1,
                            ),
                            "技术面得分": st.column_config.NumberColumn("技术面", format="%.3f"),
                            "量能得分": st.column_config.NumberColumn("量能", format="%.3f"),
                            "动量得分": st.column_config.NumberColumn("动量", format="%.3f"),
                            "入场价格": st.column_config.NumberColumn("入场", format="¥%.2f"),
                            "止损价格": st.column_config.NumberColumn("止损", format="¥%.2f"),
                            "目标价格": st.column_config.NumberColumn("目标", format="¥%.2f"),
                            "最新价": st.column_config.NumberColumn("最新", format="¥%.2f"),
                            "风险回报比": st.column_config.NumberColumn("风险比", format="1:%.2f"),
                        }
                    )
                    
                    # 下载按钮
                    csv = df_results.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 下载分析结果(CSV)",
                        data=csv,
                        file_name=f"cchan_results_{analysis_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # 个股详情查看
                    st.markdown("---")
                    st.markdown('<h2 class="sub-header">🔍 个股详情分析</h2>', unsafe_allow_html=True)
                    
                    selected_stock = st.selectbox(
                        "选择股票查看详细分析",
                        options=df_results['股票代码'].tolist(),
                        format_func=lambda x: f"{x} - {df_results[df_results['股票代码']==x]['股票名称'].iloc[0]}"
                    )
                    
                    if selected_stock:
                        stock_detail = df_results[df_results['股票代码'] == selected_stock].iloc[0]
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("#### 基本信息")
                            st.write(f"**股票代码**: {stock_detail['股票代码']}")
                            st.write(f"**股票名称**: {stock_detail['股票名称']}")
                            st.write(f"**最新价**: ¥{stock_detail['最新价']}")
                            st.write(f"**信号类型**: {stock_detail['信号类型']}")
                            st.write(f"**趋势**: {stock_detail['趋势']}")
                            
                            st.markdown("#### 交易策略")
                            st.write(f"**入场价格**: ¥{stock_detail['入场价格']}")
                            st.write(f"**止损价格**: ¥{stock_detail['止损价格']}")
                            st.write(f"**目标价格**: ¥{stock_detail['目标价格']}")
                            st.write(f"**风险回报比**: 1:{stock_detail['风险回报比']}")
                            
                            st.markdown("#### 多因子评分")
                            
                            # 使用进度条显示评分
                            st.write(f"**技术面得分**: {stock_detail['技术面得分']}")
                            st.progress(stock_detail['技术面得分'])
                            st.write(f"**量能得分**: {stock_detail['量能得分']}")
                            st.progress(stock_detail['量能得分'])
                            st.write(f"**动量得分**: {stock_detail['动量得分']}")
                            st.progress(stock_detail['动量得分'])
                            st.write(f"**波动率得分**: {stock_detail['波动率得分']}")
                            st.progress(stock_detail['波动率得分'])
                            st.write(f"**综合得分**: {stock_detail['综合得分']}")
                            st.progress(stock_detail['综合得分'])
                        
                        with col2:
                            # 绘制图表
                            if selected_stock in kline_data:
                                df_stock = kline_data[selected_stock]
                                # 重新分析获取详细数据用于图表
                                chan_analyzer = AdvancedChanAnalyzer(df_stock)
                                chan_result = chan_analyzer.analyze()
                                
                                fig = plot_stock_chart(selected_stock, df_stock, chan_result)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # 保存结果到文件
                    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                             f'cchan_results_{analysis_date.strftime("%Y%m%d")}.json')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        # 转换DataFrame为可序列化的格式
                        json_results = df_results.to_dict('records')
                        json.dump(json_results, f, ensure_ascii=False, indent=2)
                    
                    st.success(f"✅ 分析完成！结果已保存至: {output_file}")
                    
                else:
                    st.warning("当前市场条件下未找到符合条件的股票")
        
        else:
            # 初始界面
            st.info("👈 请在侧边栏配置参数后点击「开始分析」")
            
            # 显示功能说明
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🎯 精准缠论")
                st.write("""
                - 自动识别分型
                - 线段划分
                - 中枢确认
                - 买卖点信号
                """)
            
            with col2:
                st.markdown("### 🔄 多因子融合")
                st.write("""
                - 技术面分析
                - 量能分析
                - 动量分析
                - 波动率分析
                """)
            
            with col3:
                st.markdown("### 📊 实时分析")
                st.write("""
                - 支持日期选择
                - 表格化展示
                - 个股详情
                - 结果导出
                """)
    
    finally:
        bs.logout()

if __name__ == '__main__':
    main()
