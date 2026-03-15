#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CChanTrader-AI Streamlit选股表格版
- 支持日期选择
- 只显示选股结果表格
- 精准缠论算法升级
- 多因子融合系统
"""

import streamlit as st
import pandas as pd
import numpy as np
import baostock as bs
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="CChanTrader-AI 选股系统",
    page_icon="📊",
    layout="wide"
)

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
    
    def identify_fractal_points(self):
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
    
    def identify_segments(self):
        """识别线段"""
        highs, lows = self.identify_fractal_points()
        
        # 合并所有极值点
        all_points = []
        for h in highs:
            all_points.append((h, self.df['high'].iloc[h], 'high'))
        for l in lows:
            all_points.append((l, self.df['low'].iloc[l], 'low'))
        
        # 按时间排序
        all_points.sort(key=lambda x: x[0])
        
        segments = []
        for i in range(len(all_points) - 1):
            start_idx, start_price, start_type = all_points[i]
            end_idx, end_price, end_type = all_points[i + 1]
            
            # 高低点交替才能形成线段
            if start_type != end_type:
                direction = 'up' if start_type == 'low' else 'down'
                
                # 计算线段区间的高低点
                segment_data = self.df.iloc[start_idx:end_idx+1]
                high = segment_data['high'].max()
                low = segment_data['low'].min()
                
                # 计算线段强度
                strength = abs(end_price - start_price) / start_price
                
                # 计算成交量分布
                volume_profile = segment_data['volume'].mean()
                
                # 线段长度（K线数）
                duration = end_idx - start_idx + 1
                
                # 过滤太短的线段
                if duration >= ADVANCED_PARAMS["chan"]["min_segment_bars"]:
                    segments.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'direction': direction,
                        'start_price': start_price,
                        'end_price': end_price,
                        'high': high,
                        'low': low,
                        'strength': strength,
                        'volume_profile': volume_profile,
                        'duration': duration
                    })
        
        return segments
    
    def identify_pivots(self, segments):
        """识别中枢"""
        pivots = []
        
        if len(segments) < 3:
            return pivots
        
        for i in range(len(segments) - 2):
            seg1, seg2, seg3 = segments[i], segments[i+1], segments[i+2]
            
            # 检查三段式中枢
            if (seg1['direction'] != seg2['direction'] and 
                seg2['direction'] != seg3['direction'] and
                seg1['direction'] == seg3['direction']):
                
                # 计算中枢边界
                if seg1['direction'] == 'up':  # 上-下-上型中枢
                    pivot_high = min(seg1['end_price'], seg3['end_price'])
                    pivot_low = seg2['end_price']
                else:  # 下-上-下型中枢
                    pivot_high = seg2['end_price']
                    pivot_low = max(seg1['end_price'], seg3['end_price'])
                
                # 检查中枢有效性
                if pivot_high > pivot_low:
                    center = (pivot_high + pivot_low) / 2
                    strength = (pivot_high - pivot_low) / center
                    
                    # 过滤强度不足的中枢
                    if strength >= ADVANCED_PARAMS["chan"]["pivot_strength_min"]:
                        # 计算成交量密度
                        pivot_data = self.df.iloc[seg1['start_idx']:seg3['end_idx']+1]
                        volume_density = pivot_data['volume'].mean()
                        
                        # 计算突破概率
                        breakout_prob = self._calculate_breakout_probability(pivot_data)
                        
                        # 方向偏向
                        direction_bias = 'up' if seg3['strength'] > seg1['strength'] else 'down'
                        
                        pivots.append({
                            'start_idx': seg1['start_idx'],
                            'end_idx': seg3['end_idx'],
                            'high': pivot_high,
                            'low': pivot_low,
                            'center': center,
                            'strength': strength,
                            'volume_density': volume_density,
                            'breakout_probability': breakout_prob,
                            'direction_bias': direction_bias
                        })
        
        return pivots
    
    def _calculate_breakout_probability(self, pivot_data):
        """计算突破概率"""
        try:
            vol_ratio = pivot_data['vol_ratio'].mean() if 'vol_ratio' in pivot_data.columns else 1.0
            volatility = pivot_data['close'].pct_change().std()
            prob = min(0.9, max(0.1, vol_ratio * 0.3 + volatility * 100 * 0.2))
            return prob
        except:
            return 0.5
    
    def analyze(self):
        """完整分析"""
        if len(self.df) < 10:
            return self._empty_result()
        
        # 识别线段和中枢
        self.segments = self.identify_segments()
        self.pivots = self.identify_pivots(self.segments)
        
        # 趋势判断
        trend = self._determine_trend()
        
        # 信号识别
        signals = self._identify_signals()
        
        # 量价分析
        volume_analysis = self._analyze_volume()
        
        return {
            'segments': self.segments,
            'pivots': self.pivots,
            'trend': trend,
            'signals': signals,
            'volume_analysis': volume_analysis,
            'technical_data': self.df.iloc[-1].to_dict() if not self.df.empty else {}
        }
    
    def _determine_trend(self):
        """判断趋势"""
        if not self.segments:
            return 'side'
        
        recent_segments = self.segments[-3:] if len(self.segments) >= 3 else self.segments
        
        if len(recent_segments) >= 2:
            last_high = max(seg['high'] for seg in recent_segments if seg['direction'] == 'up')
            last_low = min(seg['low'] for seg in recent_segments if seg['direction'] == 'down')
            
            current_price = self.df['close'].iloc[-1]
            
            ma5 = self.df['ma5'].iloc[-1] if 'ma5' in self.df.columns else current_price
            ma20 = self.df['ma20'].iloc[-1] if 'ma20' in self.df.columns else current_price
            
            if current_price > ma5 > ma20 and current_price > last_low * 1.02:
                return 'up'
            elif current_price < ma5 < ma20 and current_price < last_high * 0.98:
                return 'down'
        
        return 'side'
    
    def _identify_signals(self):
        """识别买卖信号"""
        signals = {'1_buy': [], '2_buy': [], '3_buy': [], '1_sell': [], '2_sell': []}
        
        if not self.pivots:
            return signals
        
        current_price = self.df['close'].iloc[-1]
        
        # 检查最近的中枢
        for pivot in self.pivots[-2:]:
            # 二买信号
            if current_price > pivot['high'] * (1 + ADVANCED_PARAMS["chan"]["breakout_threshold"]):
                signals['2_buy'].append({
                    'price': current_price,
                    'pivot_center': pivot['center'],
                    'breakout_strength': (current_price - pivot['high']) / pivot['high'],
                    'confidence': pivot['breakout_probability']
                })
            
            # 三买信号
            elif pivot['low'] <= current_price <= pivot['high'] and pivot['direction_bias'] == 'up':
                signals['3_buy'].append({
                    'price': current_price,
                    'pivot_center': pivot['center'],
                    'support_strength': (current_price - pivot['low']) / (pivot['high'] - pivot['low']),
                    'confidence': pivot['breakout_probability'] * 0.8
                })
        
        return signals
    
    def _analyze_volume(self):
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
    
    def _empty_result(self):
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
    
    def __init__(self, df: pd.DataFrame, chan_result: dict):
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
            
            # 成交量趋势得分
            trend_score = 1.0 if vol_analysis['volume_trend'] == 'increasing' else 0.3
            
            # 量价配合得分
            correlation = vol_analysis['price_volume_correlation']
            corr_score = max(0, correlation) if correlation > 0 else 0
            
            # 成交量放大得分
            vol_ratio = vol_analysis['current_volume_ratio']
            ratio_score = min(1.0, vol_ratio / 2.0) if vol_ratio > 1 else 0.2
            
            # 突然放量得分
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
    
    def calculate_multi_factor_score(self):
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
        
        return {
            'technical_score': round(technical_score, 3),
            'volume_score': round(volume_score, 3),
            'momentum_score': round(momentum_score, 3),
            'volatility_score': round(volatility_score, 3),
            'total_score': round(total_score, 3),
            'risk_score': round(risk_score, 3)
        }


# ============================================================================
# 高级选股函数
# ============================================================================

def advanced_stock_selection(symbol: str, df: pd.DataFrame) -> dict:
    """高级选股函数"""
    try:
        # 数据质量检查
        if len(df) < 60 or df['volume'].sum() == 0:
            return None
        
        # 价格范围过滤
        current_price = float(df['close'].iloc[-1])
        price_range = ADVANCED_PARAMS["selection"]["price_range"]
        if not (price_range[0] <= current_price <= price_range[1]):
            return None
        
        # 流动性过滤
        avg_amount = df['amount'].iloc[-20:].mean() if 'amount' in df.columns else 0
        if avg_amount < ADVANCED_PARAMS["selection"]["min_liquidity"]:
            return None
        
        # 缠论分析
        chan_analyzer = AdvancedChanAnalyzer(df)
        chan_result = chan_analyzer.analyze()
        
        # 多因子分析
        multi_factor = MultiFactorAnalyzer(df, chan_result)
        factor_score = multi_factor.calculate_multi_factor_score()
        
        # 综合评分过滤
        if factor_score['total_score'] < ADVANCED_PARAMS["selection"]["min_score"]:
            return None
        
        # 波动率过滤
        if factor_score['volatility_score'] < (1 - ADVANCED_PARAMS["selection"]["max_volatility"]):
            return None
        
        # 信号确认
        has_buy_signal = bool(chan_result['signals']['2_buy'] or chan_result['signals']['3_buy'])
        if not has_buy_signal:
            return None
        
        # 计算入场点和止损点
        entry_price = current_price
        
        # 基于中枢计算止损
        stop_loss = entry_price * (1 - ADVANCED_PARAMS["risk"]["stop_loss_pct"])
        if chan_result['pivots']:
            latest_pivot = chan_result['pivots'][-1]
            pivot_stop = latest_pivot['low'] * 0.98
            stop_loss = max(stop_loss, pivot_stop)
        
        # 目标价位
        risk_amount = entry_price - stop_loss
        take_profit = entry_price + risk_amount * ADVANCED_PARAMS["risk"]["take_profit_ratio"]
        
        # 信号类型
        signal_type = '2_buy' if chan_result['signals']['2_buy'] else '3_buy'
        
        # 股票名称映射
        stock_names = {
            'sh.600000': '浦发银行', 'sh.600036': '招商银行', 'sh.600519': '贵州茅台',
            'sz.000001': '平安银行', 'sz.000858': '五粮液', 'sz.300750': '宁德时代',
            # 可以继续添加更多股票名称映射
        }
        stock_name = stock_names.get(symbol, symbol.split('.')[-1])
        
        return {
            '股票代码': symbol,
            '股票名称': stock_name,
            '最新价': round(entry_price, 2),
            '信号类型': signal_type,
            '综合评分': factor_score['total_score'],
            '技术面': factor_score['technical_score'],
            '量能': factor_score['volume_score'],
            '动量': factor_score['momentum_score'],
            '波动率': factor_score['volatility_score'],
            '趋势': chan_result['trend'],
            '线段数': len(chan_result['segments']),
            '中枢数': len(chan_result['pivots']),
            '止损价': round(stop_loss, 2),
            '目标价': round(take_profit, 2),
            '风报比': round((take_profit - entry_price) / (entry_price - stop_loss), 2)
        }
        
    except Exception as e:
        print(f"高级选股分析 {symbol} 错误: {e}")
        return None


# ============================================================================
# Streamlit主程序
# ============================================================================

def main():
    """Streamlit主程序"""
    st.title("📊 CChanTrader-AI 智能选股系统")
    st.markdown("---")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 选股配置")
        
        # 日期选择
        analysis_date = st.date_input(
            "选择分析日期",
            value=datetime.now(),
            max_value=datetime.now()
        )
        
        st.markdown("---")
        
        # 选股参数
        st.subheader("选股参数")
        min_score = st.slider(
            "最低综合评分",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05
        )
        
        max_stocks = st.slider(
            "最大分析股票数",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )
        
        include_3buy = st.checkbox("包含三买信号", value=True)
        include_2buy = st.checkbox("包含二买信号", value=True)
        
        st.markdown("---")
        
        # 执行按钮
        analyze_button = st.button("🚀 开始选股", type="primary", use_container_width=True)
        
        if st.button("🔄 重置参数", use_container_width=True):
            st.rerun()
    
    # 主界面
    if analyze_button:
        with st.spinner("正在进行智能选股分析..."):
            # 更新选股阈值
            ADVANCED_PARAMS["selection"]["min_score"] = min_score
            
            # 连接BaoStock
            lg = bs.login()
            if lg.error_code != '0':
                st.error(f"BaoStock连接失败: {lg.error_msg}")
                return
            
            try:
                # 获取股票列表
                query_date = analysis_date.strftime('%Y-%m-%d')
                stock_rs = bs.query_all_stock(query_date)
                stock_df = stock_rs.get_data()
                
                if stock_df.empty:
                    st.warning("未获取到股票列表数据")
                    return
                
                # 过滤股票
                a_stocks = stock_df[stock_df['code'].str.contains('sh.6|sz.0|sz.3')].head(max_stocks)
                
                # 进度条
                progress_bar = st.progress(0, text="获取K线数据...")
                
                # 获取K线数据
                end_date = analysis_date.strftime('%Y-%m-%d')
                start_date = (analysis_date - timedelta(days=200)).strftime('%Y-%m-%d')
                
                kline_data = {}
                for idx, stock in enumerate(a_stocks.iterrows()):
                    _, stock = stock
                    code = stock['code']
                    
                    rs = bs.query_history_k_data_plus(
                        code,
                        'date,code,open,high,low,close,volume,amount',
                        start_date=start_date,
                        end_date=end_date,
                        frequency='d'
                    )
                    day_df = rs.get_data()
                    
                    if not day_df.empty and len(day_df) >= 60:
                        kline_data[code] = day_df
                    
                    progress_bar.progress((idx + 1) / len(a_stocks), text=f"数据获取: {idx+1}/{len(a_stocks)}")
                
                # 高级选股分析
                progress_bar.progress(0, text="执行选股分析...")
                selected_stocks = []
                
                for idx, (symbol, df) in enumerate(kline_data.items()):
                    result = advanced_stock_selection(symbol, df)
                    if result:
                        # 信号类型过滤
                        signal_type = result['信号类型']
                        if (signal_type == '2_buy' and include_2buy) or (signal_type == '3_buy' and include_3buy):
                            selected_stocks.append(result)
                    
                    progress_bar.progress((idx + 1) / len(kline_data), text=f"选股分析: {idx+1}/{len(kline_data)}")
                
                progress_bar.empty()
                
                # 按综合评分排序
                selected_stocks.sort(key=lambda x: x['综合评分'], reverse=True)
                
                # 显示结果
                st.markdown("---")
                
                if selected_stocks:
                    # 统计信息
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("符合条件的股票", len(selected_stocks))
                    with col2:
                        buy2_count = len([s for s in selected_stocks if s['信号类型'] == '2_buy'])
                        st.metric("二买信号", buy2_count)
                    with col3:
                        buy3_count = len([s for s in selected_stocks if s['信号类型'] == '3_buy'])
                        st.metric("三买信号", buy3_count)
                    with col4:
                        avg_score = np.mean([s['综合评分'] for s in selected_stocks])
                        st.metric("平均评分", f"{avg_score:.3f}")
                    
                    st.markdown("---")
                    
                    # 转换为DataFrame
                    df_results = pd.DataFrame(selected_stocks)
                    
                    # 格式化显示
                    df_results['综合评分'] = df_results['综合评分'].apply(lambda x: f"{x:.3f}")
                    df_results['技术面'] = df_results['技术面'].apply(lambda x: f"{x:.3f}")
                    df_results['量能'] = df_results['量能'].apply(lambda x: f"{x:.3f}")
                    df_results['动量'] = df_results['动量'].apply(lambda x: f"{x:.3f}")
                    df_results['波动率'] = df_results['波动率'].apply(lambda x: f"{x:.3f}")
                    
                    # 添加颜色标记
                    def color_signal(val):
                        if val == '2_buy':
                            return 'background-color: #90EE90'  # 浅绿色
                        elif val == '3_buy':
                            return 'background-color: #87CEEB'  # 浅蓝色
                        return ''
                    
                    def color_trend(val):
                        if val == 'up':
                            return 'color: red'
                        elif val == 'down':
                            return 'color: green'
                        return ''
                    
                    # 显示表格
                    st.subheader(f"📋 选股结果 ({len(selected_stocks)}只)")
                    
                    styled_df = df_results.style.applymap(color_signal, subset=['信号类型'])\
                                                   .applymap(color_trend, subset=['趋势'])
                    
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=600
                    )
                    
                    # 下载按钮
                    csv = df_results.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 下载选股结果",
                        data=csv,
                        file_name=f"cchan_selection_{analysis_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.warning("❌ 当前市场条件下未找到符合条件的股票")
                    
                    # 显示参数建议
                    st.info("💡 建议调整参数：")
                    st.write("- 降低最低综合评分")
                    st.write("- 扩大分析股票数量")
                    st.write("- 检查市场整体状况")
                
            finally:
                bs.logout()
    
    else:
        # 初始界面说明
        st.info("👈 请在左侧配置选股参数，然后点击「开始选股」")
        
        st.markdown("""
        ### 🎯 系统特点
        
        - **精准缠论算法**：自动识别线段和中枢，生成买卖信号
        - **多因子融合**：技术面、量能、动量、波动率综合评分
        - **智能风控**：自动计算止损位和目标位
        - **实时分析**：支持选择任意交易日进行分析
        
        ### 📊 信号说明
        
        - **二买**：突破中枢上沿，强势上涨信号
        - **三买**：回踩中枢后再次向上，稳健入场信号
        
        ### ⚡ 使用说明
        
        1. 在左侧选择分析日期
        2. 调整选股参数（可选）
        3. 点击「开始选股」执行分析
        4. 查看并下载选股结果
        """)


if __name__ == '__main__':
    main()
