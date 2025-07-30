# Stock Analyzer

> Financial analysis platform with AI-powered insights and professional reporting

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 🚀 Features

- **📊 Advanced Technical Analysis** - 9+ indicators including RSI, MACD, Bollinger Bands, and more
- **🤖 AI-Powered Event Analysis** - OpenAI GPT integration with cost optimization
- **📰 Sentiment Analysis** - Custom financial lexicon with dependency-free implementation
- **📈 Professional Reports** - Publication-quality Excel reports with embedded charts
- **🔄 Multi-Source Data** - Yahoo Finance, Alpha Vantage with intelligent fallback
- **⚡ Batch Processing** - Analyze multiple stocks simultaneously
- **🎯 Dynamic Intelligence** - Two-phase learning system for cost-effective analysis

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features-1)
- [Configuration](#configuration)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🏃 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/stock-analyzer.git
cd stock-analyzer

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (optional)

# Analyze a stock
python main.py --ticker AAPL

# Interactive mode
python main.py --interactive
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- 4GB RAM (8GB recommended)
- Internet connection for data APIs

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/stock-analyzer.git
cd stock-analyzer
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Environment Setup (Optional)

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
OPENAI_API_KEY=your-openai-api-key-here
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key-here
```

> **Note**: API keys are optional. The system works with Yahoo Finance by default and can generate sample data for testing.

## 🎯 Usage

### Command Line Interface

```bash
# Analyze single stock
python main.py --ticker AAPL

# Batch analysis from file
python main.py --batch stocks.txt

# Custom output directory
python main.py --ticker MSFT --output /path/to/output

# Interactive mode
python main.py --interactive
```

### Interactive Mode Commands

```
analyze AAPL        # Analyze Apple stock
batch stocks.txt    # Process multiple stocks
settings           # Show current configuration
help               # Display help
quit               # Exit program
```

### Python API

```python
from stock_analyzer_handler import StockAnalyzerHandler

# Initialize analyzer
analyzer = StockAnalyzerHandler()

# Analyze single stock
result = analyzer.analyze_stock('AAPL')

if result['success']:
    print(f"Report saved: {result['output_file']}")
else:
    print(f"Error: {result['error']}")
```

## ✨ Features

### Technical Analysis

- **Trend Indicators**: SMA (20,50,200), EMA (12,26), MACD
- **Momentum**: RSI (14), Stochastic Oscillator
- **Volatility**: Bollinger Bands, Average True Range (ATR)
- **Volume**: Volume SMA, On-Balance Volume (OBV)
- **Support/Resistance**: Dynamic level detection

### AI-Powered Analysis

- **Dynamic Intelligence**: Learning phase → Knowledge phase optimization
- **Event Detection**: Significant price movement analysis with GPT
- **Cost Optimization**: 60-80% API cost reduction through smart method selection
- **Confidence Scoring**: 0.0-1.0 reliability assessment

### Sentiment Analysis

- **Financial Lexicon**: Custom domain-specific terminology
- **Intensity Classification**: Strong/Moderate/Weak sentiment levels
- **Amplifier Detection**: Earnings, mergers, IPO impact weighting
- **Batch Processing**: Multiple articles with relevance scoring

### Professional Reporting

#### 9-Sheet Excel Report Structure:
1. **Executive Summary** - Key metrics and trading signals
2. **Company Profile** - Business information and financials
3. **Technical Charts** - High-quality Matplotlib visualizations
4. **Technical Analysis** - 60-day indicator data
5. **Sentiment Analysis** - AI event breakdown
6. **Performance Metrics** - Risk-adjusted returns
7. **Raw Data** - Complete historical dataset
8. **Data Quality** - Validation and integrity reports
9. **Metadata** - Analysis documentation

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Keys (Optional)
OPENAI_API_KEY=your-openai-api-key-here
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key-here

# Configuration
DEBUG=False
LOG_LEVEL=INFO
USE_SAMPLE_DATA=False
CACHE_ENABLED=True
MAX_RETRIES=3
```

### Config File Settings

Edit `config.py` for advanced configuration:

```python
# Data Collection
DATA_PERIOD_YEARS = 5
DATA_SOURCES = ['yfinance', 'alpha_vantage']

# Technical Indicators
MA_PERIODS = [5, 10, 20, 50, 200]
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

# Event Analysis
SIGNIFICANCE_THRESHOLD = 3.0  # Learning phase
KNOWLEDGE_THRESHOLD = 7.5     # Knowledge phase
```

## 📚 Examples

### Single Stock Analysis

```bash
python main.py --ticker AAPL
```

**Output**: `AAPL_analysis_report_20250730_143022.xlsx`

### Batch Processing

Create `stocks.txt`:
```
AAPL
MSFT
GOOGL
AMZN
TSLA
```

```bash
python main.py --batch stocks.txt
```

### Portfolio Analysis

```python
from stock_analyzer_handler import StockAnalyzerHandler
from excel_report_generator import ExcelReportGenerator

analyzer = StockAnalyzerHandler()
report_gen = ExcelReportGenerator()

# Analyze multiple stocks
tickers = ['AAPL', 'MSFT', 'GOOGL']
results = []

for ticker in tickers:
    result = analyzer.analyze_stock(ticker)
    if result['success']:
        results.append(result['data'])

# Generate portfolio report
portfolio_report = report_gen.create_portfolio_report(results)
print(f"Portfolio report: {portfolio_report}")
```

## 🔧 API Reference

### Core Classes

#### `StockAnalyzerHandler`
Main interface for stock analysis.

```python
class StockAnalyzerHandler:
    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """Analyze single stock and generate report."""
        
    def validate_ticker(self, ticker: str) -> bool:
        """Validate ticker symbol format."""
```

#### `StockDataProcessor`
Core data processing engine.

```python
class StockDataProcessor:
    def process_stock(self, ticker: str) -> Dict[str, Any]:
        """Complete stock processing pipeline."""
        
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
```

#### `EnhancedEventAnalyzer`
AI-powered event analysis.

```python
class EnhancedEventAnalyzer:
    def analyze_event_intelligent(self, event: PriceEvent) -> EventAnalysis:
        """Intelligent event analysis with cost optimization."""
        
    def batch_analyze_events(self, events: List[PriceEvent]) -> Dict[str, EventAnalysis]:
        """Analyze multiple events efficiently."""
```

### Data Structures

```python
@dataclass
class PriceEvent:
    date: datetime
    ticker: str
    open_price: float
    close_price: float
    change_percent: float
    volume: int
    is_significant: bool

@dataclass
class EventAnalysis:
    event_reason: str
    event_type: str
    sentiment: str
    confidence_score: float
    impact_level: str
    news_sources_count: int
    analysis_timestamp: datetime
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install --upgrade -r requirements.txt
```

**API Rate Limits**
- Alpha Vantage: 5 calls/minute, 25/day (free tier)
- OpenAI: Check your usage limits
- System automatically uses fallbacks

**Memory Issues**
- Reduce `DATA_PERIOD_YEARS` in config
- Enable `CACHE_ENABLED=True`
- Analyze fewer stocks simultaneously

**Excel Generation Errors**
```bash
# Install/update openpyxl
pip install --upgrade openpyxl matplotlib
```

### Debug Mode

Enable detailed logging:
```bash
export DEBUG=True
export LOG_LEVEL=DEBUG
python main.py --ticker AAPL
```

## 📊 Performance Tips

- **Enable Caching**: Set `CACHE_ENABLED=True` in `.env`
- **Use Batch Mode**: More efficient for multiple stocks
- **API Key Setup**: Reduces reliance on free tier limits
- **Sample Data**: Use `USE_SAMPLE_DATA=True` for testing

## 🧪 Testing

```bash
# Run individual component tests
python utils.py
python validators.py
python sentiment_analyzer.py

# Test with sample data
USE_SAMPLE_DATA=True python main.py --ticker TEST
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black stock_analyzer/
flake8 stock_analyzer/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- OpenAI for GPT API integration
- Yahoo Finance for market data
- Alpha Vantage for additional data sources
- Python community for excellent libraries
