"""
Stock Analyzer Handler
Coordinates stock data processing and Excel report generation
"""

from data_processor import StockDataProcessor
from excel_generator import ExcelReportGenerator
from utils import get_logger
import os
from typing import Dict


class StockAnalyzerHandler:
    """Handles the full stock analysis pipeline using internal processors."""

    def __init__(self):
        self.logger = get_logger("StockAnalyzerHandler")
        self.data_processor = StockDataProcessor()
        self.excel_generator = ExcelReportGenerator()
        self.logger.info("Stock Analyzer Handler initialized")

    def analyze_stock(self, ticker: str) -> Dict:
        """
        Run stock analysis pipeline for the given ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with success status and output file path
        """
        try:
            self.logger.info(f"Starting analysis for {ticker}")

            # Step 1: Process stock data
            data_bundle = self.data_processor.process_stock(ticker)

            # Step 2: Generate Excel report
            output_path = self.excel_generator.generate_report(data_bundle)
            
            self.logger.info(f"Analysis completed for {ticker}")
            return {
                'success': True,
                'output_file': output_path
            }

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Optional test entry point
if __name__ == "__main__":
    handler = StockAnalyzerHandler()
    ticker = "AAPL"
    result = handler.analyze_stock(ticker)

    if result['success']:
        print(f"✓ Analysis successful: {result['output_file']}")
    else:
        print(f"✗ Analysis failed: {result['error']}")
