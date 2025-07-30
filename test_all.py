"""
Comprehensive test suite for Stock Analyzer.
Tests all modules and identifies issues.
"""

import sys
import os
import traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all modules to test
try:
    from config import config
    print("✓ Config module imported successfully")
except Exception as e:
    print(f"✗ Failed to import config: {e}")
    sys.exit(1)

try:
    from utils import get_logger, ProgressTracker, DataFormatter, CacheManager, FileManager
    print("✓ Utils module imported successfully")
except Exception as e:
    print(f"✗ Failed to import utils: {e}")
    sys.exit(1)

try:
    from validators import (
        input_validator, data_quality_validator, 
        api_response_validator, config_validator,
        ValidationError
    )
    print("✓ Validators module imported successfully")
except Exception as e:
    print(f"✗ Failed to import validators: {e}")
    sys.exit(1)

try:
    from data_processor import StockDataProcessor
    print("✓ Data processor module imported successfully")
except Exception as e:
    print(f"✗ Failed to import data_processor: {e}")
    sys.exit(1)


class StockAnalyzerTester:
    """Comprehensive testing class for Stock Analyzer."""
    
    def __init__(self):
        self.logger = get_logger("StockAnalyzerTester")
        self.results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'errors': []
        }
        
    def run_all_tests(self):
        """Run all test suites."""
        print("\n" + "="*70)
        print("STOCK ANALYZER COMPREHENSIVE TEST SUITE")
        print("="*70)
        
        # Test configuration
        self._test_configuration()
        
        # Test validators
        self._test_validators()
        
        # Test utilities
        self._test_utilities()
        
        # Test data processor
        self._test_data_processor()
        
        # Test integration
        self._test_integration()
        
        # Print summary
        self._print_summary()
        
    def _test_configuration(self):
        """Test configuration settings."""
        print("\n1. TESTING CONFIGURATION")
        print("-" * 40)
        
        # Check if directories exist
        try:
            config.OUTPUT_DIR.mkdir(exist_ok=True)
            config.LOGS_DIR.mkdir(exist_ok=True)
            config.CACHE_DIR.mkdir(exist_ok=True)
            self._pass("Created/verified required directories")
        except Exception as e:
            self._fail(f"Failed to create directories: {e}")
        
        # Validate configuration
        is_valid, errors = config_validator.validate_config()
        if is_valid:
            self._pass("Configuration validation passed")
        else:
            for error in errors:
                if "placeholder" in error.lower():
                    self._warn(f"Configuration warning: {error}")
                else:
                    self._fail(f"Configuration error: {error}")
        
        # Check critical settings
        if hasattr(config, 'DATA_PERIOD_YEARS'):
            self._pass(f"DATA_PERIOD_YEARS set to {config.DATA_PERIOD_YEARS}")
        else:
            self._fail("DATA_PERIOD_YEARS not found in config")
            
    def _test_validators(self):
        """Test all validator functions."""
        print("\n2. TESTING VALIDATORS")
        print("-" * 40)
        
        # Test ticker validation
        test_tickers = [
            ("AAPL", True, "Valid ticker"),
            ("MSFT", True, "Valid ticker"),
            ("GOOG.L", True, "Valid ticker with exchange"),
            ("123", False, "Invalid - numbers only"),
            ("TOOLONG", False, "Invalid - too many characters"),
            ("", False, "Invalid - empty"),
            ("aapl", True, "Valid - should be uppercased"),
        ]
        
        for ticker, should_pass, description in test_tickers:
            try:
                validated = input_validator.validate_ticker(ticker)
                if should_pass:
                    self._pass(f"Ticker validation: {ticker} -> {validated} ({description})")
                else:
                    self._fail(f"Ticker validation should have failed: {ticker} ({description})")
            except ValidationError:
                if not should_pass:
                    self._pass(f"Ticker validation correctly failed: {ticker} ({description})")
                else:
                    self._fail(f"Ticker validation incorrectly failed: {ticker} ({description})")
        
        # Test date validation
        try:
            start_dt, end_dt = input_validator.validate_date_range(None, None)
            self._pass(f"Date validation with defaults: {start_dt.date()} to {end_dt.date()}")
        except Exception as e:
            self._fail(f"Date validation failed: {e}")
        
        # Test data quality validation
        test_df = self._create_test_dataframe()
        is_valid, warnings, report = data_quality_validator.validate_dataframe(test_df, "TEST")
        if is_valid:
            self._pass(f"Data quality validation passed (score: {report['completeness_score']:.1f}%)")
        else:
            self._fail(f"Data quality validation failed: {warnings}")
            
    def _test_utilities(self):
        """Test utility functions."""
        print("\n3. TESTING UTILITIES")
        print("-" * 40)
        
        # Test logger
        try:
            test_logger = get_logger("TestLogger")
            test_logger.info("Test log message")
            self._pass("Logger functionality working")
        except Exception as e:
            self._fail(f"Logger failed: {e}")
        
        # Test data formatter
        try:
            formatter = DataFormatter()
            assert formatter.format_currency(1234.56) == "$1,234.56"
            assert formatter.format_percentage(12.345) == "12.35%"
            assert formatter.format_large_number(1.5e9) == "1.50B"
            self._pass("Data formatter working correctly")
        except Exception as e:
            self._fail(f"Data formatter failed: {e}")
        
        # Test cache manager
        try:
            cache = CacheManager()
            test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
            cache.set("test_key", test_data)
            retrieved = cache.get("test_key")
            if retrieved == test_data:
                self._pass("Cache manager working correctly")
            else:
                self._fail("Cache manager data mismatch")
        except Exception as e:
            self._fail(f"Cache manager failed: {e}")
            
    def _test_data_processor(self):
        """Test the main data processor."""
        print("\n4. TESTING DATA PROCESSOR")
        print("-" * 40)
        
        processor = StockDataProcessor()
        
        # Test with sample data generation
        test_ticker = "AAPL"
        
        # Test data fetching
        try:
            # Force sample data for testing
            original_debug = config.DEBUG_MODE
            original_sample = config.USE_SAMPLE_DATA
            config.DEBUG_MODE = True
            config.USE_SAMPLE_DATA = True
            
            hist_data = processor._fetch_historical_data(test_ticker)
            if hist_data is not None and not hist_data.empty:
                self._pass(f"Historical data fetched: {len(hist_data)} rows")
                
                # Test data cleaning
                cleaned_data = processor._clean_historical_data(hist_data)
                self._pass(f"Data cleaned successfully: {len(cleaned_data)} rows")
                
                # Test technical indicators
                technical_data = processor._calculate_technical_indicators(cleaned_data)
                indicators_count = len(technical_data.columns) - len(cleaned_data.columns)
                self._pass(f"Technical indicators calculated: {indicators_count} indicators")
                
                # Test summary statistics
                summary = processor._calculate_summary_statistics(technical_data, test_ticker)
                if 'current_price' in summary:
                    self._pass(f"Summary statistics calculated: Current price ${summary['current_price']}")
                else:
                    self._fail("Summary statistics missing current price")
                    
            else:
                self._fail("Failed to fetch historical data")
                
            # Restore original settings
            config.DEBUG_MODE = original_debug
            config.USE_SAMPLE_DATA = original_sample
            
        except Exception as e:
            self._fail(f"Data processor error: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
    def _test_integration(self):
        """Test full integration."""
        print("\n5. TESTING FULL INTEGRATION")
        print("-" * 40)
        
        # Test processing a complete stock
        test_tickers = ["AAPL", "MSFT", "GOOGL"]
        
        processor = StockDataProcessor()
        
        # Force sample data for testing
        original_debug = config.DEBUG_MODE
        original_sample = config.USE_SAMPLE_DATA
        config.DEBUG_MODE = True
        config.USE_SAMPLE_DATA = True
        
        for ticker in test_tickers[:1]:  # Test just one for speed
            try:
                print(f"\nProcessing {ticker}...")
                result = processor.process_stock(ticker)
                
                # Validate result structure
                required_keys = ['ticker', 'company_info', 'summary_statistics', 
                               'data_quality', 'historical_data', 'metadata']
                missing_keys = [k for k in required_keys if k not in result]
                
                if not missing_keys:
                    self._pass(f"Successfully processed {ticker}")
                    
                    # Check some specific values
                    if result['summary_statistics']['current_price'] > 0:
                        self._pass(f"{ticker} current price: ${result['summary_statistics']['current_price']}")
                    
                    if result['data_quality']['completeness_score'] > 90:
                        self._pass(f"{ticker} data quality score: {result['data_quality']['completeness_score']:.1f}%")
                    else:
                        self._warn(f"{ticker} low data quality score: {result['data_quality']['completeness_score']:.1f}%")
                else:
                    self._fail(f"Missing keys in result: {missing_keys}")
                    
            except Exception as e:
                self._fail(f"Failed to process {ticker}: {e}")
                self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # Restore original settings
        config.DEBUG_MODE = original_debug
        config.USE_SAMPLE_DATA = original_sample
        
    def _create_test_dataframe(self) -> pd.DataFrame:
        """Create a test dataframe for validation."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        data = {
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(100, 200, len(dates)),
            'Low': np.random.uniform(100, 200, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }
        df = pd.DataFrame(data, index=dates)
        
        # Fix OHLC logic
        for i in range(len(df)):
            df.loc[df.index[i], 'High'] = max(
                df.loc[df.index[i], 'Open'],
                df.loc[df.index[i], 'High'],
                df.loc[df.index[i], 'Low'],
                df.loc[df.index[i], 'Close']
            )
            df.loc[df.index[i], 'Low'] = min(
                df.loc[df.index[i], 'Open'],
                df.loc[df.index[i], 'High'],
                df.loc[df.index[i], 'Low'],
                df.loc[df.index[i], 'Close']
            )
        
        return df
        
    def _pass(self, message: str):
        """Record a passing test."""
        self.results['passed'] += 1
        print(f"✓ {message}")
        
    def _fail(self, message: str):
        """Record a failing test."""
        self.results['failed'] += 1
        self.results['errors'].append(message)
        print(f"✗ {message}")
        
    def _warn(self, message: str):
        """Record a warning."""
        self.results['warnings'] += 1
        print(f"⚠ {message}")
        
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total = self.results['passed'] + self.results['failed']
        pass_rate = (self.results['passed'] / total * 100) if total > 0 else 0
        
        print(f"Total Tests: {total}")
        print(f"Passed: {self.results['passed']} ({pass_rate:.1f}%)")
        print(f"Failed: {self.results['failed']}")
        print(f"Warnings: {self.results['warnings']}")
        
        if self.results['errors']:
            print("\nERRORS:")
            for i, error in enumerate(self.results['errors'], 1):
                print(f"{i}. {error}")
        
        if self.results['failed'] == 0:
            print("\n✅ ALL TESTS PASSED!")
        else:
            print("\n❌ SOME TESTS FAILED - Please review errors above")
            
        # Provide recommendations
        print("\nRECOMMENDATIONS:")
        if self.results['warnings'] > 0:
            print("• Some warnings were found (mostly API key placeholders)")
            print("• The system will work with sample data for testing")
        
        print("• To use real data, configure API keys in config.py")
        print("• Yahoo Finance should work without API keys for basic data")
        

def main():
    """Run the comprehensive test suite."""
    tester = StockAnalyzerTester()
    
    try:
        tester.run_all_tests()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)
        

if __name__ == "__main__":
    main()