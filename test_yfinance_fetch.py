# test_yfinance_fetch.py

import yfinance as yf

def test_yfinance_fetch(ticker="AAPL", period="5y"):
    print(f"Fetching {period} of daily data for '{ticker}' ...")
    try:
        data = yf.Ticker(ticker).history(period=period, interval="1d")
        if data is not None and not data.empty:
            print(f"\n✓ Successfully retrieved {len(data)} rows for '{ticker}'.")
            print(data.head())
            print("\nColumns:", list(data.columns))
        else:
            print(f"\n✗ No data returned for '{ticker}'. Try another ticker or shorter period.")
    except Exception as e:
        print(f"\n✗ Error fetching data for '{ticker}': {e}")

if __name__ == "__main__":
    # You can change the ticker and period here if you like
    test_yfinance_fetch("AAPL", "5y")
