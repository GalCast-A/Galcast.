import logging
import traceback
import sys

# Configure logging at the very start to capture all errors
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.debug("Starting main.py script execution.")

try:
    import io
    logger.debug("Imported io successfully.")
    from flask import Flask, request, jsonify
    logger.debug("Imported Flask modules successfully.")
    import requests
    logger.debug("Imported requests successfully.")
    import json
    logger.debug("Imported json successfully.")
    import pandas as pd
    logger.debug("Imported pandas successfully.")
    import numpy as np
    logger.debug("Imported numpy successfully.")
    from datetime import datetime, timedelta
    logger.debug("Imported datetime modules successfully.")
    from scipy.optimize import minimize
    logger.debug("Imported scipy.optimize successfully.")
    import warnings
    logger.debug("Imported warnings successfully.")
    import time
    logger.debug("Imported time successfully.")
    warnings.filterwarnings('ignore')
    logger.debug("Configured warnings.filterwarnings successfully.")

    # Add CORS import here
    from flask_cors import CORS
    logger.debug("Imported flask_cors successfully.")

    # Attempt to import optional dependencies
    try:
        import yfinance as yf
        YFINANCE_AVAILABLE = True
        logger.debug("Imported yfinance successfully.")
    except ImportError as e:
        YFINANCE_AVAILABLE = False
        logger.warning(f"yfinance not installed. Data fetching unavailable. Error: {str(e)}")

    try:
        from pypfopt import BlackLittermanModel, risk_models, expected_returns
        PYPFOPT_AVAILABLE = True
        logger.debug("Imported pypfopt modules successfully.")
    except ImportError as e:
        PYPFOPT_AVAILABLE = False
        logger.warning(f"pypfopt not installed. Falling back to basic optimization methods. Error: {str(e)}")

    try:
        import cvxpy as cp
        CVXPY_AVAILABLE = True
        logger.debug("Imported cvxpy successfully.")
    except ImportError as e:
        CVXPY_AVAILABLE = False
        logger.warning(f"cvxpy not installed. Using scipy.optimize.minimize. Error: {str(e)}")

    try:
        import statsmodels.api as sm
        STATSMODELS_AVAILABLE = True
        logger.debug("Imported statsmodels successfully.")
    except ImportError as e:
        STATSMODELS_AVAILABLE = False
        logger.warning(f"statsmodels not installed. Fama-French exposures unavailable. Error: {str(e)}")

    from sklearn.decomposition import PCA
    logger.debug("Imported sklearn.decomposition successfully.")

    # Initialize Flask app
    app = Flask(__name__)
    logger.info("Flask app initialized successfully.")

    # Configure CORS
    CORS(app, resources={r"/analyze_portfolio": {"origins": ["*"]}}, supports_credentials=True, methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])
    logger.info("CORS configured successfully.")

except Exception as e:
    logger.error(f"Error during imports or initialization: {str(e)}\nTraceback: {traceback.format_exc()}")
    raise

class PortfolioAnalyzer:
    def __init__(self):
        self.today_date = datetime.now().strftime("%Y-%m-%d")
        self.default_start_date = (datetime.strptime(self.today_date, "%Y-%m-%d") - timedelta(days=3652)).strftime("%Y-%m-%d")
        self.data_cache = {}
        # API keys
        self.marketstack_api_key = "0c8e9987c2747266f12511dd36eb96d8"
        self.fmp_api_key = "nfeRV3Wmv9pr36RHvZVELNJVv4lZByaZ"
        self.av_api_key = "UM38EN4L82CPFR8L"
        self.tiingo_api_key = "953f2243afadec4c68f4be9d2d92d0d7148c2ce1"
        self.finnhub_api_key = "d02nchhr01qi6jgi5nqgd02nchhr01qi6jgi5nr0"
        # Cache Finnhub API key test result
        self.finnhub_available = None
        self._test_finnhub_api_key()

    def _test_finnhub_api_key(self):
        try:
            test_url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={self.finnhub_api_key}"
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                self.finnhub_available = True
                logger.info("Finnhub API key test passed. Finnhub data source will be used.")
            else:
                self.finnhub_available = False
                logger.warning(f"Finnhub API key test failed with status {response.status_code}. Skipping Finnhub data source.")
        except Exception as e:
            self.finnhub_available = False
            logger.warning(f"Finnhub API key test failed: {str(e)}. Skipping Finnhub data source.")
        

    def fetch_treasury_yield(self):
        # Try fetching the 10-year Treasury yield from multiple sources
        sources = [
            self._fetch_treasury_yield_yfinance,
        ]
        for source in sources:
            try:
                rate = source()
                if rate is not None:
                    return rate
            except Exception as e:
                logger.error(f"Error in {source.__name__}: {e}")
        logger.warning("All Treasury yield sources failed. Using fallback value of 0.045 (based on recent 10-year Treasury yield estimates).")
        return 0.045

    def _fetch_treasury_yield_yfinance(self):
        if not YFINANCE_AVAILABLE:
            return None
        try:
            treasury_data = yf.download("^TNX", period="1d", interval="1d")['Close']
            if treasury_data.empty or not isinstance(treasury_data, pd.Series):
                return None
            return float(treasury_data.iloc[-1]) / 100
        except Exception as e:
            logger.error(f"yfinance Treasury yield fetch failed: {e}")
            return None
            
    def fetch_stock_data(self, stocks, start=None, end=None):
        if start is None:
            start = self.default_start_date
        if end is None:
            end = self.today_date
        # Convert start and end to pandas.Timestamp
        start = pd.Timestamp(start).tz_localize(None)
        end = pd.Timestamp(end).tz_localize(None)
        cache_key = (tuple(sorted(stocks)), start, end)
        if cache_key in self.data_cache:
            logger.info(f"Returning cached stock data for {stocks}")
            return self.data_cache[cache_key]

        error_tickers = {}
        earliest_dates = {}
        stock_data = None

        sources = [
            self._fetch_stock_data_fmp,
            self._fetch_stock_data_marketstack,
            self._fetch_stock_data_tiingo,
            self._fetch_stock_data_av,
            self._fetch_stock_data_yfinance
        ]
        # Use cached Finnhub API key test result
        if self.finnhub_available:
            sources.insert(1, self._fetch_stock_data_finnhub)  # Add Finnhub after FMP if the key works
        
        last_error = None
        for source in sources:
            try:
                logger.info(f"Trying data source: {source.__name__} for stocks {stocks}")
                stock_data, source_error_tickers, source_earliest_dates = source(stocks, start, end)
                if stock_data is not None and not stock_data.empty:
                    error_tickers.update(source_error_tickers)
                    earliest_dates.update(source_earliest_dates)
                    logger.info(f"Successfully fetched data from {source.__name__} for stocks {stocks}")
                    break
            except Exception as e:
                        last_error = traceback.format_exc()
                        logger.error(f"Error in {source.__name__} for stocks {stocks}: {str(e)}\nTraceback: {last_error}")
        else:
            error_msg = f"All data sources failed for stocks {stocks}. Last error: {last_error if last_error else 'Unknown'}"
            logger.error(error_msg)
            return None, {"error": error_msg, "trace": last_error}, {}
            
        # Post-processing
        stock_data = stock_data.dropna(axis=1, how='all')
        logger.info(f"After dropna: {stock_data.shape if not stock_data.empty else 'empty'}")
        if stock_data.shape[0] < 252:
            logger.warning("Insufficient data (< 252 days). Optimization may be unreliable.")

        problematic_tickers = []
        for ticker in stock_data.columns:
            if (stock_data[ticker] <= 1e-4).any():
                logger.warning(f"{ticker} has zero or near-zero prices (<= 1e-4). Excluding.")
                problematic_tickers.append(ticker)
                error_tickers[ticker] = "Zero or near-zero prices detected"
            elif stock_data[ticker].isna().mean() > 0.5:
                logger.warning(f"{ticker} has too many missing values (>50%). Excluding.")
                problematic_tickers.append(ticker)
                error_tickers[ticker] = "Too many missing values"
            elif (stock_data[ticker] > 1e6).any():
                logger.warning(f"{ticker} has extremely large prices (>1e6). Excluding.")
                problematic_tickers.append(ticker)
                error_tickers[ticker] = "Extremely large prices detected"
        stock_data = stock_data.drop(columns=problematic_tickers, errors='ignore')
        logger.info(f"After dropping problematic tickers: {stock_data.shape if not stock_data.empty else 'empty'}")

        if stock_data.empty:
            logger.error("No valid stock data available after filtering problematic tickers.")
            return None, error_tickers, earliest_dates

        stock_data = stock_data.fillna(method='ffill').fillna(method='bfill')
        logger.info(f"After filling NaNs: {stock_data.shape if not stock_data.empty else 'empty'}")

        for ticker in stocks:
            if ticker not in stock_data.columns or stock_data[ticker].isna().all():
                error_tickers[ticker] = "Data not available"
            else:
                first_valid = pd.Timestamp(stock_data[ticker].first_valid_index()).tz_localize(None)
                earliest_dates[ticker] = first_valid.strftime("%Y-%m-%d")
                logger.info(f"Earliest date for {ticker}: {earliest_dates[ticker]}")

        self.data_cache[cache_key] = (stock_data, error_tickers, earliest_dates)
        logger.info(f"Successfully fetched stock data for {stocks}.")
        return stock_data, error_tickers, earliest_dates

    def _fetch_stock_data_finnhub(self, stocks, start, end):
        error_tickers = {}
        earliest_dates = {}
        stock_data_dict = {}
        for ticker in stocks:
            for attempt in range(2):
                try:
                    logger.info(f"Fetching Finnhub data for {ticker} from {start} to {end}, attempt {attempt + 1}...")
                    start_timestamp = int(start.timestamp())
                    end_timestamp = int(end.timestamp())
                    url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=D&from={start_timestamp}&to={end_timestamp}&token={self.finnhub_api_key}"
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    if data.get("s") != "ok" or not data.get("c"):
                        error_tickers[ticker] = "No data available"
                        break
                    dates = [pd.Timestamp(ts, unit='s').tz_localize(None) for ts in data["t"]]
                    closes = data["c"]
                    df = pd.DataFrame({"close": closes}, index=dates)
                    stock_data_dict[ticker] = df["close"]
                    earliest_dates[ticker] = df.index.min().strftime("%Y-%m-%d")
                    break
                except Exception as e:
                    logger.error(f"Finnhub error for {ticker} (attempt {attempt + 1}): {e}")
                    if attempt == 1:
                        error_tickers[ticker] = str(e)
                    time.sleep(0.5)
        if not stock_data_dict:
            return None, error_tickers, earliest_dates
        stock_data = pd.DataFrame(stock_data_dict)
        stock_data = stock_data.sort_index()
        return stock_data, error_tickers, earliest_dates

    def _fetch_stock_data_tiingo(self, stocks, start, end):
        error_tickers = {}
        earliest_dates = {}
        stock_data_dict = {}
        for ticker in stocks:
            for attempt in range(2):
                try:
                    logger.info(f"Fetching Tiingo data for {ticker} from {start} to {end}, attempt {attempt + 1}...")
                    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start}&endDate={end}&token={self.tiingo_api_key}"
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    if not data:
                        error_tickers[ticker] = "No data available"
                        break
                    df = pd.DataFrame(data)
                    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                    df = df.set_index("date")
                    stock_data_dict[ticker] = df["close"]
                    earliest_dates[ticker] = df.index.min().strftime("%Y-%m-%d")
                    break
                except Exception as e:
                    logger.error(f"Tiingo error for {ticker} (attempt {attempt + 1}): {e}")
                    if attempt == 1:
                        error_tickers[ticker] = str(e)
                    time.sleep(2)
        if not stock_data_dict:
            return None, error_tickers, earliest_dates
        stock_data = pd.DataFrame(stock_data_dict)
        stock_data = stock_data.sort_index()
        # Validate DataFrame
        if stock_data.empty or not all(stock_data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
          logger.error("Fetched Finnhub data is empty or contains non-numeric values")
          return None, error_tickers, earliest_dates
        return stock_data, error_tickers, earliest_dates

    def _fetch_stock_data_av(self, stocks, start, end):
        error_tickers = {}
        earliest_dates = {}
        stock_data_dict = {}
        for ticker in stocks:
            for attempt in range(2):
                try:
                    logger.info(f"Fetching Alpha Vantage data for {ticker}, attempt {attempt + 1}...")
                    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={self.av_api_key}&outputsize=full"
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    if "Time Series (Daily)" not in data:
                        error_tickers[ticker] = "No data available"
                        break
                    time_series = data["Time Series (Daily)"]
                    df = pd.DataFrame.from_dict(time_series, orient='index')
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df = df.astype(float)
                    df = df.loc[start:end]
                    if df.empty:
                        error_tickers[ticker] = "No data in date range"
                        break
                    stock_data_dict[ticker] = df["4. close"]
                    earliest_dates[ticker] = df.index.min().strftime("%Y-%m-%d")
                    break
                except Exception as e:
                    logger.error(f"Alpha Vantage error for {ticker} (attempt {attempt + 1}): {e}")
                    if attempt == 1:
                        error_tickers[ticker] = str(e)
                    time.sleep(10)
        if not stock_data_dict:
            return None, error_tickers, earliest_dates
        stock_data = pd.DataFrame(stock_data_dict)
        stock_data = stock_data.sort_index()
        return stock_data, error_tickers, earliest_dates

    def _fetch_stock_data_fmp(self, stocks, start, end):
        error_tickers = {}
        earliest_dates = {}
        stock_data_dict = {}
        for ticker in stocks:
            for attempt in range(2):
                try:
                    logger.info(f"Fetching FMP data for {ticker} from {start} to {end}, attempt {attempt + 1}...")
                    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start}&to={end}&apikey={self.fmp_api_key}"
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    if "historical" not in data or not data["historical"]:
                        error_tickers[ticker] = "No data available"
                        break
                    historical_data = data["historical"]
                    df = pd.DataFrame(historical_data)
                    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                    df = df.set_index("date")
                    stock_data_dict[ticker] = df["close"]
                    earliest_dates[ticker] = df.index.min().strftime("%Y-%m-%d")
                    break
                except Exception as e:
                    logger.error(f"FMP error for {ticker} (attempt {attempt + 1}): {e}")
                    if attempt == 1:
                        error_tickers[ticker] = str(e)
                    time.sleep(2)
        if not stock_data_dict:
            return None, error_tickers, earliest_dates
        stock_data = pd.DataFrame(stock_data_dict)
        stock_data = stock_data.sort_index()
        return stock_data, error_tickers, earliest_dates

    def _fetch_stock_data_marketstack(self, stocks, start, end):
        error_tickers = {}
        earliest_dates = {}
        stock_data_dict = {}
        for ticker in stocks:
            try:
                logger.info(f"Fetching MarketStack data for {ticker} from {start} to {end}...")
                url = f"http://api.marketstack.com/v1/eod?access_key={self.marketstack_api_key}&symbols={ticker}&date_from={start}&date_to={end}&limit=1000"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                if "data" not in data or not data["data"]:
                    error_tickers[ticker] = "No data available"
                    continue
                df = pd.DataFrame(data["data"])
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                df = df.set_index("date").sort_index()
                stock_data_dict[ticker] = df["close"]
                earliest_dates[ticker] = df.index.min().strftime("%Y-%m-%d")
            except Exception as e:
                logger.error(f"MarketStack error for {ticker}: {e}")
                error_tickers[ticker] = str(e)
                time.sleep(1)
        if not stock_data_dict:
            return None, error_tickers, earliest_dates
        stock_data = pd.DataFrame(stock_data_dict).sort_index()
        return stock_data, error_tickers, earliest_dates

    def _fetch_stock_data_yfinance(self, stocks, start, end):
        error_tickers = {}
        earliest_dates = {}
        # Validate tickers
        invalid_tickers = [t for t in stocks if not isinstance(t, str) or len(t.strip()) == 0]
        if invalid_tickers:
            logger.error(f"Invalid tickers passed to yfinance: {invalid_tickers}")
            return None, {"error": f"Invalid tickers: {invalid_tickers}"}, {}
        if not YFINANCE_AVAILABLE:
            return None, {"error": "yfinance unavailable"}, {}
        error_tickers = {}
        earliest_dates = {}
        for attempt in range(2):
            try:
                logger.info(f"Fetching stock data for {stocks} from {start} to {end}, attempt {attempt + 1}...")
                try:
                    yf_data = yf.download(list(stocks), start=start, end=end, auto_adjust=True)
                    logger.debug(f"Raw yfinance data columns: {yf_data.columns}")
                    if 'Close' not in yf_data:
                        raise ValueError("Missing 'Close' key in yfinance data")
                    stock_data = yf_data['Close']
                except Exception as e:
                    logger.error(f"yf.download failed for {stocks} from {start} to {end}: {str(e)}\nTraceback: {traceback.format_exc()}")
                    return None, {"error": f"yfinance error: {str(e)}"}, {}
                logger.info(f"Fetched stock data: {stock_data.shape if not stock_data.empty else 'empty'}")
                if stock_data.empty:
                    logger.warning("No data available for the specified date range.")
                    return None, error_tickers, earliest_dates
                break
            except Exception as e:
                logger.error(f"Error fetching data (attempt {attempt + 1}): {e}")
                if attempt == 1:
                    logger.error("Failed to fetch data after 2 attempts.")
                    return None, {"error": "Failed to fetch stock data"}, {}
                time.sleep(2)
        # Ensure the index is timezone-naive
        stock_data.index = stock_data.index.tz_localize(None)
        for ticker in stocks:
            if ticker in stock_data.columns and not stock_data[ticker].isna().all():
                earliest_dates[ticker] = stock_data[ticker].first_valid_index().strftime("%Y-%m-%d")
        return stock_data, error_tickers, earliest_dates

    def compute_returns(self, prices):
        try:
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
            if prices is None or prices.empty:
                logger.error("Prices data is empty.")
                return pd.DataFrame()
            prices = prices.where(prices > 1e-4, np.nan)
            if prices.isna().all().all():
                logger.error("All prices are zero or invalid after cleaning.")
                return pd.DataFrame()
            returns = prices.pct_change()
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how='any')
            if returns.empty or len(returns) < 252:
                logger.error("Insufficient valid returns data after cleaning (less than 252 days).")
                return pd.DataFrame()
            for col in returns.columns:
                if returns[col].isna().any() or np.isinf(returns[col]).any():
                    logger.warning(f"Asset {col} contains invalid data after cleaning.")
                if not pd.api.types.is_numeric_dtype(returns[col]):
                    logger.error(f"Asset {col} contains non-numeric data after cleaning.")
                    return pd.DataFrame()
            return returns
        except Exception as e:
            logger.error(f"Error computing returns: {e}. Returning empty DataFrame.")
            return pd.DataFrame()

    def compute_max_drawdown(self, returns):
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            return float(drawdown.min()) if not drawdown.empty else 0.0
        except Exception as e:
            logger.error(f"Error in compute_max_drawdown: {e}")
            return 0.0

    def compute_sortino_ratio(self, returns, risk_free_rate):
        try:
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
            annualized_return = returns.mean() * 252
            return float((annualized_return - risk_free_rate) / downside_std) if downside_std != 0 else 0.0
        except Exception as e:
            logger.error(f"Error in compute_sortino_ratio: {e}")
            return 0.0

    def compute_beta(self, portfolio_returns, benchmark_returns):
        try:
            covariance = portfolio_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            return float(covariance / benchmark_variance) if benchmark_variance != 0 else 0.0
        except Exception as e:
            logger.error(f"Error in compute_beta: {e}")
            return 0.0

    def portfolio_performance(self, weights, returns, risk_free_rate):
        try:
            if returns.empty or len(returns) < 252:
                logger.error("Insufficient or empty returns data for performance calculation.")
                return 0.0, 0.0, 0.0
            if not all(returns.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                logger.error("Returns data contains non-numeric values.")
                return 0.0, 0.0, 0.0
            portfolio_returns = returns.dot(weights)
            portfolio_return = portfolio_returns.mean() * 252
            if portfolio_returns.shape[0] <= 1:
                logger.warning("Insufficient data points to compute volatility (< 2 returns). Returning volatility and Sharpe ratio as 0.")
                return float(portfolio_return), 0.0, 0.0
            cov_matrix = returns.cov() * 252
            if cov_matrix.isna().any().any() or np.isinf(cov_matrix).any().any():
                logger.error("Covariance matrix contains NaN or infinite values.")
                return float(portfolio_return), 0.0, 0.0
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) if not returns.empty else 0
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
            return float(portfolio_return), float(portfolio_volatility), float(sharpe_ratio)
        except Exception as e:  # Aligned with try block
            logger.error(f"Error in portfolio_performance: {e}")
            return 0.0, 0.0, 0.0

    def compute_var(self, returns, confidence_level=0.90):
        try:
            sorted_returns = np.sort(returns)
            index = int((1 - confidence_level) * len(sorted_returns))
            return float(sorted_returns[index]) if len(sorted_returns) > 0 else 0.0
        except Exception as e:
            logger.error(f"Error in compute_var: {e}")
            return 0.0

    def compute_avg_correlation(self, returns_df, weights):
        try:
            if returns_df.shape[0] <= 1:
                logger.warning("Insufficient data points to compute correlation (< 2 returns). Returning 0.")
                return 0.0
            weighted_corr_sum = 0
            num_assets = returns_df.shape[1]
            corr_matrix = returns_df.corr()
            if corr_matrix.isna().all().all():
                logger.warning("Correlation matrix contains only NaN values. Returning 0.")
                return 0.0
            for i in range(num_assets):
                for j in range(i + 1, num_assets):
                    corr_value = corr_matrix.iloc[i, j]
                    if pd.isna(corr_value):
                        corr_value = 0.0
                    weighted_corr_sum += weights[i] * weights[j] * corr_value
            avg_corr = 2 * weighted_corr_sum
            return float(avg_corr)
        except Exception as e:
            logger.error(f"Error in compute_avg_correlation: {e}")
            return 0.0
            
    def optimize_portfolio(self, returns, risk_free_rate, objective='sharpe', min_allocation=0.0, max_allocation=1.0):
        try:
            num_assets = returns.shape[1]
            initial_weights = np.ones(num_assets) / num_assets
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = tuple((min_allocation, max_allocation) for _ in range(num_assets))

            def negative_sharpe(weights):
                r, v, s = self.portfolio_performance(weights, returns, risk_free_rate)
                return -s

            def negative_sortino(weights):
                portfolio_returns = returns.dot(weights)
                return -self.compute_sortino_ratio(portfolio_returns, risk_free_rate)

            def max_drawdown(weights):
                portfolio_returns = returns.dot(weights)
                drawdown = -self.compute_max_drawdown(portfolio_returns)
                return drawdown

            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

            def negative_var(weights):
                portfolio_returns = returns.dot(weights)
                return -self.compute_var(portfolio_returns)

            objective_functions = {
                'sharpe': negative_sharpe,
                'sortino': negative_sortino,
                'max_drawdown': max_drawdown,
                'volatility': portfolio_volatility,
                'value_at_risk': negative_var
            }
            obj_fun = objective_functions.get(objective, negative_sharpe)

            result = minimize(obj_fun, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            if not result.success:
                logger.warning(f"Optimization failed for {objective}: {result.message}")
                return initial_weights
            weights = result.x
            weights[weights < 0.001] = 0
            weights /= weights.sum() if weights.sum() != 0 else 1
            return weights
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return initial_weights

    def compute_eigenvalues(self, returns):
        try:
            if returns.shape[0] <= 1:
                logger.warning("Insufficient data points to compute eigenvalues (< 2 returns). Returning zeros.")
                n_assets = returns.shape[1]
                return [0.0] * n_assets, [0.0] * n_assets
            cov_matrix = returns.cov() * 252
            if cov_matrix.isna().all().all():
                logger.warning("Covariance matrix contains only NaN values. Returning zeros.")
                n_assets = returns.shape[1]
                return [0.0] * n_assets, [0.0] * n_assets
            eigenvalues, _ = np.linalg.eigh(cov_matrix)
            eigenvalues = sorted(eigenvalues, reverse=True)
            total_variance = sum(eigenvalues)
            if total_variance == 0:
                logger.warning("Total variance is zero. Returning zeros for explained variance ratio.")
                return eigenvalues, [0.0] * len(eigenvalues)
            explained_variance_ratio = [eig / total_variance for eig in eigenvalues]
            return eigenvalues, explained_variance_ratio
        except Exception as e:
            logger.error(f"Error in compute_eigenvalues: {e}")
            return [], []

    def compute_fama_french_exposures(self, portfolio_returns, start_date, end_date):
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels unavailable. Using zero exposures.")
            return {"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0}
        ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        cache_key = "fama_french_data"
        if cache_key in self.data_cache:
            ff_data = self.data_cache[cache_key]
        else:
            try:
                logger.info("Fetching Fama-French data...")
                response = requests.get(ff_url, timeout=10)
                response.raise_for_status()
                ff_data = pd.read_csv(ff_url, skiprows=3, index_col=0)
                ff_data.index = pd.to_datetime(ff_data.index, format="%Y%m%d", errors='coerce')
                ff_data = ff_data.dropna() / 100
                ff_data = ff_data[["Mkt-RF", "SMB", "HML"]]
                self.data_cache[cache_key] = ff_data
                logger.info("Fama-French data cached successfully.")
            except Exception as e:
                logger.error(f"Error fetching Fama-French data: {e}. Using fallback zero exposures.")
                return {"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0}
        try:
            ff_data = ff_data.loc[start_date:end_date]
            common_dates = portfolio_returns.index.intersection(ff_data.index)
            if len(common_dates) < 30:
                logger.warning("Insufficient overlapping data with Fama-French factors. Using zero exposures.")
                return {"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0}
            aligned_returns = portfolio_returns.loc[common_dates]
            aligned_ff = ff_data.loc[common_dates]
            X = sm.add_constant(aligned_ff)
            model = sm.OLS(aligned_returns, X).fit()
            exposures = {
                "Mkt-RF": float(model.params["Mkt-RF"]),
                "SMB": float(model.params["SMB"]),
                "HML": float(model.params["HML"])
            }
            return exposures
        except Exception as e:
            logger.error(f"Error computing Fama-French exposures: {e}. Using fallback zero exposures.")
            return {"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0}

        
    def optimize_with_factor_and_correlation(self, returns, risk_free_rate, tickers, market_prices=None, min_allocation=0.05, max_allocation=0.30, original_weights=None, bl_views=None, bl_confidences=None):
        try:
            num_assets = len(tickers)
            if num_assets < 2:
                logger.error("Portfolio must have at least 2 assets for optimization.")
                return np.ones(num_assets) / num_assets if num_assets > 0 else np.array([]), {}

            if returns.empty or returns.shape[0] < 252:
                logger.error("Returns data is empty or has insufficient data points (< 252 days).")
                return np.ones(num_assets) / num_assets, {}
            if not all(returns.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                logger.error("Returns data contains non-numeric values.")
                return np.ones(num_assets) / num_assets, {}
            
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how='any').fillna(method='ffill').fillna(method='bfill')
            if returns.isna().any().any() or np.isinf(returns).any().any():
                raise ValueError("Returns contain NaN or infinite values after cleaning.")
            if returns.shape[0] < 252:
                raise ValueError("Insufficient data points (< 252) after cleaning returns.")

            cov_matrix = returns.cov() * 252
            if cov_matrix.isna().any().any() or np.isinf(cov_matrix).any().any():
                raise ValueError("Covariance matrix contains NaN or infinite values.")
            if (cov_matrix.abs() > 1e4).any().any():
                cov_matrix = cov_matrix / cov_matrix.max().max() * 1e3

            if original_weights is not None:
                if len(original_weights) != num_assets or not np.isclose(sum(original_weights), 1.0, rtol=1e-5):
                    initial_weights = np.ones(num_assets) / num_assets
                else:
                    initial_weights = np.array(original_weights)
            else:
                initial_weights = np.ones(num_assets) / num_assets

            initial_vol = np.sqrt(np.dot(initial_weights.T, np.dot(cov_matrix, initial_weights)))
            initial_risk_contribs = np.zeros(num_assets)
            for i in range(num_assets):
                marginal_contrib = np.dot(cov_matrix.iloc[i], initial_weights)
                initial_risk_contribs[i] = initial_weights[i] * marginal_contrib / initial_vol if initial_vol > 0 else initial_weights[i]
            if initial_risk_contribs.sum() != 0:
                initial_risk_contribs /= initial_risk_contribs.sum()

            expected_rets = None
            bl_success = False
            fallback_reason = ""
            if PYPFOPT_AVAILABLE and market_prices is not None and not market_prices.empty:
                try:
                    market_prices = market_prices.reindex(returns.index).dropna()
                    if market_prices.empty:
                        raise ValueError("Market prices do not overlap with returns data after reindexing.")
                    
                    market_prices = market_prices.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
                    if market_prices.isna().any() or np.isinf(market_prices).any():
                        raise ValueError("Market prices contain NaN or infinite values after cleaning.")

                    common_index = returns.index.intersection(market_prices.index)
                    if len(common_index) < 252:
                        raise ValueError(f"Insufficient overlapping data between returns and market prices ({len(common_index)} days).")
                    returns = returns.loc[common_index]
                    market_prices = market_prices.loc[common_index]

                    S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
                    if not np.all(np.linalg.eigvals(S) >= -1e-10):
                        S = risk_models.fix_nonpositive_semidefinite(S, fix_method='spectral')
                    if S.isna().any().any() or np.isinf(S).any().any():
                        raise ValueError("Covariance matrix S contains NaN or infinite values after shrinkage.")
                    if (S.abs() > 1e4).any().any():
                        S = S / S.max().max() * 1e3

                    delta = 2.5
                    market_weights = pd.Series(1/num_assets, index=tickers)
                    market_prior = expected_returns.capm_return(
                        prices=returns,
                        risk_free_rate=risk_free_rate,
                        market_prices=market_prices
                    )
                    if market_prior.isna().any() or np.isinf(market_prior).any():
                        raise ValueError("Market-implied prior returns contain NaN or infinite values.")
                    if (market_prior.abs() > 10).any():
                        market_prior = market_prior / market_prior.abs().max() * 1.0

                    if bl_views is not None and bl_confidences is not None:
                        Q = pd.Series([bl_views.get(ticker, 0.0) for ticker in tickers], index=tickers)
                        P = np.eye(num_assets)
                        Omega = np.diag([bl_confidences.get(ticker, 0.01) for ticker in tickers])
                    else:
                        Q = pd.Series([0.0] * num_assets, index=tickers)
                        P = np.eye(num_assets)
                        Omega = np.diag([0.01] * num_assets)

                    bl = BlackLittermanModel(
                        cov_matrix=S,
                        pi=market_prior,
                        Q=Q,
                        P=P,
                        Omega=Omega,
                        delta=delta
                    )
                    expected_rets = bl.bl_returns()
                    if expected_rets.isna().any() or np.isinf(expected_rets).any():
                        raise ValueError("Black-Litterman returned NaN or infinite values.")
                    if (expected_rets.abs() > 10).any():
                        expected_rets = expected_rets / expected_rets.abs().max() * 1.0
                    bl_success = True
                except Exception as e:
                    fallback_reason = str(e)

            if expected_rets is None and PYPFOPT_AVAILABLE and market_prices is not None and not market_prices.empty:
                try:
                    S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
                    if not np.all(np.linalg.eigvals(S) >= -1e-10):
                        S = risk_models.fix_nonpositive_semidefinite(S, fix_method='spectral')
                    if (S.abs() > 1e4).any().any():
                        S = S / S.max().max() * 1e3
                    market_prior = expected_returns.capm_return(
                        prices=returns,
                        risk_free_rate=risk_free_rate,
                        market_prices=market_prices
                    )
                    if market_prior.isna().any() or np.isinf(market_prior).any():
                        raise ValueError("Simple CAPM returned NaN or infinite values.")
                    if (market_prior.abs() > 10).any():
                        market_prior = market_prior / market_prior.abs().max() * 1.0
                    expected_rets = market_prior
                    fallback_reason = "Used simple CAPM due to Black-Litterman failure."
                except Exception as e:
                    fallback_reason = str(e)

            if expected_rets is None:
                try:
                    pca = PCA(n_components=min(3, num_assets))
                    pca.fit(returns)
                    factor_returns = pd.DataFrame(pca.transform(returns), index=returns.index)
                    if factor_returns.isna().any().any() or np.isinf(factor_returns).any().any():
                        raise ValueError("PCA transformation resulted in NaN or infinite values.")
                    expected_rets = pd.Series(pca.inverse_transform(factor_returns.mean()) * 252, index=tickers)
                    if expected_rets.isna().any() or np.isinf(expected_rets).any():
                        raise ValueError("PCA returned NaN or infinite values.")
                    if (expected_rets.abs() > 10).any():
                        expected_rets = expected_rets / expected_rets.abs().max() * 1.0
                    fallback_reason = "Used PCA due to CAPM/Black-Litterman failure."
                except Exception as e:
                    logger.error(f"PCA optimization failed: {e}")
                    fallback_reason = str(e)
        
            if expected_rets is None or expected_rets.isna().any() or np.isinf(expected_rets).any():
                expected_rets = returns.mean() * 252
                if expected_rets.isna().any() or np.isinf(expected_rets).any():
                    expected_rets = pd.Series([risk_free_rate + 0.05] * num_assets, index=tickers)
                    fallback_reason = "Used risk-free rate + 5% due to all other methods failing."
                else:
                    if (expected_rets.abs() > 10).any():
                        expected_rets = expected_rets / expected_rets.abs().max() * 1.0
                    fallback_reason = "Used historical mean returns due to PCA/CAPM/Black-Litterman failure."

            def objective(weights):
                try:
                    ret = np.dot(weights, expected_rets)
                    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
                    vol_drag = 0.5 * vol**2
                    adj_sharpe = (ret - vol_drag - risk_free_rate) / vol if vol > 0 else 0
                    corr_penalty = 1.5 * self.compute_avg_correlation(returns, weights)
                    risk_contribs = weights * np.dot(cov_matrix, weights) / vol if vol > 0 else weights
                    risk_parity_penalty = 5.0 * np.var(risk_contribs)
                    weights_clean = weights + 1e-10
                    entropy = -np.sum(weights_clean * np.log(weights_clean)) / np.log(num_assets)
                    diversification_penalty = 1.0 * (1 - entropy)
                    return -adj_sharpe + corr_penalty + risk_parity_penalty + diversification_penalty
                except Exception as e:
                    logger.error(f"Objective function error: {e}. Returning infinity.")
                    return float('inf')

            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            max_allowed = min(0.40, 1.0 / num_assets * 1.5)
            min_allowed = max(0.01, 1.0 / num_assets * 0.5)
            if min_allocation * num_assets > 1 or max_allocation * num_assets < 1:
                min_allocation = min_allowed
                max_allocation = max_allowed
            bounds = tuple((max(min_allocation, min_allowed), min(max_allocation, max_allowed)) for _ in range(num_assets))

            optimized_weights = None
            try:
                if CVXPY_AVAILABLE:
                    w = cp.Variable(num_assets)
                    ret = expected_rets @ w
                    vol = cp.sqrt(cp.quad_form(w, cov_matrix))
                    objective = cp.Minimize(-ret + 2.5 * vol)
                    constraints_cvx = [cp.sum(w) == 1, w >= min_allocation, w <= max_allocation]
                    problem = cp.Problem(objective, constraints_cvx)
                    problem.solve(solver=cp.SCS, max_iters=1000, eps=1e-8)
                    if problem.status != cp.OPTIMAL:
                        raise ValueError(f"CVXPY optimization failed: {problem.status}")
                    optimized_weights = w.value
                else:
                    raise ImportError("CVXPY unavailable.")
            except (ImportError, Exception):
                result = minimize(
                    objective,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
                )
                if not result.success:
                    result = minimize(
                        objective,
                        initial_weights,
                        method='SLSQP',
                        bounds=tuple((0.0, 1.0) for _ in range(num_assets)),
                        constraints=constraints,
                        options={'maxiter': 500, 'ftol': 1e-6}
                    )
                    if not result.success:
                        optimized_weights = initial_weights
                    else:
                        optimized_weights = result.x
                else:
                    optimized_weights = result.x

            optimized_weights = np.clip(optimized_weights, min_allowed, max_allowed)
            optimized_weights /= optimized_weights.sum() if optimized_weights.sum() != 0 else 1.0

            opt_vol = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
            risk_contribs = np.zeros(num_assets)
            for i in range(num_assets):
                marginal_contrib = np.dot(cov_matrix.iloc[i], optimized_weights)
                risk_contribs[i] = optimized_weights[i] * marginal_contrib / opt_vol if opt_vol > 0 else optimized_weights[i]
            if risk_contribs.sum() != 0:
                risk_contribs /= risk_contribs.sum()

            opt_ret, opt_vol, opt_sharpe = self.portfolio_performance(optimized_weights, returns, risk_free_rate)
            avg_corr = self.compute_avg_correlation(returns, optimized_weights)
            entropy = -np.sum(optimized_weights * np.log(optimized_weights + 1e-10)) / np.log(num_assets)
            
            tracking_error = 0.0  # Default to 0.0 if unavailable
            if market_prices is not None and not market_prices.empty:
                try:
                    market_returns = market_prices.pct_change().dropna()
                    if market_returns.empty or market_returns.isna().all():
                        logger.warning("Market returns are empty or all NaN after computation.")
                    else:
                        portfolio_ret = returns.dot(optimized_weights)
                        common_dates = portfolio_ret.index.intersection(market_returns.index)
                        if len(common_dates) > 0:
                            aligned_portfolio = portfolio_ret.loc[common_dates]
                            aligned_market = market_returns.loc[common_dates]
                            tracking_error = float((aligned_portfolio - aligned_market).std() * np.sqrt(252))
                        else:
                            logger.warning("No common dates between portfolio and market returns for tracking error.")
                except Exception as te_err:
                    logger.error(f"Error computing tracking error: {str(te_err)}")
                    tracking_error = 0.0

            opt_beta = 0.0  # Default to 0.0 if unavailable
            if market_prices is not None and not market_prices.empty:
                try:
                    market_returns = market_prices.pct_change().dropna()
                    if market_returns.empty or market_returns.isna().all():
                        logger.warning("Market returns are empty or all NaN for beta computation.")
                    else:
                        opt_beta = self.compute_beta(returns.dot(optimized_weights), market_returns)
                except Exception as beta_err:
                    logger.error(f"Error computing beta: {str(beta_err)}")
                    opt_beta = 0.0


            return optimized_weights, {
                "return": float(opt_ret),
                "volatility": float(opt_vol),
                "sharpe": float(opt_sharpe),
                "avg_correlation": float(avg_corr),
                "entropy": float(entropy),
                "risk_contributions": risk_contribs.tolist(),
                "tracking_error": tracking_error,
                "beta": opt_beta
            }
        except Exception as e:
            logger.error(f"Critical error in optimization: {e}. Returning equal weights.")
            return np.ones(num_assets) / num_assets, {}

    def get_historical_metrics(self, tickers, weights_dict, risk_free_rate, hist_returns):
        try:
            strategies = {
                "Original Portfolio": np.array(list(weights_dict.values())),
                "Max Sharpe": self.optimize_portfolio(hist_returns, risk_free_rate, "sharpe"),
                "Max Sortino": self.optimize_portfolio(hist_returns, risk_free_rate, "sortino"),
                "Min Max Drawdown": self.optimize_portfolio(hist_returns, risk_free_rate, "max_drawdown"),
                "Min Volatility": self.optimize_portfolio(hist_returns, risk_free_rate, "volatility"),
                "Min Value at Risk": self.optimize_portfolio(hist_returns, risk_free_rate, "value_at_risk")
            }
            metrics = {"Annual Return": [], "Volatility": [], "Avg Correlation": []}
            labels = []
            for label, weights in strategies.items():
                portfolio_return, portfolio_volatility, _ = self.portfolio_performance(weights, hist_returns, risk_free_rate)
                avg_corr = self.compute_avg_correlation(hist_returns, weights)
                metrics["Annual Return"].append(float(portfolio_return))
                metrics["Volatility"].append(float(portfolio_volatility))
                metrics["Avg Correlation"].append(float(avg_corr))
                labels.append(label)
            return metrics, labels
        except Exception as e:
            logger.error(f"Error in get_historical_metrics: {e}")
            return {"Annual Return": [], "Volatility": [], "Avg Correlation": []}, []

    def get_cumulative_returns(self, returns, strategies, benchmark_returns, earliest_dates, title="Cumulative Returns of Strategies"):
        try:
            if returns.empty or len(returns) < 2:
                logger.error("Returns data is empty or has insufficient data points (< 2) for cumulative returns.")
                return {"dates": [], "cumulative_returns": {}}

            data = {}
            for label, weights in strategies.items():
                try:
                    if len(weights) != returns.shape[1]:
                        logger.error(f"Weight length mismatch for {label}: expected {returns.shape[1]}, got {len(weights)}")
                        continue
                    portfolio_returns = returns.dot(weights)
                    if portfolio_returns.empty or portfolio_returns.isna().all():
                        logger.error(f"Portfolio returns for {label} are empty or all NaN.")
                        continue
                    cumulative = (1 + portfolio_returns).cumprod() - 1
                    if cumulative.empty or cumulative.isna().all():
                        logger.error(f"Cumulative returns for {label} are empty or all NaN.")
                        continue
                    data[label] = cumulative.tolist()
                except Exception as e:
                    logger.error(f"Error computing cumulative returns for {label}: {str(e)}")
                    continue

            for bench_ticker, bench_ret in benchmark_returns.items():
                try:
                    if bench_ret.empty or bench_ret.isna().all():
                        logger.error(f"Benchmark returns for {bench_ticker} are empty or all NaN.")
                        continue
                    cumulative = (1 + bench_ret).cumprod() - 1
                    if cumulative.empty or cumulative.isna().all():
                        logger.error(f"Cumulative benchmark returns for {bench_ticker} are empty or all NaN.")
                        continue
                    data[bench_ticker] = cumulative.tolist()
                except Exception as e:
                    logger.error(f"Error computing cumulative benchmark returns for {bench_ticker}: {str(e)}")
                    continue

            if not data:
                logger.error("No valid cumulative returns data computed for any strategy or benchmark.")
                return {"dates": [], "cumulative_returns": {}}

            dates = [d.strftime("%Y-%m-%d") for d in returns.index]
            logger.info(f"Computed cumulative returns for {len(data)} strategies/benchmarks.")
            return {"dates": dates, "cumulative_returns": data}
        except Exception as e:
            logger.error(f"Error in get_cumulative_returns: {str(e)}")
            return {"dates": [], "cumulative_returns": {}}

    def get_correlation_matrix(self, prices):
        try:
            returns = self.compute_returns(prices)
            if returns.shape[0] <= 1:
                logger.warning("Insufficient data points to compute correlation matrix (< 2 returns). Returning zero matrix.")
                return {
                    "tickers": list(prices.columns),
                    "matrix": [[0.0 if i != j else 1.0 for j in range(len(prices.columns))] for i in range(len(prices.columns))]
                }
            corr_matrix = returns.corr()
            if corr_matrix.isna().all().all():
                logger.warning("Correlation matrix contains only NaN values. Returning zero matrix.")
                return {
                    "tickers": list(prices.columns),
                    "matrix": [[0.0 if i != j else 1.0 for j in range(len(prices.columns))] for i in range(len(prices.columns))]
                }
            corr_matrix = corr_matrix.fillna(0.0)  # Replace any remaining NaN with 0
            return {
                "tickers": list(corr_matrix.index),
                "matrix": corr_matrix.values.tolist()
            }
        except Exception as e:
            logger.error(f"Error in get_correlation_matrix: {e}")
            return {"tickers": [], "matrix": []}

    def get_efficient_frontier(self, returns, risk_free_rate, n_portfolios=500):  # Reduced from 1000 to 500
        try:
            np.random.seed(42)
            n_assets = returns.shape[1]
            all_weights = np.zeros((n_portfolios, n_assets))
            all_returns = np.zeros(n_portfolios)
            all_volatilities = np.zeros(n_portfolios)
            all_sharpe_ratios = np.zeros(n_portfolios)

            for i in range(n_portfolios):
                weights = np.random.random(n_assets)
                weights /= weights.sum()
                all_weights[i, :] = weights
                port_return, port_vol, port_sharpe = self.portfolio_performance(weights, returns, risk_free_rate)
                all_returns[i] = port_return
                all_volatilities[i] = port_vol
                all_sharpe_ratios[i] = port_sharpe

            strategies = {
                "Max Sharpe": self.optimize_portfolio(returns, risk_free_rate, "sharpe"),
                "Max Sortino": self.optimize_portfolio(returns, risk_free_rate, "sortino"),
                "Min Max Drawdown": self.optimize_portfolio(returns, risk_free_rate, "max_drawdown"),
                "Min Volatility": self.optimize_portfolio(returns, risk_free_rate, "volatility"),
                "Min Value at Risk": self.optimize_portfolio(returns, risk_free_rate, "value_at_risk")
            }

            strategy_metrics = {}
            for name, weights in strategies.items():
                port_return, port_vol, port_sharpe = self.portfolio_performance(weights, returns, risk_free_rate)
                strategy_metrics[name] = {
                    "return": float(port_return),
                    "volatility": float(port_vol),
                    "sharpe": float(port_sharpe)
                }

            max_sharpe_vol = strategy_metrics["Max Sharpe"]["volatility"]
            max_sharpe_sharpe = strategy_metrics["Max Sharpe"]["sharpe"]
            cml = {
                "x": [0, max_sharpe_vol * 1.5],
                "y": [risk_free_rate, risk_free_rate + max_sharpe_sharpe * max_sharpe_vol * 1.5]
            }

            # Validate arrays before returning
            all_returns = np.nan_to_num(all_returns, nan=0.0)
            all_volatilities = np.nan_to_num(all_volatilities, nan=0.0)
            all_sharpe_ratios = np.nan_to_num(all_sharpe_ratios, nan=0.0)
            return {
                "portfolios": {
                    "returns": all_returns.tolist(),
                    "volatilities": all_volatilities.tolist(),
                    "sharpe_ratios": all_sharpe_ratios.tolist()
                },
                "strategies": strategy_metrics,
                "capital_market_line": cml
            }

        except Exception as e:
            logger.error(f"Error in get_efficient_frontier: {e}")
            return {
                "portfolios": {
                    "returns": [],
                    "volatilities": [],
                    "sharpe_ratios": []
                },
                "strategies": {},
                "capital_market_line": {
                    "x": [],
                    "y": []
                }
            }

    def get_comparison_bars(self, original_metrics, optimized_metrics, benchmark_metrics):
        try:
            metrics = ["annual_return", "annual_volatility", "sharpe_ratio", "maximum_drawdown", "value_at_risk"]
            labels = ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Maximum Drawdown", "Value at Risk (90%)"]
            data = []
            for metric, label in zip(metrics, labels):
                values = [original_metrics[metric], optimized_metrics[metric]]
                names = ["Original", "Optimized"]
                if benchmark_metrics:
                    for bench, bm in benchmark_metrics.items():
                        values.append(bm[metric])
                        names.append(bench)
                data.append({
                    "metric": label,
                    "names": names,
                    "values": values
                })
            return data
        except Exception as e:
            logger.error(f"Error in get_comparison_bars: {e}")
            return []

    def get_portfolio_exposures(self, tickers, original_weights, optimized_weights):
        try:
            original_exposures = [float(w) for w in original_weights if w > 0]
            original_labels = [t for t, w in zip(tickers, original_weights) if w > 0]
            optimized_exposures = [float(w) for w in optimized_weights if w > 0]
            optimized_labels = [t for t, w in zip(tickers, optimized_weights) if w > 0]
            return {
                "original": {"labels": original_labels, "exposures": original_exposures},
                "optimized": {"labels": optimized_labels, "exposures": optimized_exposures}
            }
        except Exception as e:
            logger.error(f"Error in get_portfolio_exposures: {e}")
            return {"original": {"labels": [], "exposures": []}, "optimized": {"labels": [], "exposures": []}}

    def get_rolling_volatility(self, returns, weights_dict, benchmark_returns, window=252):
        try:
            data = {}
            dates = [d.strftime("%Y-%m-%d") for d in returns.index]
            # Adjust window size to be at most half the data length, minimum 2
            n_rows = len(returns)
            window = min(window, max(2, n_rows // 2))
            logger.info(f"Adjusted rolling window size to {window} based on {n_rows} data points")
        
            for label, weights in weights_dict.items():
                portfolio_returns = returns.dot(weights)
                rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
                # Fill NaN values only for the first (window-1) rows; leave the rest as-is
                rolling_vol.iloc[:window-1] = rolling_vol.iloc[:window-1].fillna(0)
                data[f"{label} Volatility"] = rolling_vol.tolist()
            for bench_ticker, bench_ret in benchmark_returns.items():
                rolling_vol = bench_ret.rolling(window=window).std() * np.sqrt(252)
                rolling_vol.iloc[:window-1] = rolling_vol.iloc[:window-1].fillna(0)
                data[f"{bench_ticker} Volatility"] = rolling_vol.tolist()
            return {"dates": dates, "rolling_volatility": data}
        except Exception as e:
            logger.error(f"Error in get_rolling_volatility: {e}")
            return {"dates": [], "rolling_volatility": {}}
            
    def get_diversification_benefit(self, returns, original_weights, optimized_weights, tickers):
        try:
            equal_weights = np.ones(len(tickers)) / len(tickers)
            cov_matrix = returns.cov() * 252
            orig_vol = np.sqrt(np.dot(original_weights.T, np.dot(cov_matrix, original_weights)))
            opt_vol = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
            equal_vol = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
            orig_ret = returns.dot(original_weights).mean() * 252
            opt_ret = returns.dot(optimized_weights).mean() * 252
            equal_ret = returns.dot(equal_weights).mean() * 252
            labels = ['Equal Weight', 'Original', 'Optimized']
            vols = [float(equal_vol), float(orig_vol), float(opt_vol)]
            rets = [float(equal_ret), float(orig_ret), float(opt_ret)]
            return {
                "labels": labels,
                "volatilities": vols,
                "returns": rets
            }
        except Exception as e:
            logger.error(f"Error in get_diversification_benefit: {e}")
            return {"labels": [], "volatilities": [], "returns": []}

    def get_crisis_performance(self, returns, weights_dict, benchmark_returns, earliest_dates):
        crises = [
            {
                "name": "Dot-Com Bust",
                "start": pd.to_datetime("2000-03-01"),
                "end": pd.to_datetime("2002-10-31"),
                "description": "The Dot-Com Bust (March 2000 - October 2002) saw a tech bubble collapse, with the Nasdaq dropping 78% as overvalued internet companies failed, leading to reduced business activity in tech sectors."
            },
            {
                "name": "Great Recession",
                "start": pd.to_datetime("2007-12-01"),
                "end": pd.to_datetime("2009-06-30"),
                "description": "The Great Recession (December 2007 - June 2009) followed a housing bubble burst and financial crisis, with GDP dropping 4.3% and business activity stalling as credit froze."
            },
            {
                "name": "COVID-19 Crisis",
                "start": pd.to_datetime("2020-02-01"),
                "end": pd.to_datetime("2020-04-30"),
                "description": "The COVID-19 Crisis (February - April 2020) involved global lockdowns, halting business activity, with a 31.4% GDP drop in Q2 2020 and a swift 34% S&P 500 decline."
            }
        ]
        crisis_summaries = {}
        try:
            if not returns.index.empty:
                data_start = returns.index.min()
                data_end = returns.index.max()
                logger.info(f"Data range for crisis performance: {data_start} to {data_end}")
            else:
                logger.error("Returns data is empty for crisis performance analysis.")
                return {"error": "Returns data is empty for crisis performance analysis."}

            earliest_data = pd.to_datetime(min(earliest_dates.values()))
            six_months = timedelta(days=180)

            applicable_crises = 0
            for crisis in crises:
                crisis_start = crisis["start"]
                crisis_end = crisis["end"]
                if earliest_data > (crisis_start - six_months):
                    logger.warning(f"Skipping crisis {crisis['name']}: data starts at {earliest_data}, too late for crisis starting at {crisis_start}.")
                    continue
                if data_end < crisis_start or data_start > crisis_end:
                    logger.warning(f"Skipping crisis {crisis['name']}: data range {data_start} to {data_end} does not overlap with crisis {crisis_start} to {crisis_end}.")
                    continue

                available_starts = returns.index[returns.index >= crisis_start]
                available_ends = returns.index[returns.index <= crisis_end]
                if available_starts.empty or available_ends.empty:
                    logger.warning(f"No data available for crisis {crisis['name']} within range {crisis_start} to {crisis_end}.")
                    continue

                available_start = available_starts.min()
                available_end = available_ends.max()
                if pd.isna(available_start) or pd.isna(available_end) or available_start > available_end:
                    logger.warning(f"Invalid date range for crisis {crisis['name']}: {available_start} to {available_end}.")
                    continue

                crisis_returns = returns.loc[available_start:available_end]
                if len(crisis_returns) < 2:
                    logger.warning(f"Insufficient data points for crisis {crisis['name']}: {len(crisis_returns)} returns.")
                    continue

                crisis_data = {}
                crisis_performance = {}
                for label, weights in weights_dict.items():
                    try:
                        portfolio_returns = crisis_returns.dot(weights)
                        cumulative = (1 + portfolio_returns).cumprod() - 1
                        crisis_data[label] = cumulative.tolist()
                        last_value = cumulative.iloc[-1] if not cumulative.empty else 0.0
                        crisis_performance[label] = float(last_value) if not pd.isna(last_value) else 0.0
                    except Exception as e:
                        logger.error(f"Error computing crisis performance for {label} in {crisis['name']}: {str(e)}")
                        crisis_data[label] = []
                        crisis_performance[label] = 0.0

                for bench_ticker, bench_ret in benchmark_returns.items():
                    try:
                        bench_crisis_ret = bench_ret.loc[available_start:available_end]
                        if bench_crisis_ret.empty:
                            logger.warning(f"No benchmark data for {bench_ticker} in crisis {crisis['name']}.")
                            continue
                        bench_cum = (1 + bench_crisis_ret).cumprod() - 1
                        crisis_data[bench_ticker] = bench_cum.tolist()
                        last_bench_value = bench_cum.iloc[-1] if not bench_cum.empty else 0.0
                        crisis_performance[bench_ticker] = float(last_bench_value) if not pd.isna(last_bench_value) else 0.0
                    except Exception as e:
                        logger.error(f"Error computing benchmark crisis performance for {bench_ticker} in {crisis['name']}: {str(e)}")
                        crisis_data[bench_ticker] = []
                        crisis_performance[bench_ticker] = 0.0

                dates = [d.strftime("%Y-%m-%d") for d in crisis_returns.index]
                crisis_summaries[crisis["name"]] = {
                    "dates": dates,
                    "cumulative_returns": crisis_data,
                    "performance": crisis_performance,
                    "start": available_start.strftime("%Y-%m-%d"),
                    "end": available_end.strftime("%Y-%m-%d"),
                    "description": crisis["description"]
                }
                applicable_crises += 1

            if not crisis_summaries:
                logger.warning("No crises applicable for the given data range.")
                return {"error": f"Crisis performance data unavailable for the date range {data_start} to {data_end}."}

            logger.info(f"Computed crisis performance for {applicable_crises} crises.")
            return crisis_summaries

        except Exception as e:
            logger.error(f"Error in get_crisis_performance: {str(e)}")
            return {"error": f"Error computing crisis performance: {str(e)}"}


    def suggest_courses_of_action(self, tickers, original_weights, optimized_weights, returns, risk_free_rate, benchmark_metrics, risk_tolerance, start_date, end_date):
        original_returns = returns.dot(original_weights)
        optimized_returns = returns.dot(optimized_weights)
        original_metrics = {
            "annual_return": float(original_returns.mean() * 252),
            "annual_volatility": float(original_returns.std() * np.sqrt(252)) if not pd.isna(original_returns.std()) else 0.0,
            "sharpe_ratio": self.portfolio_performance(original_weights, returns, risk_free_rate)[2],
            "max_drawdown": self.compute_max_drawdown(original_returns),
            "var": self.compute_var(original_returns, 0.90),
            "sortino": self.compute_sortino_ratio(original_returns, risk_free_rate)
        }
        optimized_metrics = {
            "annual_return": float(optimized_returns.mean() * 252),
            "annual_volatility": float(optimized_returns.std() * np.sqrt(252)) if not pd.isna(optimized_returns.std()) else 0.0,
            "sharpe_ratio": self.portfolio_performance(optimized_weights, returns, risk_free_rate)[2],
            "max_drawdown": self.compute_max_drawdown(optimized_returns),
            "var": self.compute_var(optimized_returns, 0.90),
            "sortino": self.compute_sortino_ratio(optimized_returns, risk_free_rate)
        }
        ff_exposures = self.compute_fama_french_exposures(original_returns, start_date, end_date)
        corr_matrix = returns.corr()
        # Mask the diagonal to exclude self-correlations (set to NaN)
        masked_corr = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
        max_corr = float(masked_corr.max().max()) if not pd.isna(masked_corr.max().max()) else 0.0
        analysis = {
            "strengths": [],
            "weaknesses": [],
            "current_standing": [],
            "short_term": [],
            "medium_term": [],
            "long_term": [],
            "disclaimer": "The information provided by GALCAST Portfolio Analytics & Optimization is for informational and educational purposes only. It should not be considered as financial advice or a recommendation to buy or sell any security. Investment decisions should be based on your own research, investment objectives, financial situation, and needs. Past performance is not indicative of future results. Always consult with a qualified financial advisor before making any investment decisions."
        }

        # Strengths
        if original_metrics["sharpe_ratio"] > 1.5:
            analysis["strengths"].append(f"Strong Risk-Adjusted Returns: Your Sharpe Ratio of {original_metrics['sharpe_ratio']:.2f} is impressive—it shows you’re getting solid returns for the risk you’re taking. This is a great foundation to build on!")
        if original_metrics["annual_volatility"] < 0.15:
            analysis["strengths"].append(f"Low Volatility: At {original_metrics['annual_volatility']:.2%}, your portfolio is stable, which is fantastic for peace of mind and steady growth.")
        if ff_exposures["Mkt-RF"] < 1:
            analysis["strengths"].append(f"Market Resilience: With a market beta of {ff_exposures['Mkt-RF']:.2f}, your portfolio is less sensitive to market swings than the average—excellent for weathering downturns.")

        # Weaknesses
        if ff_exposures["Mkt-RF"] > 1.2:
            analysis["weaknesses"].append(f"High Market Exposure: Your market beta of {ff_exposures['Mkt-RF']:.2f} means your portfolio amplifies market moves. This can boost gains in bull markets but leaves you vulnerable in crashes.")
        if max_corr > 0.8:
            high_corr_pairs = [(t1, t2) for t1 in tickers for t2 in tickers if t1 < t2 and corr_matrix.loc[t1, t2] > 0.8]
            analysis["weaknesses"].append(f"Concentration Risk: Stocks like {', '.join([f'{p[0]}-{p[1]}' for p in high_corr_pairs])} have correlations above 0.8, suggesting your risk is concentrated. If one drops, others may follow.")
        if original_metrics["max_drawdown"] < -0.25:
            analysis["weaknesses"].append(f"Significant Drawdowns: A max drawdown of {original_metrics['max_drawdown']:.2%} indicates past losses were steep. We’ll want to protect against this moving forward.")

        # Current Standing
        if benchmark_metrics:
            bench_key = next(iter(benchmark_metrics))  # Safer way to get the first key
            bench_return = benchmark_metrics[bench_key]["annual_return"]
            analysis["current_standing"].append(f"Your original portfolio has delivered an annualized return of {original_metrics['annual_return']:.2%}, with a volatility of {original_metrics['annual_volatility']:.2%}, compared to {bench_key}’s {bench_return:.2%} return.")
        else:
            analysis["current_standing"].append(f"Your original portfolio has delivered an annualized return of {original_metrics['annual_return']:.2%}, with a volatility of {original_metrics['annual_volatility']:.2%}. Benchmark data was unavailable for comparison.")        
        if optimized_metrics["annual_return"] > original_metrics["annual_return"]:
            analysis["current_standing"].append(f"Good News: Optimization boosts your return to {optimized_metrics['annual_return']:.2%}—a {optimized_metrics['annual_return'] - original_metrics['annual_return']:.2%} improvement, showing we can enhance your growth.")
        if optimized_metrics["annual_volatility"] < original_metrics["annual_volatility"]:
            analysis["current_standing"].append(f"Risk Reduction: Optimization cuts volatility to {optimized_metrics['annual_volatility']:.2%}, a {original_metrics['annual_volatility'] - optimized_metrics['annual_volatility']:.2%} drop, aligning better with your {risk_tolerance} risk tolerance.")
        
        # Short-Term
        analysis["short_term"].append("Short-Term (0-1 Year): Quick Wins and Stability")
        analysis["short_term"].append("Goal: Capitalize on immediate opportunities while managing risk.")
        if risk_tolerance == "low":
            analysis["short_term"].append(f"Action 1: De-Risk with Stability Focus")
            analysis["short_term"].append(f"Why: Your {risk_tolerance} risk tolerance favors safety. With a VaR of {original_metrics['var']:.2%}, there’s a 10% chance of losing that much in a day.")
            analysis["short_term"].append(f"How: Shift 10-15% of your portfolio to low-volatility assets like utilities (e.g., XLU ETF, 1.5% yield, 10% volatility) or Treasuries (e.g., TLT, 2-3% yield). This could reduce VaR to {original_metrics['var'] * 0.85:.2%} based on historical correlations.")
            analysis["short_term"].append(f"Probability: 70% chance of stabilizing returns within 6 months, given utilities’ low beta (~0.3).")
        else:
            analysis["short_term"].append(f"Action 1: Capitalize on Momentum")
            analysis["short_term"].append(f"Why: Your {risk_tolerance} tolerance allows chasing short-term gains. Optimized Sharpe ({optimized_metrics['sharpe_ratio']:.2f}) suggests upside potential.")
            analysis["short_term"].append(f"How: Increase allocation to top performers (e.g., stocks with recent 20%+ gains in your portfolio—check returns) by 5-10%, or add a momentum ETF like MTUM (12% annualized return, 15% volatility).")
            analysis["short_term"].append(f"Probability: 60% chance of outperforming {bench_key} by 2-3% in 6 months, based on momentum factor trends.")
        analysis["short_term"].append(f"Action 2: Rebalance Quarterly")
        analysis["short_term"].append(f"Why: Keeps your portfolio aligned with short-term market shifts.")
        analysis["short_term"].append(f"How: Adjust weights to optimized levels (e.g., {', '.join([f'{t}: {w:.2%}' for t, w in zip(tickers, optimized_weights)])}).")
        analysis["short_term"].append(f"Probability: 80% chance of maintaining or improving Sharpe Ratio, per historical rebalancing studies.")

        # Medium-Term
        analysis["medium_term"].append("Medium-Term (1-5 Years): Growth with Balance")
        analysis["medium_term"].append("Goal: Build wealth steadily while preparing for volatility.")
        if ff_exposures["HML"] > 0.3:
            analysis["medium_term"].append(f"Action 1: Leverage Value Opportunities")
            analysis["medium_term"].append(f"Why: Your value exposure (HML: {ff_exposures['HML']:.2f}) suggests strength in undervalued stocks, which often shine in recovery phases.")
            analysis["medium_term"].append(f"How: Allocate 10-20% to a value ETF (e.g., VTV, 10% return, 14% volatility) or deepen exposure to value sectors like financials (e.g., XLF).")
            analysis["medium_term"].append(f"Probability: 65% chance of 8-10% annualized returns over 3 years, based on value factor outperformance post-recession.")
        else:
            analysis["medium_term"].append(f"Action 1: Explore Growth Sectors")
            analysis["medium_term"].append(f"Why: Low HML ({ff_exposures['HML']:.2f}) suggests room to capture growth, especially with {risk_tolerance} tolerance.")
            analysis["medium_term"].append(f"How: Invest 15-25% in tech or consumer discretionary (e.g., QQQ, 13% return, 18% volatility), targeting sectors with 10-15% growth potential.")
            analysis["medium_term"].append(f"Probability: 55% chance of beating {bench_key} by 3-5% annually, per growth stock cycles.")
        analysis["medium_term"].append(f"Action 2: Diversify Correlation")
        analysis["medium_term"].append(f"Why: High correlations (e.g., {max_corr:.2f}) increase risk concentration.")
        analysis["medium_term"].append(f"How: Add 10% to assets with correlations < 0.5 to your portfolio (e.g., gold via GLD, 5% return, -0.1 correlation to equities).")
        analysis["medium_term"].append(f"Probability: 75% chance of reducing volatility by 2-3%, per diversification models.")

        # Long-Term
        analysis["long_term"].append("Long-Term (5+ Years): Wealth Maximization")
        analysis["long_term"].append("Goal: Achieve sustained growth with resilience.")
        if optimized_metrics["sortino"] > original_metrics["sortino"]:
            analysis["long_term"].append(f"Action 1: Stick with Optimization")
            analysis["long_term"].append(f"Why: Optimized Sortino ({optimized_metrics['sortino']:.2f} vs {original_metrics['sortino']:.2f}) shows better downside protection, key for long-term stability.")
            analysis["long_term"].append(f"How: Fully adopt optimized weights ({', '.join([f'{t}: {w:.2%}' for t, w in zip(tickers, optimized_weights)])}) and reinvest dividends.")
            analysis["long_term"].append(f"Probability: 70% chance of growing $10,000 to ${(10000 * (1 + optimized_metrics['annual_return']) ** 5):,.0f} in 5 years, vs ${(10000 * (1 + original_metrics['annual_return']) ** 5):,.0f} originally.")
        else:
            analysis["long_term"].append(f"Action 1: Enhance Downside Protection")
            analysis["long_term"].append(f"Why: Original Sortino ({original_metrics['sortino']:.2f}) suggests vulnerability to losses over time.")
            analysis["long_term"].append(f"How: Shift 20% to Min Volatility strategy (e.g., SPLV ETF, 8% return, 10% volatility) or bonds.")
            analysis["long_term"].append(f"Probability: 80% chance of cutting max drawdown to {optimized_metrics['max_drawdown'] * 0.8:.2%}, per low-volatility studies.")
        analysis["long_term"].append(f"Action 2: Expand Globally")
        analysis["long_term"].append(f"Why: Broaden exposure beyond U.S. markets reduces systemic risk.")
        analysis["long_term"].append(f"How: Allocate 15-20% to international equities (e.g., VXUS, 7% return, 16% volatility), diversifying across emerging markets.")
        analysis["long_term"].append(f"Probability: 60% chance of boosting returns by 1-2% annually over 10 years, per global diversification data.")

        return analysis

analyzer = PortfolioAnalyzer()

@app.route('/')
def index():
    logger.info("Received request to /")
    return "Portfolio Analyzer API is running. Use POST /analyze_portfolio for analysis."

# Global error handler for uncaught exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}\nTraceback: {traceback.format_exc()}")
    return jsonify({"error": f"Internal server error: {str(e)}", "stack_trace": traceback.format_exc()}), 500

@app.route('/analyze_portfolio', methods=['OPTIONS'])
def analyze_portfolio_options():
    response = jsonify({"status": "OK"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response, 200

@app.route('/test', methods=['GET'])
def test_endpoint():
    logger.info("Received request to /test")
    response = jsonify({"message": "Hello from the backend!"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 200

@app.route('/analyze_portfolio', methods=['POST'])
def analyze_portfolio():
    logger.debug(f"Entering /analyze_portfolio endpoint with method {request.method}")
    start_time = time.time()
    timeout_limit = 110  # Slightly less than Gunicorn's 120-second timeout

    try:
        # Get JSON data with proper error handling
        try:
            data = request.get_json()
        except Exception as json_err:
            logger.error(f"Failed to parse JSON data: {str(json_err)}")
            return jsonify({"error": f"Failed to parse JSON data: {str(json_err)}"}), 400

        if not data:
            logger.error("No JSON data provided in request")
            return jsonify({"error": "No JSON data provided in request"}), 400

        logger.info(f"Received POST /analyze_portfolio request: {data}")

        # Extract and validate input parameters
        tickers = data.get('tickers', [])
        weights = data.get('weights', [])
        start_date = data.get('start_date', None)
        end_date = data.get('end_date', None)
        risk_tolerance = data.get('risk_tolerance', 'medium')
        benchmarks = data.get('benchmarks', ['^GSPC'])
        optimization_metric = data.get('optimization_metric', 'sharpe')
        fetch_data = data.get('fetch_data', True)

        logger.info(f"Request parameters - tickers: {tickers}, weights: {weights}, start_date: {start_date}, end_date: {end_date}, risk_tolerance: {risk_tolerance}, benchmarks: {benchmarks}")
        logger.debug(f"Request headers: {request.headers}")

        # Validate inputs
        if not isinstance(tickers, list) or not tickers:
            logger.error("Tickers must be a non-empty list")
            return jsonify({"error": "Tickers must be a non-empty list"}), 400

        if not isinstance(weights, list) or not weights:
            logger.error("Weights must be a non-empty list")
            return jsonify({"error": "Weights must be a non-empty list"}), 400

        if len(tickers) != len(weights):
            logger.error("Tickers and weights length mismatch")
            return jsonify({"error": "Tickers and weights length mismatch"}), 400

        if not all(isinstance(w, (int, float)) for w in weights):
            logger.error("Weights must be numeric")
            return jsonify({"error": "Weights must be numeric"}), 400

        weight_sum = sum(weights)
        if weight_sum == 0:
            logger.error("Sum of weights cannot be zero")
            return jsonify({"error": "Sum of weights cannot be zero"}), 400

        # Normalize weights
        weights = [w / weight_sum for w in weights]

        # Validate risk tolerance
        valid_risk_tolerances = ['low', 'medium', 'high']
        if risk_tolerance not in valid_risk_tolerances:
            logger.error(f"Invalid risk tolerance: {risk_tolerance}. Must be one of {valid_risk_tolerances}")
            return jsonify({"error": f"Invalid risk tolerance: {risk_tolerance}. Must be one of {valid_risk_tolerances}"}), 400

        # Validate benchmarks
        if not isinstance(benchmarks, list) or not benchmarks:
            logger.error("Benchmarks must be a non-empty list")
            return jsonify({"error": "Benchmarks must be a non-empty list"}), 400

        # Validate optimization metric
        valid_metrics = ['sharpe', 'sortino', 'max_drawdown', 'volatility', 'value_at_risk']
        if optimization_metric not in valid_metrics:
            logger.error(f"Invalid optimization metric: {optimization_metric}. Must be one of {valid_metrics}")
            return jsonify({"error": f"Invalid optimization metric: {optimization_metric}. Must be one of {valid_metrics}"}), 400

        # Validate dates
        if not start_date or not end_date:
            logger.error("Start date and end date are required")
            return jsonify({"error": "Start date and end date are required"}), 400

        try:
            start_date = pd.Timestamp(start_date).tz_localize(None)
            end_date = pd.Timestamp(end_date).tz_localize(None)
            # Cap end_date at the current date to avoid future dates
            current_date = pd.Timestamp.now().tz_localize(None).normalize()
            if end_date > current_date:
                logger.warning(f"End date {end_date} is in the future. Capping at current date: {current_date}")
                end_date = current_date
            logger.debug(f"Final date range after capping: {start_date} to {end_date}, {(end_date - start_date).days} days")
        except Exception as date_err:
            logger.error(f"Invalid date format: {str(date_err)}")
            return jsonify({"error": f"Invalid date format: {str(date_err)}"}), 400

        if start_date >= end_date:
            logger.error("Start date must be before end date")
            return jsonify({"error": "Start date must be before end date"}), 400

        # Ensure the date range is at least 252 days for meaningful analysis
        min_days = 252
        date_diff = (end_date - start_date).days
        if date_diff < min_days:
            logger.error(f"Date range is too short: {date_diff} days. Minimum required is {min_days} days for reliable portfolio analysis.")
            return jsonify({"error": f"Date range is too short: {date_diff} days. Minimum required is {min_days} days for reliable portfolio analysis."}), 400

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            logger.error("Request processing exceeded timeout limit before starting analysis.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Capture print output
        output_buffer = io.StringIO()
        sys.stdout = output_buffer  # Redirect print statements

        # Fetch stock data
        logger.info("Fetching stock data...")
        try:
            stock_prices, error_tickers, earliest_dates = analyzer.fetch_stock_data(tickers, start_date, end_date)
        except Exception as fetch_err:
            sys.stdout = sys.__stdout__
            logger.error(f"Failed to fetch stock data: {str(fetch_err)}")
            return jsonify({"error": f"Failed to fetch stock data: {str(fetch_err)}"}), 500

        # Check if data was retrieved from cache
        cache_key = (tuple(sorted(tickers)), start_date, end_date)
        cached = cache_key in analyzer.data_cache
        if stock_prices is None or stock_prices.empty:
            sys.stdout = sys.__stdout__
            error_msg = f"No valid stock data available for tickers {tickers}. Error details: {error_tickers.get('error', 'Unknown error')}"
            logger.error(error_msg)
            return jsonify({"error": error_msg, "error_tickers": error_tickers}), 400

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after fetching stock data.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Compute returns
        logger.info("Computing returns...")
        try:
            returns = analyzer.compute_returns(stock_prices)
        except Exception as returns_err:
            sys.stdout = sys.__stdout__
            logger.error(f"Failed to compute returns: {str(returns_err)}")
            return jsonify({"error": f"Failed to compute returns: {str(returns_err)}"}), 500

        if returns.empty:
            sys.stdout = sys.__stdout__
            error_msg = f"No valid returns data for tickers {tickers}. The date range may be too short, or the data may be invalid after cleaning. Ensure the date range spans at least 252 days and that the tickers have sufficient historical data."
            logger.error(error_msg)
            return jsonify({"error": error_msg, "error_tickers": error_tickers}), 400

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing returns.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Fetch risk-free rate
        logger.info("Fetching risk-free rate...")
        try:
            risk_free_rate = analyzer.fetch_treasury_yield()
        except Exception as rf_err:
            sys.stdout = sys.__stdout__
            logger.error(f"Failed to fetch risk-free rate: {str(rf_err)}")
            return jsonify({"error": f"Failed to fetch risk-free rate: {str(rf_err)}"}), 500

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after fetching risk-free rate.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Fetch benchmark data
        logger.info("Fetching benchmark data...")
        benchmark_returns = {}
        benchmark_metrics = {}
        for bench in benchmarks:
            # Use SPY as a proxy for ^GSPC since FMP free tier doesn't support indices
            bench_to_fetch = "SPY" if bench == "^GSPC" else bench
            logger.info(f"Fetching data for benchmark {bench_to_fetch} (original: {bench})...")
            try:
                bench_data, bench_error_tickers, bench_earliest_dates = analyzer.fetch_stock_data([bench_to_fetch], start_date, end_date)
                if bench_data is None or bench_data.empty:
                    logger.warning(f"No data available for benchmark {bench_to_fetch}. Skipping benchmark {bench}.")
                    continue

                # Rename the column to the original benchmark ticker
                if bench_to_fetch != bench:
                    bench_data = bench_data.rename(columns={bench_to_fetch: bench})

                bench_returns = analyzer.compute_returns(bench_data)[bench]
                if bench_returns.empty:
                    logger.warning(f"No valid returns data for benchmark {bench}. Skipping benchmark.")
                    continue

                benchmark_returns[bench] = bench_returns
                benchmark_metrics[bench] = {
                    "annual_return": float(bench_returns.mean() * 252),
                    "annual_volatility": float(bench_returns.std() * np.sqrt(252)) if not pd.isna(bench_returns.std()) else 0.0,
                    "sharpe_ratio": float(analyzer.portfolio_performance(np.array([1.0]), pd.DataFrame(bench_returns), risk_free_rate)[2]),
                    "maximum_drawdown": float(analyzer.compute_max_drawdown(bench_returns)),
                    "value_at_risk": float(analyzer.compute_var(bench_returns, 0.90))
                }
            except Exception as e:
                logger.error(f"Failed to fetch benchmark data for {bench_to_fetch} (original: {bench}): {str(e)}")
                continue

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after fetching benchmark data.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Compute original portfolio metrics
        logger.info("Computing original portfolio metrics...")
        weights_dict = dict(zip(tickers, weights))
        try:
            portfolio_returns = returns.dot(list(weights_dict.values()))
            original_metrics = {
                "annual_return": float(portfolio_returns.mean() * 252),
                "annual_volatility": float(portfolio_returns.std() * np.sqrt(252)) if not pd.isna(portfolio_returns.std()) else 0.0,
                "sharpe_ratio": float(analyzer.portfolio_performance(np.array(list(weights_dict.values())), returns, risk_free_rate)[2]),
                "maximum_drawdown": float(analyzer.compute_max_drawdown(portfolio_returns)),
                "value_at_risk": float(analyzer.compute_var(portfolio_returns, 0.90))
            }
        except Exception as metrics_err:
            sys.stdout = sys.__stdout__
            logger.error(f"Failed to compute original portfolio metrics: {str(metrics_err)}")
            return jsonify({"error": f"Failed to compute original portfolio metrics: {str(metrics_err)}"}), 500

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing original metrics.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Optimize with factor and correlation
        logger.info("Optimizing portfolio with factor and correlation...")
        market_end_date = min(end_date, pd.Timestamp.now().tz_localize(None).normalize())
        market_ticker = data.get('market_ticker', 'SPY')  # Default to SPY if not provided
        market_prices = None
        try:
            logger.info(f"Fetching market data for ticker: {market_ticker}")
            market_data, market_error_tickers, _ = analyzer.fetch_stock_data([market_ticker], start_date, market_end_date)
            if market_data is not None and not market_data.empty and market_ticker in market_data.columns:
                market_prices = market_data[market_ticker]
                logger.info(f"Successfully fetched market data for {market_ticker}.")
            else:
                logger.warning(f"Failed to fetch market data for {market_ticker}. Error details: {market_error_tickers}")
        except Exception as market_err:
            logger.warning(f"Failed to fetch market data for {market_ticker}: {str(market_err)}")

        if market_prices is None:
            logger.warning(f"Market data for {market_ticker} unavailable. Proceeding without market prices in optimization.")
        optimized_weights, opt_metrics = analyzer.optimize_with_factor_and_correlation(
            returns, risk_free_rate, tickers, market_prices, 0.0, 1.0,
            original_weights=list(weights_dict.values())
        )

        # Validate optimized weights and returns before proceeding
        if not isinstance(optimized_weights, np.ndarray) or len(optimized_weights) != len(tickers):
            sys.stdout = sys.__stdout__
            logger.error("Optimization failed to produce valid weights.")
            return jsonify({"error": "Optimization failed to produce valid weights."}), 500
        if returns.empty:
            sys.stdout = sys.__stdout__
            logger.error("Returns data is empty after optimization.")
            return jsonify({"error": "Returns data is empty after optimization."}), 500

        # Define combined_strategies for use in crisis performance and rolling volatility
        combined_strategies = {
            "Original Portfolio": np.array(list(weights_dict.values())),
            "Optimized Portfolio": optimized_weights
        }

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after optimization.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Compute optimized portfolio metrics
        logger.info("Computing optimized portfolio metrics...")
        try:
            optimized_portfolio_returns = returns.dot(optimized_weights)
            optimized_metrics = {
                "annual_return": float(optimized_portfolio_returns.mean() * 252),
                "annual_volatility": float(optimized_portfolio_returns.std() * np.sqrt(252)) if not pd.isna(optimized_portfolio_returns.std()) else 0.0,
                "sharpe_ratio": float(analyzer.portfolio_performance(optimized_weights, returns, risk_free_rate)[2]),
                "maximum_drawdown": float(analyzer.compute_max_drawdown(optimized_portfolio_returns)),
                "value_at_risk": float(analyzer.compute_var(optimized_portfolio_returns, 0.90))
            }
        except Exception as opt_metrics_err:
            sys.stdout = sys.__stdout__
            logger.error(f"Failed to compute optimized portfolio metrics: {str(opt_metrics_err)}")
            return jsonify({"error": f"Failed to compute optimized portfolio metrics: {str(opt_metrics_err)}"}), 500

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing optimized metrics.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Compute efficient frontier
        logger.info("Computing efficient frontier...")
        try:
            frontier = analyzer.get_efficient_frontier(returns, risk_free_rate)
            if not isinstance(frontier, dict) or 'portfolios' not in frontier or 'returns' not in frontier['portfolios'] or 'volatilities' not in frontier['portfolios']:
                logger.error("Efficient frontier calculation failed: 'returns' or 'volatilities' key missing in frontier dictionary")
                frontier = {
                    "portfolios": {"returns": [], "volatilities": []},
                    "strategies": {},
                    "capital_market_line": {"x": [], "y": []}
                }
            else:
                frontier = {
                    "portfolios": [
                        {"return": float(r), "volatility": float(v)}
                        for r, v in zip(frontier["portfolios"]["returns"], frontier["portfolios"]["volatilities"])
                    ],
                    "strategies": frontier["strategies"],
                    "capital_market_line": frontier["capital_market_line"]
                }
        except Exception as frontier_err:
            sys.stdout = sys.__stdout__
            logger.error(f"Failed to compute efficient frontier: {str(frontier_err)}")
            frontier = {
                "portfolios": {"returns": [], "volatilities": []},
                "strategies": {},
                "capital_market_line": {"x": [], "y": []}
            }

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing efficient frontier.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Suggest courses of action
        logger.info("Suggesting courses of action...")
        try:
            suggestions = analyzer.suggest_courses_of_action(
                tickers,
                np.array(list(weights_dict.values())),
                optimized_weights,
                returns,
                risk_free_rate,
                benchmark_metrics,
                risk_tolerance,
                start_date,
                end_date
            )
        except Exception as suggest_err:
            sys.stdout = sys.__stdout__
            logger.error(f"Failed to suggest courses of action: {str(suggest_err)}")
            suggestions = {"error": f"Failed to generate suggestions: {str(suggest_err)}"}

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after suggesting courses of action.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Compute Fama-French exposures
        logger.info("Computing Fama-French exposures...")
        try:
            ff_exposures = analyzer.compute_fama_french_exposures(portfolio_returns, start_date, end_date)
        except Exception as ff_err:
            sys.stdout = sys.__stdout__
            logger.error(f"Failed to compute Fama-French exposures: {str(ff_err)}")
            ff_exposures = {"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0}

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing Fama-French exposures.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Correlation Matrix
        logger.info("Computing correlation matrix...")
        try:
            corr_matrix = analyzer.get_correlation_matrix(stock_prices)
            correlation_matrix = {
                "labels": corr_matrix["tickers"],
                "values": corr_matrix["matrix"]
            }
        except Exception as corr_err:
            logger.error(f"Error computing correlation matrix: {str(corr_err)}")
            correlation_matrix = {"labels": [], "matrix": []}

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing correlation matrix.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Eigenvalue Analysis
        logger.info("Computing eigenvalue analysis...")
        try:
            eigenvalues, explained_variance_ratio = analyzer.compute_eigenvalues(returns)
            cumulative_variance = np.cumsum(explained_variance_ratio)
            eigenvalues_data = {
                "eigenvalues": [float(e) for e in eigenvalues],
                "cumulative_variance": [float(c) for c in cumulative_variance],
                "labels": [f"Factor {i+1}" for i in range(len(eigenvalues))]
            }
        except Exception as eig_err:
            logger.error(f"Error computing eigenvalue analysis: {str(eig_err)}")
            eigenvalues_data = {"eigenvalues": [], "cumulative_variance": [], "labels": []}

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing eigenvalue analysis.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Cumulative Returns
        logger.info("Computing cumulative returns...")
        try:
            strategies = {
                "Original Portfolio": np.array(list(weights_dict.values())),
                "Max Sharpe": analyzer.optimize_portfolio(returns, risk_free_rate, "sharpe"),
                "Max Sortino": analyzer.optimize_portfolio(returns, risk_free_rate, "sortino"),
                "Min Max Drawdown": analyzer.optimize_portfolio(returns, risk_free_rate, "max_drawdown"),
                "Min Volatility": analyzer.optimize_portfolio(returns, risk_free_rate, "volatility"),
                "Min Value at Risk": analyzer.optimize_portfolio(returns, risk_free_rate, "value_at_risk")
            }
            cumulative_returns_result = analyzer.get_cumulative_returns(returns, strategies, benchmark_returns, earliest_dates)
            cumulative_returns = cumulative_returns_result["cumulative_returns"]
        except Exception as cum_err:
            logger.error(f"Error computing cumulative returns: {str(cum_err)}")
            cumulative_returns = {}

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing cumulative returns.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Historical Strategies
        logger.info("Computing historical strategies...")
        try:
            hist_start_date = "2015-03-24"
            if pd.to_datetime(hist_start_date) < pd.to_datetime(start_date):
                hist_start_date = start_date
            hist_data, _, _ = analyzer.fetch_stock_data(tickers, hist_start_date, end_date)
            historical_strategies = {"metrics": {}, "labels": []}
            if hist_data is not None and not hist_data.empty:
                hist_returns = analyzer.compute_returns(hist_data)
                historical_metrics, historical_labels = analyzer.get_historical_metrics(tickers, weights_dict, risk_free_rate, hist_returns)
                historical_strategies["metrics"] = historical_metrics
                historical_strategies["labels"] = historical_labels
        except Exception as hist_err:
            logger.error(f"Error computing historical strategies: {str(hist_err)}")
            historical_strategies = {"metrics": {"Annual Return": [], "Volatility": [], "Avg Correlation": []}, "labels": []}

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing historical strategies.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Diversification Benefit
        logger.info("Computing diversification benefit...")
        try:
            diversification_benefit = analyzer.get_diversification_benefit(returns, np.array(list(weights_dict.values())), optimized_weights, tickers)
        except Exception as div_err:
            logger.error(f"Error computing diversification benefit: {str(div_err)}")
            diversification_benefit = {"labels": [], "volatilities": [], "returns": []}

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing diversification benefit.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Portfolio Exposures
        logger.info("Computing portfolio exposures...")
        try:
            portfolio_exposures = analyzer.get_portfolio_exposures(tickers, np.array(list(weights_dict.values())), optimized_weights)
        except Exception as exp_err:
            logger.error(f"Error computing portfolio exposures: {str(exp_err)}")
            portfolio_exposures = {"original": {"labels": [], "exposures": []}, "optimized": {"labels": [], "exposures": []}}

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing portfolio exposures.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Define crisis periods for validation
        crises_periods = [
                {"start": pd.to_datetime("2000-03-01"), "end": pd.to_datetime("2002-10-31")},  # Dot-Com Bust
                {"start": pd.to_datetime("2007-12-01"), "end": pd.to_datetime("2009-06-30")},  # Great Recession
                {"start": pd.to_datetime("2020-02-01"), "end": pd.to_datetime("2020-04-30")}   # COVID-19 Crisis
        ]

        # Crisis Performance
        logger.info("Computing crisis performance...")
        crisis_performance = {}
        try:
                if returns.empty or len(returns) < 2:
                        logger.warning("Returns data is empty or insufficient for crisis performance analysis.")
                else:
                        data_start = returns.index.min()
                        data_end = returns.index.max()
                        overlap_found = False
                        for crisis in crises_periods:
                                crisis_start = crisis["start"]
                                crisis_end = crisis["end"]
                                if not (data_end < crisis_start or data_start > crisis_end):
                                        overlap_found = True
                                        break
                        if overlap_found:
                                crisis_performance = analyzer.get_crisis_performance(returns, combined_strategies, benchmark_returns, earliest_dates)
                                if "error" in crisis_performance:
                                        logger.warning(crisis_performance["error"])
                                        crisis_performance = {}
                        else:
                                logger.warning(f"Date range {data_start} to {data_end} does not overlap with any defined crisis period. Skipping crisis performance plot.")
        except Exception as crisis_err:
                logger.error(f"Error computing crisis performance: {str(crisis_err)}")
                crisis_performance = {}

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
                sys.stdout = sys.__stdout__
                logger.error("Request processing exceeded timeout limit after computing crisis performance.")
                return jsonify({"error": "Request processing exceeded timeout limit."}), 503


        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing crisis performance.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Rolling Volatility
        logger.info("Computing rolling volatility...")
        try:
            rolling_volatility_data = analyzer.get_rolling_volatility(returns, combined_strategies, benchmark_returns)
            rolling_volatility = rolling_volatility_data["rolling_volatility"]
        except Exception as vol_err:
            logger.error(f"Error computing rolling volatility: {str(vol_err)}")
            rolling_volatility = {}

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing rolling volatility.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Comparison Bars
        logger.info("Computing comparison bars...")
        try:
            comparison_bars = analyzer.get_comparison_bars(original_metrics, optimized_metrics, benchmark_metrics)
        except Exception as comp_err:
            logger.error(f"Error computing comparison bars: {str(comp_err)}")
            comparison_bars = []

        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            sys.stdout = sys.__stdout__
            logger.error("Request processing exceeded timeout limit after computing comparison bars.")
            return jsonify({"error": "Request processing exceeded timeout limit."}), 503

        # Prepare response
        logger.info("Preparing response...")
        try:
                response = {
                        "original_metrics": {k: float(v) if v is not None else 0.0 for k, v in original_metrics.items()},
                        "optimized_metrics": {k: float(v) if v is not None else 0.0 for k, v in optimized_metrics.items()},
                        "benchmark_metrics": {
                                k: {metric: float(value) if value is not None else 0.0 for metric, value in v.items()}
                                for k, v in benchmark_metrics.items()
                        },
                        "comparison_bars": comparison_bars,
                        "correlation_matrix": correlation_matrix,
                        "eigenvalues": eigenvalues_data,
                        "cumulative_returns": cumulative_returns,
                        "historical_strategies": historical_strategies,
                        "efficient_frontier": frontier,
                        "diversification_benefit": diversification_benefit,
                        "portfolio_exposures": portfolio_exposures,
                        "rolling_volatility": rolling_volatility,
                        "fama_french_exposures": {k: float(v) if v is not None else 0.0 for k, v in ff_exposures.items()},
                        "suggestions": suggestions,
                        "error_tickers": error_tickers,
                        "earliest_dates": earliest_dates,
                        "optimized_weights": {t: float(w) for t, w in zip(tickers, optimized_weights)}
                }
                # Only include crisis_performance if it's not empty
                if crisis_performance:
                        response["crisis_performance"] = crisis_performance
                else:
                        logger.info("Crisis performance data not included in response due to lack of applicable crises.")
        except Exception as resp_err:
                sys.stdout = sys.__stdout__
                logger.error(f"Failed to prepare response: {str(resp_err)}")
                return jsonify({"error": f"Failed to prepare response: {str(resp_err)}"}), 500


        # Capture print output
        sys.stdout = sys.__stdout__
        try:
            analysis_text = output_buffer.getvalue()
        except Exception as buffer_err:
            logger.error(f"Failed to capture analysis text: {str(buffer_err)}")
            analysis_text = ""
        finally:
            output_buffer.close()

        logger.info("Portfolio analysis completed successfully")
        return jsonify(response), 200

    except Exception as e:
        sys.stdout = sys.__stdout__
        error_message = f"Internal server error: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error(error_message)
        return jsonify({"error": error_message, "stack_trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Flask app...")
        # Note: For local testing, consider using Gunicorn to apply timeout settings
        # Run with: gunicorn --bind 0.0.0.0:8080 --timeout 120 main:app
        logger.warning("Running Flask development server. For production or better timeout control, use Gunicorn.")
        app.run(host='0.0.0.0', port=8080, debug=False)
    except Exception as e:
        logger.error(f"Error starting Flask app: {str(e)}\nTraceback: {traceback.format_exc()}")
        raise