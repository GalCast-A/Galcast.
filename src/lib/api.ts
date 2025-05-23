import axios from 'axios';
import axiosRetry from 'axios-retry';
import rateLimit from 'axios-rate-limit';

interface PortfolioAnalysisParams {
  tickers: string[];
  weights: number[];
  start_date: string;
  end_date: string;
  risk_tolerance: string;
  benchmarks: string[];
  optimization_metric: string;
}

// Create rate-limited API clients
const polygonApi = rateLimit(
  axios.create({
    baseURL: 'https://api.polygon.io/v3',
    timeout: 10000,
    params: {
      apiKey: import.meta.env.VITE_POLYGON_API_KEY
    }
  }),
  { maxRequests: 5, perMilliseconds: 60000 }
);

const finnhubApi = rateLimit(
  axios.create({
    baseURL: 'https://finnhub.io/api/v1',
    timeout: 10000,
    headers: {
      'X-Finnhub-Token': import.meta.env.VITE_FINNHUB_API_KEY
    }
  }),
  { maxRequests: 30, perMilliseconds: 60000 }
);

const stockNewsApi = rateLimit(
  axios.create({
    baseURL: 'https://stocknewsapi.com/api/v1',
    timeout: 10000,
    params: {
      token: import.meta.env.VITE_STOCKNEWS_API_KEY
    }
  }),
  { maxRequests: 30, perMilliseconds: 60000 }
);

// Apply retry logic with exponential backoff and jitter
const retryConfig = {
  retries: 3,
  retryDelay: (retryCount: number) => {
    return Math.min(1000 * Math.pow(2, retryCount), 10000) + Math.random() * 1000;
  },
  retryCondition: (error: any) => {
    return axiosRetry.isNetworkOrIdempotentRequestError(error) ||
           error.response?.status === 429 ||
           (error.response?.status >= 500 && error.response?.status <= 599);
  }
};

[polygonApi, finnhubApi, stockNewsApi].forEach(api => {
  axiosRetry(api, retryConfig);
});

const handleApiError = (error: any, apiName: string) => {
  console.error(`${apiName} API error:`, error);

  if (!navigator.onLine) {
    throw new Error('No internet connection. Please check your network and try again.');
  }

  if (error.code === 'ERR_NETWORK') {
    throw new Error(`Unable to connect to ${apiName}. Please check your internet connection and try again.`);
  }

  if (error.response?.status === 429) {
    throw new Error(`${apiName} rate limit exceeded. Please wait a moment and try again.`);
  }

  if (error.code === 'ECONNABORTED') {
    throw new Error(`${apiName} request timed out. Please try again.`);
  }

  if (error.response?.status === 401 || error.response?.status === 403) {
    throw new Error(`Authentication failed with ${apiName}. Please check your API keys.`);
  }

  throw new Error(`${apiName} service error. Please try again later.`);
};

export async function searchStocks(query: string) {
  if (!query || query.length < 2) return [];

  if (!navigator.onLine) {
    throw new Error('No internet connection. Please check your network and try again.');
  }

  // Check if API keys are configured
  if (!import.meta.env.VITE_POLYGON_API_KEY && !import.meta.env.VITE_FINNHUB_API_KEY) {
    throw new Error('Stock API configuration is missing. Please check your environment variables.');
  }

  try {
    // Try Polygon.io first
    if (import.meta.env.VITE_POLYGON_API_KEY) {
      try {
        const response = await polygonApi.get('/reference/tickers', {
          params: {
            search: query,
            active: true,
            sort: 'ticker',
            order: 'asc',
            limit: 10,
            market: 'stocks'
          }
        });

        if (response.data?.results?.length > 0) {
          return response.data.results.map((item: any) => ({
            value: item.ticker,
            label: `${item.ticker} - ${item.name}`
          }));
        }
      } catch (error: any) {
        console.warn('Polygon.io search failed, falling back to Finnhub');
        handleApiError(error, 'Polygon.io');
      }
    }

    // Fall back to Finnhub
    if (import.meta.env.VITE_FINNHUB_API_KEY) {
      try {
        const response = await finnhubApi.get('/search', {
          params: { q: query }
        });

        if (response.data?.result) {
          return response.data.result
            .filter((item: any) => item.type === 'Common Stock')
            .map((item: any) => ({
              value: item.symbol,
              label: `${item.symbol} - ${item.description}`
            }));
        }
      } catch (error: any) {
        handleApiError(error, 'Finnhub');
      }
    }

    return [];
  } catch (error: any) {
    // If we get here, both APIs have failed
    throw new Error('All stock search services are currently unavailable. Please try again later.');
  }
}

export async function getNewsAndSentiment(symbol: string) {
  if (!symbol) return null;

  try {
    if (!navigator.onLine) {
      throw new Error('No internet connection');
    }

    // Get company news from Finnhub
    const newsResponse = await finnhubApi.get('/company-news', {
      params: {
        symbol,
        from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        to: new Date().toISOString().split('T')[0]
      }
    });

    // Get sentiment data from StockNews API
    const sentimentResponse = await stockNewsApi.get('/stats', {
      params: {
        tickers: symbol,
        section: 'general',
        date: 'last7days'
      }
    });

    const news = newsResponse.data || [];
    const sentiment = processSentimentData(sentimentResponse.data?.data || {});

    return {
      buzz: {
        articlesInLastWeek: news.length,
        buzz: calculateBuzz(news),
        weeklyAverage: calculateWeeklyAverage(news)
      },
      sentiment: {
        bearishPercent: sentiment.bearish,
        bullishPercent: sentiment.bullish
      },
      companyNewsScore: calculateNewsScore(news),
      sectorAverageBullishPercent: sentiment.sectorBullish,
      sectorAverageBearishPercent: sentiment.sectorBearish,
      news: news.map((article: any) => ({
        title: article.headline,
        description: article.summary,
        published_at: article.datetime * 1000,
        url: article.url,
        source: article.source
      })).slice(0, 10)
    };
  } catch (error: any) {
    handleApiError(error, 'News and Sentiment');
  }
}

function processSentimentData(data: any) {
  const sentiment = data.sentiment || {};
  
  return {
    bullish: sentiment.positive || 50,
    bearish: sentiment.negative || 50,
    sectorBullish: sentiment.sector_positive || 55,
    sectorBearish: sentiment.sector_negative || 45
  };
}

function calculateBuzz(news: any[]) {
  return Math.min(1, news.length / 20);
}

function calculateWeeklyAverage(news: any[]) {
  return Math.max(1, news.length / 7);
}

function calculateNewsScore(news: any[]) {
  if (!news.length) return 0.5;
  
  const recentNews = news.slice(0, 20);
  const totalScore = recentNews.reduce((score: number, article: any) => {
    const age = (Date.now() - article.datetime * 1000) / (24 * 60 * 60 * 1000);
    return score + (1 / (1 + age));
  }, 0);
  
  return Math.min(1, totalScore / 20);
}

export async function portfolioAnalysis(params: PortfolioAnalysisParams) {
  try {
    const response = await axios.post('https://galcast-analytics-413625117094.us-central1.run.app/analyze_portfolio', params);
    return response.data;
  } catch (error: any) {
    handleApiError(error, 'Portfolio Analysis');
  }
}