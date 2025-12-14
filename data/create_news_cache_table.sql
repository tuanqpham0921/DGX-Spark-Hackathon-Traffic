-- Create table for caching news articles
-- This improves performance and provides context for AI chatbot

CREATE TABLE IF NOT EXISTS public.news_cache (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    summary TEXT,
    published_date TIMESTAMP,
    source TEXT,
    category VARCHAR(50), -- 'traffic' or 'political_economy'
    keywords TEXT[], -- Array of keywords that matched
    content TEXT, -- Full article content if available
    cached_at TIMESTAMP DEFAULT NOW(),
    region VARCHAR(50), -- DC, Virginia, or All
    
    -- Indexes for fast queries
    CONSTRAINT unique_url UNIQUE(url)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_news_published_date ON public.news_cache(published_date DESC);
CREATE INDEX IF NOT EXISTS idx_news_category ON public.news_cache(category);
CREATE INDEX IF NOT EXISTS idx_news_cached_at ON public.news_cache(cached_at);
CREATE INDEX IF NOT EXISTS idx_news_region ON public.news_cache(region);

-- Create composite index for common queries
CREATE INDEX IF NOT EXISTS idx_news_category_date ON public.news_cache(category, published_date DESC);

COMMENT ON TABLE public.news_cache IS 'Cached news articles from Tavily API for traffic and political/economic news';
COMMENT ON COLUMN public.news_cache.category IS 'traffic = traffic-related news, political_economy = political/economic news affecting traffic';
COMMENT ON COLUMN public.news_cache.keywords IS 'Keywords that matched this article';

