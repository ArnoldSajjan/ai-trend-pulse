"""
Configuration file for AI TrendPulse
Contains all API endpoints, constants, and settings
"""
from dataclasses import dataclass, field
from typing import Dict, List
import os
from datetime import datetime

@dataclass
class Config:
    # Cache settings
    CACHE_TTL: int = 300  # 5 minutes
    REQUEST_TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    
    # Google AdSense Configuration
    GOOGLE_ADSENSE_CLIENT: str = os.getenv("GOOGLE_ADSENSE_CLIENT", "ca-pub-XXXXXXXXXXXXXXXX")
    GOOGLE_ADSENSE_SLOTS: Dict[str, str] = field(default_factory=lambda: {
        "header": "1234567890",
        "sidebar": "2345678901",
        "content": "3456789012",
        "footer": "4567890123"
    })
    
    # Free AI News Sources
    AI_NEWS_SOURCES: Dict[str, str] = field(default_factory=lambda: {
        # RSS Feeds
        "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "VentureBeat AI": "https://venturebeat.com/ai/feed/",
        "The Verge AI": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "MIT Technology Review": "https://www.technologyreview.com/feed/",
        "AI News": "https://artificialintelligence-news.com/feed/",
        "OpenAI Blog": "https://openai.com/blog/rss/",
        "Google AI Blog": "https://blog.google/technology/ai/rss/",
        "DeepMind Blog": "https://deepmind.com/blog/feed/basic/",
        "NVIDIA AI Blog": "https://blogs.nvidia.com/blog/category/deep-learning-ai/feed/",
        "Meta AI": "https://ai.meta.com/blog/rss/",
        "Anthropic News": "https://www.anthropic.com/rss.xml",
        "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
        "AI Business": "https://aibusiness.com/rss.xml",
        "Analytics India Magazine": "https://analyticsindiamag.com/feed/",
        "Towards Data Science": "https://towardsdatascience.com/feed",
        "KDnuggets": "https://www.kdnuggets.com/feed",
        "AI Weekly": "https://aiweekly.substack.com/feed",
        "The Batch": "https://www.deeplearning.ai/the-batch/feed/",
        "AI Trends": "https://www.aitrends.com/feed/",
        "IEEE Spectrum AI": "https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss"
    })
    
    # Free API Endpoints
    AI_DATA_SOURCES: Dict[str, str] = field(default_factory=lambda: {
        # Reddit APIs
        "reddit_ai": "https://www.reddit.com/r/artificial/top.json",
        "reddit_ml": "https://www.reddit.com/r/MachineLearning/top.json",
        "reddit_openai": "https://www.reddit.com/r/OpenAI/top.json",
        "reddit_chatgpt": "https://www.reddit.com/r/ChatGPT/top.json",
        "reddit_singularity": "https://www.reddit.com/r/singularity/top.json",
        "reddit_localllama": "https://www.reddit.com/r/LocalLLaMA/top.json",
        "reddit_midjourney": "https://www.reddit.com/r/midjourney/top.json",
        
        # Hacker News
        "hn_api": "https://hn.algolia.com/api/v1/search",
        "hn_top": "https://hacker-news.firebaseio.com/v0/topstories.json",
        "hn_new": "https://hacker-news.firebaseio.com/v0/newstories.json",
        "hn_ask": "https://hacker-news.firebaseio.com/v0/askstories.json",
        "hn_show": "https://hacker-news.firebaseio.com/v0/showstories.json",
        
        # GitHub
        "github_trending": "https://api.github.com/search/repositories",
        "github_topics": "https://api.github.com/search/topics",
        
        # Dev.to
        "devto_ai": "https://dev.to/api/articles?tag=ai",
        "devto_ml": "https://dev.to/api/articles?tag=machinelearning",
        
        # Product Hunt (Unofficial)
        "producthunt_posts": "https://api.producthunt.com/v1/posts",
        
        # Papers & Research
        "arxiv_api": "http://export.arxiv.org/api/query",
        "paperswithcode": "https://paperswithcode.com/api/v1/papers/",
        
        # Lobsters
        "lobsters_ai": "https://lobste.rs/t/ai.json",
        
        # IndieHackers
        "indiehackers": "https://www.indiehackers.com/feed.json",
        
        # Mastodon AI instances
        "mastodon_ai": "https://sigmoid.social/api/v1/timelines/public",
        
        # YouTube Data API Alternative
        "youtube_rss": "https://www.youtube.com/feeds/videos.xml?channel_id=",
        
        # Google Trends (Unofficial)
        "google_trends": "https://trends.google.com/trends/api/explore",
        
        # Alternative Data Sources
        "newsapi_everything": "https://newsapi.org/v2/everything",  # Free tier available
        "gnews": "https://gnews.io/api/v4/search",  # Free tier available
        "mediastack": "http://api.mediastack.com/v1/news",  # Free tier available
        "currentsapi": "https://api.currentsapi.services/v1/search",  # Free tier available
    })
    
    # AI YouTube Channels for RSS
    AI_YOUTUBE_CHANNELS: Dict[str, str] = field(default_factory=lambda: {
        "Two Minute Papers": "UCbfYPyITQ-7l4upoX8nvctg",
        "Yannic Kilcher": "UCZHmQk67mSJgfCCTn7xBfew",
        "Lex Fridman": "UCSHZKyawb77ixDdsGog4iWA",
        "AI Explained": "UCqaKX_G8J4PK88rRUHqaQAA",
        "The AI Advantage": "UCN7-TgJr6sAiE3MCBzU5RLw",
        "Matt Wolfe": "UCq0t82bHVZvbBvvqNRM5rKw",
        "AI Coffee Break": "UCobqgqE4i5Kf7wrxRxhToQA",
        "Machine Learning Street Talk": "UCMLtBahI5DMrt0NPvDSoIRQ",
        "AI Search": "UCKIvhP2tFWv_uEfvqU_G0Kw",
        "The AI Epiphany": "UCj8shBGI8q8FAjZNuqG3W5Q"
    })
    
    # Free AI Tool Discovery Sources
    AI_TOOLS_SOURCES: Dict[str, str] = field(default_factory=lambda: {
        "futurepedia": "https://www.futurepedia.io/api/tools",  # Check if available
        "theresanaiforthat": "https://theresanaiforthat.com/api/tools",  # Check if available
        "aitools_fyi": "https://aitools.fyi/api/tools",  # Check if available
        "alternativeto": "https://alternativeto.net/browse/search/?q=ai",
        "saashub": "https://www.saashub.com/best-ai-tools",
        "betalist": "https://betalist.com/tags/artificial-intelligence",
        "stackshare": "https://stackshare.io/artificial-intelligence",
        "g2_ai": "https://www.g2.com/categories/ai",
        "capterra_ai": "https://www.capterra.com/artificial-intelligence-software/",
        "github_awesome_ai": "https://raw.githubusercontent.com/sindresorhus/awesome/main/readme.md"
    })
    
    # AI Research & Papers
    AI_RESEARCH_SOURCES: Dict[str, str] = field(default_factory=lambda: {
        "semantic_scholar": "https://api.semanticscholar.org/v1/paper/search",
        "crossref": "https://api.crossref.org/works",
        "core_ac_uk": "https://core.ac.uk/api-v2/search",
        "microsoft_academic": "https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate",
        "dblp": "https://dblp.org/search/publ/api",
        "acm_dl": "https://dl.acm.org/feeds/feedContent.jsp",
        "ieee_xplore": "https://ieeexplore.ieee.org/rest/search"
    })
    
    # AI Job Boards
    AI_JOB_SOURCES: Dict[str, str] = field(default_factory=lambda: {
        "remotive": "https://remotive.io/api/remote-jobs?category=software-dev&search=ai",
        "wellfound": "https://wellfound.com/jobs",
        "ai_jobs": "https://ai-jobs.net/api/jobs",
        "indeed": "https://www.indeed.com/rss?q=artificial+intelligence",
        "simplyhired": "https://www.simplyhired.com/search?q=ai+engineer",
        "linkedin_jobs": "https://www.linkedin.com/jobs/search/?keywords=artificial%20intelligence"
    })
    
    # AI Events & Conferences
    AI_EVENTS_SOURCES: Dict[str, str] = field(default_factory=lambda: {
        "eventbrite": "https://www.eventbrite.com/rss/organizer_list_events/",
        "meetup": "https://api.meetup.com/find/events",
        "ai_events": "https://www.ai-events.org/feed/",
        "conference_alerts": "https://conferencealerts.com/rss/ai.xml"
    })
    
    # Podcasts
    AI_PODCAST_SOURCES: Dict[str, str] = field(default_factory=lambda: {
        "lex_fridman": "https://lexfridman.com/feed/podcast/",
        "twiml": "https://twimlai.com/feed/",
        "data_skeptic": "https://dataskeptic.com/feed.rss",
        "linear_digressions": "https://feeds.feedburner.com/udacity-linear-digressions",
        "ai_alignment": "https://futureoflife.org/feed/podcast/",
        "gradient_dissent": "https://www.wandb.com/podcast/feed.xml",
        "practical_ai": "https://practicalai.fm/rss",
        "mlops": "https://podcast.mlops.community/feed.xml"
    })
    
    # Social Media Monitoring
    SOCIAL_MEDIA_KEYWORDS: List[str] = field(default_factory=lambda: [
        "artificial intelligence", "machine learning", "deep learning", "neural networks",
        "gpt", "llm", "generative ai", "ai tools", "ai news", "chatgpt", "claude",
        "gemini", "midjourney", "stable diffusion", "langchain", "vector database",
        "prompt engineering", "ai safety", "agi", "transformer", "diffusion model",
        "reinforcement learning", "computer vision", "nlp", "natural language processing",
        "ai ethics", "ai regulation", "ai startup", "ai research", "foundation model",
        "multimodal ai", "ai agent", "autogpt", "babyagi", "ai workflow", "ai automation"
    ])
    
    # Monetization Settings
    MONETIZATION: Dict[str, any] = field(default_factory=lambda: {
        "show_ads": True,
        "ad_frequency": 5,  # Show ad after every 5 items
        "affiliate_programs": {
            "amazon": "your-amazon-affiliate-id",
            "gumroad": "your-gumroad-affiliate-id",
            "udemy": "your-udemy-affiliate-id"
        },
        "premium_features": {
            "api_access": False,
            "advanced_filters": False,
            "export_formats": ["json", "csv"],
            "email_alerts": False
        }
    })
    
    # UI Configuration
    UI_CONFIG: Dict[str, any] = field(default_factory=lambda: {
        "theme": "dark",
        "primary_color": "#667eea",
        "secondary_color": "#764ba2",
        "success_color": "#48bb78",
        "warning_color": "#f6ad55",
        "error_color": "#fc8181",
        "font_family": "Inter, system-ui, sans-serif",
        "max_items_per_page": 25,
        "enable_animations": True,
        "show_thumbnails": True,
        "lazy_loading": True
    })
    
    # Performance Settings
    PERFORMANCE: Dict[str, any] = field(default_factory=lambda: {
        "enable_caching": True,
        "cache_duration": 300,  # 5 minutes
        "max_concurrent_requests": 10,
        "request_timeout": 10,
        "enable_compression": True,
        "image_optimization": True,
        "lazy_load_images": True
    })
    
    # Analytics
    ANALYTICS: Dict[str, str] = field(default_factory=lambda: {
        "google_analytics_id": os.getenv("GA_MEASUREMENT_ID", "G-XXXXXXXXXX"),
        "mixpanel_token": os.getenv("MIXPANEL_TOKEN", ""),
        "hotjar_id": os.getenv("HOTJAR_ID", ""),
        "track_events": True,
        "track_page_views": True,
        "track_user_behavior": True
    })

# Initialize config
config = Config()