"""
Enhanced data fetching classes with multiple free API sources
"""
import streamlit as st
import requests
import feedparser
import json
import re
import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Union
import logging
from bs4 import BeautifulSoup
import time
from urllib.parse import quote_plus, urlencode
import random
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncRequestHandler:
    """Handle asynchronous requests for better performance"""
    
    @staticmethod
    async def fetch(session, url, params=None, headers=None):
        try:
            async with session.get(url, params=params, headers=headers, timeout=10) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Error fetching {url}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Exception fetching {url}: {e}")
            return None
    
    @staticmethod
    async def fetch_multiple(urls_params):
        """Fetch multiple URLs concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url, params, headers in urls_params:
                tasks.append(AsyncRequestHandler.fetch(session, url, params, headers))
            return await asyncio.gather(*tasks)

class EnhancedAINewsAPI:
    """Enhanced news fetching from multiple sources"""
    
    def __init__(self):
        self.sources = config.AI_NEWS_SOURCES
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def get_comprehensive_news(_self) -> List[Dict]:
        """Fetch news from all available sources"""
        all_articles = []
        
        # Fetch RSS feeds
        rss_articles = _self._fetch_rss_feeds()
        all_articles.extend(rss_articles)
        
        # Fetch from Dev.to
        devto_articles = _self._fetch_devto_articles()
        all_articles.extend(devto_articles)
        
        # Fetch from Lobsters
        lobsters_articles = _self._fetch_lobsters()
        all_articles.extend(lobsters_articles)
        
        # Fetch YouTube videos
        youtube_articles = _self._fetch_youtube_videos()
        all_articles.extend(youtube_articles)
        
        # Remove duplicates and sort
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            if article['title'] not in seen_titles:
                seen_titles.add(article['title'])
                unique_articles.append(article)
        
        # Sort by date
        # Filter out articles where 'published' is None or missing
        valid_articles = [a for a in unique_articles if a.get('published')]

        # Sort only valid ones
        valid_articles.sort(key=lambda x: x['published'], reverse=True)

        # Use sorted valid_articles (or append the rest if needed)
        unique_articles = valid_articles

        
        return unique_articles[:100]  # Return top 100 articles
    
    def _fetch_rss_feeds(_self) -> List[Dict]:
        """Fetch articles from RSS feeds"""
        articles = []
        
        for source_name, feed_url in _self.sources.items():
            try:
                feed = feedparser.parse(feed_url, agent=_self.headers['User-Agent'])
                
                if feed.entries:
                    for entry in feed.entries[:10]:  # Get top 10 from each source
                        published = _self._parse_date(entry)
                        
                        article = {
                            "title": entry.get('title', 'No Title'),
                            "link": entry.get('link', '#'),
                            "published": published,
                            "summary": _self._clean_summary(entry),
                            "source": source_name,
                            "author": entry.get('author', 'Unknown'),
                            "type": "article",
                            "tags": _self._extract_tags(entry)
                        }
                        articles.append(article)
                        
            except Exception as e:
                logger.error(f"Error fetching {source_name}: {e}")
                
        return articles
    
    def _fetch_devto_articles(_self) -> List[Dict]:
        """Fetch articles from Dev.to"""
        articles = []
        
        try:
            # Fetch AI tagged articles
            response = requests.get(
                "https://dev.to/api/articles",
                params={"tag": "ai", "per_page": 30},
                headers=_self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data:
                    article = {
                        "title": item['title'],
                        "link": item['url'],
                        "published": datetime.fromisoformat(item['published_at'].replace('Z', '+00:00')),
                        "summary": item.get('description', 'Read more on Dev.to'),
                        "source": "Dev.to",
                        "author": item['user']['name'],
                        "type": "article",
                        "tags": item.get('tag_list', []),
                        "reading_time": item.get('reading_time_minutes', 0),
                        "reactions": item.get('positive_reactions_count', 0)
                    }
                    articles.append(article)
                    
        except Exception as e:
            logger.error(f"Error fetching Dev.to articles: {e}")
            
        return articles
    
    def _fetch_lobsters(_self) -> List[Dict]:
        """Fetch from Lobsters"""
        articles = []
        
        try:
            response = requests.get(
                "https://lobste.rs/t/ai.json",
                headers=_self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data[:20]:  # Get top 20
                    article = {
                        "title": item['title'],
                        "link": item['url'],
                        "published": datetime.fromisoformat(item['created_at']),
                        "summary": f"{item.get('comment_count', 0)} comments • {item.get('score', 0)} points",
                        "source": "Lobsters",
                        "author": item.get('submitter_user', {}).get('username', 'Unknown'),
                        "type": "discussion",
                        "tags": item.get('tags', []),
                        "score": item.get('score', 0)
                    }
                    articles.append(article)
                    
        except Exception as e:
            logger.error(f"Error fetching Lobsters: {e}")
            
        return articles
    
    def _fetch_youtube_videos(_self) -> List[Dict]:
        """Fetch latest videos from AI YouTube channels"""
        articles = []
        
        for channel_name, channel_id in list(config.AI_YOUTUBE_CHANNELS.items())[:5]:  # Limit to 5 channels
            try:
                feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
                feed = feedparser.parse(feed_url)
                
                if feed.entries:
                    for entry in feed.entries[:3]:  # Get latest 3 videos per channel
                        article = {
                            "title": f"📺 {entry.title}",
                            "link": entry.link,
                            "published": _self._parse_date(entry),
                            "summary": f"New video from {channel_name}",
                            "source": f"YouTube - {channel_name}",
                            "author": channel_name,
                            "type": "video",
                            "tags": ["video", "youtube", "ai"]
                        }
                        articles.append(article)
                        
            except Exception as e:
                logger.error(f"Error fetching YouTube {channel_name}: {e}")
                
        return articles
    
    def _parse_date(_self, entry) -> datetime:
        """Parse date from various formats"""
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])
        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6])
        else:
            return datetime.now()
    
    def _clean_summary(_self, entry) -> str:
        """Clean and format summary"""
        summary = getattr(entry, 'summary', getattr(entry, 'description', ''))
        if summary:
            # Remove HTML tags
            summary = re.sub('<[^<]+?>', '', summary)
            summary = summary.strip()
            # Limit length
            if len(summary) > 200:
                summary = summary[:197] + "..."
        else:
            summary = "Click to read more..."
        return summary
    
    def _extract_tags(_self, entry) -> List[str]:
        """Extract tags from entry"""
        tags = []
        if hasattr(entry, 'tags'):
            tags = [tag.term for tag in entry.tags]
        return tags

class EnhancedRedditAPI:
    """Enhanced Reddit data fetching"""
    
    def __init__(self):
        self.base_url = "https://www.reddit.com"
        self.headers = {"User-Agent": "AITrendPulse/3.0"}
        self.subreddits = [
            "artificial", "MachineLearning", "OpenAI", "ChatGPT", 
            "singularity", "LocalLLaMA", "midjourney", "StableDiffusion",
            "ArtificialIntelligence", "deeplearning", "computervision",
            "LanguageTechnology", "reinforcementlearning", "datascience"
        ]
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def get_comprehensive_posts(_self, time_filter="day") -> List[Dict]:
        """Get posts from multiple AI-related subreddits"""
        all_posts = []
        
        for subreddit in _self.subreddits[:8]:  # Limit to avoid rate limiting
            posts = _self._fetch_subreddit_posts(subreddit, time_filter)
            all_posts.extend(posts)
        
        # Sort by score
        all_posts.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return all_posts[:50]  # Return top 50
    
    def _fetch_subreddit_posts(_self, subreddit: str, time_filter: str) -> List[Dict]:
        """Fetch posts from a specific subreddit"""
        posts = []
        
        try:
            url = f"{_self.base_url}/r/{subreddit}/top.json"
            params = {"t": time_filter, "limit": 10}
            
            response = requests.get(url, params=params, headers=_self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for post in data.get('data', {}).get('children', []):
                    post_data = post['data']
                    
                    processed_post = {
                        "title": post_data['title'],
                        "link": f"https://reddit.com{post_data['permalink']}",
                        "score": post_data.get('score', 0),
                        "comments": post_data.get('num_comments', 0),
                        "subreddit": post_data['subreddit'],
                        "author": post_data.get('author', 'deleted'),
                        "created_utc": post_data['created_utc'],
                        "url": post_data.get('url', ''),
                        "is_self": post_data.get('is_self', True),
                        "flair": post_data.get('link_flair_text', ''),
                        "upvote_ratio": post_data.get('upvote_ratio', 0)
                    }
                    posts.append(processed_post)
                    
        except Exception as e:
            logger.error(f"Error fetching r/{subreddit}: {e}")
            
        return posts

class EnhancedHackerNewsAPI:
    """Enhanced Hacker News data fetching"""
    
    def __init__(self):
        self.algolia_url = "https://hn.algolia.com/api/v1/search"
        self.firebase_url = "https://hacker-news.firebaseio.com/v0"
        self.ai_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'neural', 'gpt', 'llm', 'chatbot', 'openai', 'anthropic', 'claude',
            'gemini', 'langchain', 'vector', 'embedding', 'transformer', 'diffusion'
        ]
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def get_ai_stories(_self, limit=50) -> List[Dict]:
        """Get AI-related stories from Hacker News"""
        stories = []
        
        # Search for AI-related content
        for keyword in _self.ai_keywords[:5]:  # Limit searches
            keyword_stories = _self._search_stories(keyword)
            stories.extend(keyword_stories)
        
        # Remove duplicates
        seen_ids = set()
        unique_stories = []
        for story in stories:
            if story['id'] not in seen_ids:
                seen_ids.add(story['id'])
                unique_stories.append(story)
        
        # Sort by points
        unique_stories.sort(key=lambda x: x.get('points', 0), reverse=True)
        
        return unique_stories[:limit]
    
    def _search_stories(_self, query: str) -> List[Dict]:
        """Search for stories using Algolia API"""
        stories = []
        
        try:
            params = {
                "query": query,
                "tags": "story",
                "hitsPerPage": 20,
                "numericFilters": f"created_at_i>{int((datetime.now() - timedelta(days=7)).timestamp())}"
            }
            
            response = requests.get(_self.algolia_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for hit in data.get('hits', []):
                    story = {
                        "id": hit['objectID'],
                        "title": hit['title'],
                        "url": hit.get('url', f"https://news.ycombinator.com/item?id={hit['objectID']}"),
                        "points": hit.get('points', 0),
                        "author": hit.get('author', 'Unknown'),
                        "created_at": hit.get('created_at', ''),
                        "num_comments": hit.get('num_comments', 0),
                        "tags": query
                    }
                    stories.append(story)
                    
        except Exception as e:
            logger.error(f"Error searching HN for '{query}': {e}")
            
        return stories

class EnhancedAIToolsAPI:
    """Enhanced AI tools discovery from multiple sources"""
    
    def __init__(self):
        self.github_url = "https://api.github.com/search/repositories"
        self.headers = {"User-Agent": "AITrendPulse/3.0"}
    
    @st.cache_data(ttl=config.CACHE_TTL * 2)
    def get_comprehensive_tools(_self) -> Dict[str, List[Dict]]:
        """Get AI tools from multiple sources"""
        tools_by_category = {
            "🤖 LLMs & Chatbots": [],
            "🎨 Image & Video": [],
            "💼 Productivity": [],
            "🔧 Development": [],
            "🎵 Audio & Music": [],
            "📊 Data & Analytics": [],
            "🔬 Research & Papers": [],
            "🎮 Gaming & Entertainment": [],
            "🏥 Healthcare & Biology": [],
            "💰 Finance & Trading": []
        }
        
        # Fetch from GitHub
        github_tools = _self._fetch_github_tools()
        _self._categorize_tools(github_tools, tools_by_category)
        
        # Fetch from awesome lists
        awesome_tools = _self._fetch_awesome_lists()
        _self._categorize_tools(awesome_tools, tools_by_category)
        
        # Sort tools in each category by popularity
        for category in tools_by_category:
            tools_by_category[category].sort(
                key=lambda x: (x.get('trending', False), x.get('rating', 0), x.get('stars', 0)), 
                reverse=True
            )
            # Limit to top 20 per category
            tools_by_category[category] = tools_by_category[category][:20]
        
        return tools_by_category
    
    def _fetch_github_tools(_self) -> List[Dict]:
        """Fetch AI tools from GitHub"""
        tools = []
        
        # Search queries for different AI categories
        search_queries = [
            "ai tool stars:>100",
            "machine learning tool stars:>100",
            "llm tool stars:>100",
            "ai assistant stars:>100",
            "generative ai stars:>100",
            "ai api stars:>100",
            "prompt engineering stars:>100",
            "ai automation stars:>100"
        ]
        
        for query in search_queries[:5]:  # Limit API calls
            try:
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 20
                }
                
                response = requests.get(_self.github_url, params=params, headers=_self.headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for repo in data.get('items', []):
                        tool = {
                            "name": repo['name'].replace('-', ' ').replace('_', ' ').title(),
                            "url": repo['html_url'],
                            "description": repo.get('description', 'No description'),
                            "stars": repo['stargazers_count'],
                            "language": repo.get('language', 'Unknown'),
                            "updated": repo['updated_at'],
                            "topics": repo.get('topics', []),
                            "homepage": repo.get('homepage', ''),
                            "trending": repo['stargazers_count'] > 1000,
                            "rating": min(5.0, repo['stargazers_count'] / 5000),
                            "category": "tool",
                            "source": "GitHub"
                        }
                        tools.append(tool)
                        
            except Exception as e:
                logger.error(f"Error fetching GitHub tools for '{query}': {e}")
                
        return tools
    
    def _fetch_awesome_lists(_self) -> List[Dict]:
        """Fetch tools from awesome AI lists"""
        tools = []
        
        # Awesome AI lists on GitHub
        awesome_lists = [
            "https://raw.githubusercontent.com/steven2358/awesome-generative-ai/master/README.md",
            "https://raw.githubusercontent.com/mahseema/awesome-ai-tools/master/README.md",
            "https://raw.githubusercontent.com/ai-collection/ai-collection/main/README.md"
        ]
        
        for list_url in awesome_lists:
            try:
                response = requests.get(list_url, timeout=10)
                if response.status_code == 200:
                    tools.extend(_self._parse_awesome_list(response.text))
            except Exception as e:
                logger.error(f"Error fetching awesome list: {e}")
                
        return tools
    
    def _parse_awesome_list(_self, content: str) -> List[Dict]:
        """Parse tools from awesome list markdown"""
        tools = []
        
        # Simple regex to extract links and descriptions
        pattern = r'\[([^\]]+)\]\(([^\)]+)\)\s*[-–—]\s*([^\n]+)'
        matches = re.findall(pattern, content)
        
        for name, url, description in matches[:50]:  # Limit parsing
            if 'github.com' in url or 'http' in url:
                tool = {
                    "name": name.strip(),
                    "url": url.strip(),
                    "description": description.strip()[:200],
                    "stars": random.randint(100, 5000),  # Placeholder
                    "trending": random.choice([True, False]),
                    "rating": round(random.uniform(3.5, 5.0), 1),
                    "category": "tool",
                    "source": "Awesome List"
                }
                tools.append(tool)
                
        return tools
    
    def _categorize_tools(_self, tools: List[Dict], categories: Dict[str, List[Dict]]):
        """Categorize tools based on keywords"""
        for tool in tools:
            name_desc = f"{tool.get('name', '')} {tool.get('description', '')}".lower()
            
            if any(kw in name_desc for kw in ['chat', 'gpt', 'llm', 'language model', 'conversational']):
                categories["🤖 LLMs & Chatbots"].append(tool)
            elif any(kw in name_desc for kw in ['image', 'video', 'visual', 'diffusion', 'gan', 'photo']):
                categories["🎨 Image & Video"].append(tool)
            elif any(kw in name_desc for kw in ['productivity', 'writing', 'document', 'notes', 'workflow']):
                categories["💼 Productivity"].append(tool)
            elif any(kw in name_desc for kw in ['code', 'programming', 'developer', 'ide', 'debug']):
                categories["🔧 Development"].append(tool)
            elif any(kw in name_desc for kw in ['audio', 'music', 'voice', 'speech', 'sound']):
                categories["🎵 Audio & Music"].append(tool)
            elif any(kw in name_desc for kw in ['data', 'analytics', 'visualization', 'analysis', 'ml ops']):
                categories["📊 Data & Analytics"].append(tool)
            elif any(kw in name_desc for kw in ['research', 'paper', 'academic', 'science', 'arxiv']):
                categories["🔬 Research & Papers"].append(tool)
            elif any(kw in name_desc for kw in ['game', 'gaming', 'entertainment', 'fun', 'play']):
                categories["🎮 Gaming & Entertainment"].append(tool)
            elif any(kw in name_desc for kw in ['health', 'medical', 'biology', 'drug', 'diagnosis']):
                categories["🏥 Healthcare & Biology"].append(tool)
            elif any(kw in name_desc for kw in ['finance', 'trading', 'crypto', 'investment', 'market']):
                categories["💰 Finance & Trading"].append(tool)
            else:
                # Default to productivity
                categories["💼 Productivity"].append(tool)

class AIResearchAPI:
    """Fetch AI research papers and publications"""
    
    def __init__(self):
        self.arxiv_url = "http://export.arxiv.org/api/query"
        self.semantic_scholar_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    @st.cache_data(ttl=config.CACHE_TTL * 2)
    def get_latest_papers(_self, limit=20) -> List[Dict]:
        """Get latest AI research papers"""
        papers = []
        
        # Fetch from arXiv
        arxiv_papers = _self._fetch_arxiv_papers()
        papers.extend(arxiv_papers)
        
        # Sort by date
        papers.sort(key=lambda x: x['published'], reverse=True)
        
        return papers[:limit]
    
    def _fetch_arxiv_papers(_self) -> List[Dict]:
        """Fetch papers from arXiv"""
        papers = []
        
        # Search queries for different AI topics
        queries = [
            "cat:cs.AI",  # Artificial Intelligence
            "cat:cs.LG",  # Machine Learning
            "cat:cs.CV",  # Computer Vision
            "cat:cs.CL",  # Computation and Language
            "cat:cs.NE"   # Neural and Evolutionary Computing
        ]
        
        for query in queries:
            try:
                params = {
                    "search_query": query,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": 10
                }
                
                response = requests.get(_self.arxiv_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    # Parse XML response
                    feed = feedparser.parse(response.text)
                    
                    for entry in feed.entries:
                        paper = {
                            "title": entry.title,
                            "url": entry.link,
                            "authors": [author.name for author in entry.authors],
                            "summary": entry.summary[:300] + "...",
                            "published": datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ"),
                            "category": entry.arxiv_primary_category['term'],
                            "source": "arXiv"
                        }
                        papers.append(paper)
                        
            except Exception as e:
                logger.error(f"Error fetching arXiv papers: {e}")
                
        return papers

class AIJobsAPI:
    """Fetch AI-related job postings"""
    
    def __init__(self):
        self.headers = {"User-Agent": "AITrendPulse/3.0"}
    
    @st.cache_data(ttl=config.CACHE_TTL * 4)
    def get_ai_jobs(_self, limit=20) -> List[Dict]:
        """Get AI job postings from various sources"""
        jobs = []
        
        # Note: Most job APIs require authentication
        # This is a placeholder for demonstration
        # You would need to sign up for free tiers of job APIs
        
        # Example structure
        sample_jobs = [
            {
                "title": "Senior AI Engineer",
                "company": "Tech Corp",
                "location": "Remote",
                "salary": "$150k - $200k",
                "url": "#",
                "posted": datetime.now() - timedelta(days=1),
                "tags": ["Python", "TensorFlow", "MLOps"],
                "source": "Indeed"
            }
        ]
        
        return sample_jobs

class AIPodcastAPI:
    """Fetch AI podcast episodes"""
    
    def __init__(self):
        self.podcasts = config.AI_PODCAST_SOURCES
    
    @st.cache_data(ttl=config.CACHE_TTL * 2)
    def get_latest_episodes(_self, limit=20) -> List[Dict]:
        """Get latest AI podcast episodes"""
        episodes = []
        
        for podcast_name, feed_url in _self.podcasts.items():
            try:
                feed = feedparser.parse(feed_url)
                
                if feed.entries:
                    for entry in feed.entries[:3]:  # Get latest 3 per podcast
                        episode = {
                            "title": entry.title,
                            "url": entry.link,
                            "podcast": podcast_name.replace('_', ' ').title(),
                            "published": datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now(),
                            "duration": entry.get('itunes_duration', 'Unknown'),
                            "summary": entry.get('summary', '')[:200] + "...",
                            "source": "Podcast"
                        }
                        episodes.append(episode)
                        
            except Exception as e:
                logger.error(f"Error fetching {podcast_name}: {e}")
                
        # Sort by date
        episodes.sort(key=lambda x: x['published'], reverse=True)
        
        return episodes[:limit]

class AIEventsAPI:
    """Fetch AI conferences and events"""
    
    def __init__(self):
        self.headers = {"User-Agent": "AITrendPulse/3.0"}
    
    @st.cache_data(ttl=config.CACHE_TTL * 4)
    def get_upcoming_events(_self) -> List[Dict]:
        """Get upcoming AI events"""
        # This would require integration with event APIs
        # For now, return sample data
        events = [
            {
                "name": "NeurIPS 2025",
                "date": "December 2025",
                "location": "Vancouver, Canada",
                "url": "https://neurips.cc",
                "type": "Conference",
                "description": "Leading AI/ML research conference"
            },
            {
                "name": "AI Summit 2025",
                "date": "September 2025",
                "location": "San Francisco, USA",
                "url": "#",
                "type": "Summit",
                "description": "Industry AI applications and trends"
            }
        ]
        
        return events

# Utility functions
def format_number(num: int) -> str:
    """Format large numbers with K, M, B suffixes"""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)

def time_ago(timestamp: Union[datetime, float]) -> str:
    """Human-readable time difference"""
    if isinstance(timestamp, float):
        timestamp = datetime.fromtimestamp(timestamp)
    
    now = datetime.now()
    timestamp = timestamp.replace(tzinfo=None)  # Strip timezone info

    diff = now - timestamp
    
    if diff.days > 365:
        return f"{diff.days // 365} year{'s' if diff.days // 365 != 1 else ''} ago"
    elif diff.days > 30:
        return f"{diff.days // 30} month{'s' if diff.days // 30 != 1 else ''} ago"
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "Just now"