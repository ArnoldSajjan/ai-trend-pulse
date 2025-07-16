"""
UI Components and Enhanced Styling for AI TrendPulse
"""
import streamlit as st
from config import config
import random

class UIComponents:
    """Reusable UI components with consistent styling"""
    
    @staticmethod
    def inject_custom_css():
        """Inject custom CSS for enhanced UI"""
        st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* Global Styles */
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #48bb78;
            --warning-color: #f6ad55;
            --error-color: #fc8181;
            --dark-bg: #0a0a0a;
            --card-bg: #1a1a1a;
            --border-color: #2d2d2d;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --gradient-4: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }
        
        /* Main App Background */
        .stApp {
            background: var(--dark-bg);
            background-image: 
                radial-gradient(at 20% 80%, rgba(102, 126, 234, 0.1) 0px, transparent 50%),
                radial-gradient(at 80% 20%, rgba(118, 75, 162, 0.1) 0px, transparent 50%),
                radial-gradient(at 40% 40%, rgba(102, 126, 234, 0.05) 0px, transparent 50%);
            color: var(--text-primary);
            font-family: 'Inter', system-ui, sans-serif;
        }
        
        /* Header Styles */
        .main-header {
            background: var(--gradient-1);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        }
        
        .main-header::before {
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 4s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 0.3; }
        }
        
        .main-title {
            font-size: 4rem;
            font-weight: 800;
            margin: 0;
            position: relative;
            z-index: 1;
            text-shadow: 0 4px 15px rgba(0,0,0,0.3);
            letter-spacing: -2px;
        }
        
        .main-subtitle {
            font-size: 1.3rem;
            margin-top: 1rem;
            opacity: 0.95;
            position: relative;
            z-index: 1;
            font-weight: 400;
        }
        
        /* Card Styles */
        .content-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .content-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-1);
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
        }
        
        .content-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
            border-color: rgba(102, 126, 234, 0.3);
        }
        
        .content-card:hover::before {
            transform: scaleX(1);
        }
        
        /* Tool Cards Grid */
        .tools-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        .tool-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .tool-card.trending::after {
            content: "🔥 TRENDING";
            position: absolute;
            top: 1rem;
            right: -2rem;
            background: var(--gradient-2);
            color: white;
            padding: 0.25rem 3rem;
            font-size: 0.75rem;
            font-weight: 600;
            transform: rotate(45deg);
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        /* Badges */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .badge-primary {
            background: var(--gradient-1);
            color: white;
        }
        
        .badge-success {
            background: var(--gradient-4);
            color: white;
        }
        
        .badge-warning {
            background: var(--gradient-2);
            color: white;
        }
        
        .badge-info {
            background: var(--gradient-3);
            color: white;
        }
        
        /* Metrics */
        .metric-container {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-container:hover {
            transform: translateY(-2px);
            border-color: rgba(102, 126, 234, 0.3);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .metric-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 0.5rem;
            gap: 0.5rem;
            border: 1px solid var(--border-color);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            color: var(--text-secondary);
            font-weight: 500;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(102, 126, 234, 0.1);
            color: var(--text-primary);
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--gradient-1);
            color: white;
            border-color: transparent;
        }
        
        /* Links */
        a {
            color: #667eea !important;
            text-decoration: none;
            transition: all 0.2s ease;
            position: relative;
        }
        
        a:hover {
            color: #8da5ff !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: var(--gradient-1);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: var(--card-bg);
            border-right: 1px solid var(--border-color);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
            animation: fadeIn 0.6s ease-out;
        }
        
        /* Loading Animation */
        .loading-pulse {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--primary-color);
            animation: pulse-dot 1.5s ease-in-out infinite;
        }
        
        @keyframes pulse-dot {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.5); opacity: 0.5; }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--dark-bg);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-color);
        }
        
        /* Ad Container Styles */
        .ad-container {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            margin: 2rem 0;
            text-align: center;
            position: relative;
            min-height: 100px;
        }
        
        .ad-label {
            position: absolute;
            top: 0.5rem;
            left: 0.5rem;
            font-size: 0.7rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2.5rem;
            }
            
            .tools-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render main header"""
        st.markdown("""
        <div class="main-header animate-fade-in">
            <h1 class="main-title">🤖 AI TrendPulse</h1>
            <p class="main-subtitle">Real-time AI news, trending tools, research papers, and community discussions</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_ad_space(slot_name: str, size: str = "responsive"):
        """Render Google AdSense ad space"""
        if config.MONETIZATION["show_ads"]:
            ad_sizes = {
                "responsive": "style='display:block'",
                "rectangle": "style='display:inline-block;width:336px;height:280px'",
                "banner": "style='display:inline-block;width:728px;height:90px'",
                "square": "style='display:inline-block;width:250px;height:250px'"
            }
            
            st.markdown(f"""
            <div class="ad-container">
                <span class="ad-label">Advertisement</span>
                <ins class="adsbygoogle"
                     {ad_sizes.get(size, ad_sizes['responsive'])}
                     data-ad-client="{config.GOOGLE_ADSENSE_CLIENT}"
                     data-ad-slot="{config.GOOGLE_ADSENSE_SLOTS.get(slot_name, '')}"
                     data-ad-format="auto"
                     data-full-width-responsive="true"></ins>
                <script>
                     (adsbygoogle = window.adsbygoogle || []).push({{}});
                </script>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_metric_card(label: str, value: str, trend: str = None):
        """Render a metric card"""
        trend_html = f"<div class='metric-trend'>{trend}</div>" if trend else ""
        
        st.markdown(f"""
        <div class="metric-container animate-fade-in">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {trend_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_content_card(title: str, content: str, metadata: dict = None):
        """Render a content card"""
        metadata_html = ""
        if metadata:
            badges = []
            for key, value in metadata.items():
                badge_class = "badge-primary"
                if key == "source":
                    badge_class = "badge-info"
                elif key == "trending":
                    badge_class = "badge-warning"
                badges.append(f'<span class="badge {badge_class}">{value}</span>')
            metadata_html = f"<div class='card-metadata'>{''.join(badges)}</div>"
        
        st.markdown(f"""
        <div class="content-card animate-fade-in">
            <h3>{title}</h3>
            <p>{content}</p>
            {metadata_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_tool_card(tool: dict):
        """Render a tool card"""
        trending_class = "trending" if tool.get("trending", False) else ""
        rating_stars = "⭐" * int(tool.get("rating", 0)) if tool.get("rating", 0) > 0 else ""
        
        st.markdown(f"""
        <div class="tool-card {trending_class} animate-fade-in">
            <h4><a href="{tool['url']}" target="_blank">{tool['name']}</a></h4>
            <p>{tool['description']}</p>
            <div class="tool-meta">
                <span class="badge badge-primary">{tool.get('category', 'Tool')}</span>
                {f'<span class="badge badge-success">{rating_stars}</span>' if rating_stars else ''}
                {f'<span class="badge badge-info">{tool.get("users", "")}</span>' if tool.get("users") else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_loading_animation():
        """Render loading animation"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div class="loading-pulse"></div>
            <p style="color: var(--text-secondary); margin-top: 1rem;">Loading fresh AI content...</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_section_header(icon: str, title: str, subtitle: str = ""):
        """Render section header"""
        st.markdown(f"""
        <div class="section-header animate-fade-in">
            <h2 style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">{icon}</span>
                {title}
            </h2>
            {f'<p style="color: var(--text-secondary); margin-top: 0.5rem;">{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def inject_analytics():
        """Inject analytics scripts"""
        if config.ANALYTICS["google_analytics_id"]:
            st.markdown(f"""
            <script async src="https://www.googletagmanager.com/gtag/js?id={config.ANALYTICS['google_analytics_id']}"></script>
            <script>
              window.dataLayer = window.dataLayer || [];
              function gtag(){{dataLayer.push(arguments);}}
              gtag('js', new Date());
              gtag('config', '{config.ANALYTICS['google_analytics_id']}');
            </script>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def inject_adsense_script():
        """Inject Google AdSense script"""
        if config.MONETIZATION["show_ads"]:
            st.markdown(f"""
            <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={config.GOOGLE_ADSENSE_CLIENT}"
                 crossorigin="anonymous"></script>
            """, unsafe_allow_html=True)

class NewsCard:
    """Component for rendering news articles"""
    
    @staticmethod
    def render(article: dict, show_ad: bool = False):
        """Render a news article card"""
        from data_fetchers import time_ago
        
        # Article type emoji
        type_emoji = {
            "article": "📰",
            "video": "📺",
            "podcast": "🎙️",
            "discussion": "💬",
            "paper": "📄"
        }
        
        emoji = type_emoji.get(article.get("type", "article"), "📰")
        
        # Build metadata badges
        badges = []
        badges.append(f'<span class="badge badge-info">{emoji} {article["source"]}</span>')
        badges.append(f'<span class="badge badge-primary">👤 {article.get("author", "Unknown")}</span>')
        badges.append(f'<span class="badge badge-success">🕒 {time_ago(article["published"])}</span>')
        
        if article.get("tags"):
            for tag in article["tags"][:3]:  # Show max 3 tags
                badges.append(f'<span class="badge badge-warning">#{tag}</span>')
        
        st.markdown(f"""
        <div class="content-card animate-fade-in">
            <h3><a href="{article['link']}" target="_blank">{article['title']}</a></h3>
            <p style="margin: 1rem 0; color: var(--text-secondary);">{article['summary']}</p>
            <div class="card-metadata">
                {''.join(badges)}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show ad if needed
        if show_ad:
            UIComponents.render_ad_space("content", "rectangle")

class RedditPostCard:
    """Component for rendering Reddit posts"""
    
    @staticmethod
    def render(post: dict):
        """Render a Reddit post card"""
        from data_fetchers import format_number, time_ago
        
        # Build metadata
        metadata = {
            f"r/{post['subreddit']}": "subreddit",
            f"👍 {format_number(post['score'])}": "score",
            f"💬 {format_number(post['comments'])}": "comments",
            f"by u/{post['author']}": "author"
        }
        
        if post.get('flair'):
            metadata[post['flair']] = "flair"
        
        # Render card
        st.markdown(f"""
        <div class="content-card animate-fade-in">
            <h3><a href="{post['link']}" target="_blank">{post['title']}</a></h3>
            <div class="card-metadata" style="margin-top: 1rem;">
                {''.join([f'<span class="badge badge-primary">{k}</span>' for k in metadata.keys()])}
            </div>
            <p style="margin-top: 0.5rem; color: var(--text-secondary); font-size: 0.9rem;">
                Posted {time_ago(post['created_utc'])} • {int(post.get('upvote_ratio', 0) * 100)}% upvoted
            </p>
        </div>
        """, unsafe_allow_html=True)

class HackerNewsCard:
    """Component for rendering Hacker News stories"""
    
    @staticmethod
    def render(story: dict):
        """Render a Hacker News story card"""
        from data_fetchers import format_number
        
        # Build URL
        story_url = story.get('url', f"https://news.ycombinator.com/item?id={story['id']}")
        comments_url = f"https://news.ycombinator.com/item?id={story['id']}"
        
        st.markdown(f"""
        <div class="content-card animate-fade-in">
            <h3><a href="{story_url}" target="_blank">{story['title']}</a></h3>
            <div class="card-metadata" style="margin-top: 1rem;">
                <span class="badge badge-warning">🔥 {format_number(story.get('points', 0))} points</span>
                <span class="badge badge-info">💬 {format_number(story.get('num_comments', 0))} comments</span>
                <span class="badge badge-primary">👤 {story.get('author', 'Unknown')}</span>
                <a href="{comments_url}" target="_blank" class="badge badge-success">View Discussion</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

class ResearchPaperCard:
    """Component for rendering research papers"""
    
    @staticmethod
    def render(paper: dict):
        """Render a research paper card"""
        # Format authors
        authors = ", ".join(paper.get('authors', ['Unknown'])[:3])
        if len(paper.get('authors', [])) > 3:
            authors += f" +{len(paper['authors']) - 3} more"
        
        st.markdown(f"""
        <div class="content-card animate-fade-in">
            <h3><a href="{paper['url']}" target="_blank">📄 {paper['title']}</a></h3>
            <p style="margin: 0.5rem 0; color: var(--text-secondary); font-style: italic;">{authors}</p>
            <p style="margin: 1rem 0; color: var(--text-secondary);">{paper['summary']}</p>
            <div class="card-metadata">
                <span class="badge badge-info">{paper['source']}</span>
                <span class="badge badge-primary">{paper.get('category', 'cs.AI')}</span>
                <span class="badge badge-success">{paper['published'].strftime('%Y-%m-%d')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

class PodcastCard:
    """Component for rendering podcast episodes"""
    
    @staticmethod
    def render(episode: dict):
        """Render a podcast episode card"""
        st.markdown(f"""
        <div class="content-card animate-fade-in">
            <h3><a href="{episode['url']}" target="_blank">🎙️ {episode['title']}</a></h3>
            <p style="margin: 0.5rem 0; color: var(--text-secondary); font-weight: 600;">{episode['podcast']}</p>
            <p style="margin: 1rem 0; color: var(--text-secondary);">{episode['summary']}</p>
            <div class="card-metadata">
                <span class="badge badge-info">🎧 {episode.get('duration', 'Unknown duration')}</span>
                <span class="badge badge-success">{episode['published'].strftime('%Y-%m-%d')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

class EventCard:
    """Component for rendering events"""
    
    @staticmethod
    def render(event: dict):
        """Render an event card"""
        st.markdown(f"""
        <div class="content-card animate-fade-in">
            <h3><a href="{event['url']}" target="_blank">🎯 {event['name']}</a></h3>
            <p style="margin: 1rem 0; color: var(--text-secondary);">{event['description']}</p>
            <div class="card-metadata">
                <span class="badge badge-warning">📅 {event['date']}</span>
                <span class="badge badge-info">📍 {event['location']}</span>
                <span class="badge badge-primary">{event['type']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

class JobCard:
    """Component for rendering job postings"""
    
    @staticmethod
    def render(job: dict):
        """Render a job posting card"""
        tags_html = ''.join([f'<span class="badge badge-info">{tag}</span>' for tag in job.get('tags', [])])
        
        st.markdown(f"""
        <div class="content-card animate-fade-in">
            <h3><a href="{job['url']}" target="_blank">💼 {job['title']}</a></h3>
            <p style="margin: 0.5rem 0;">
                <span style="font-weight: 600; color: var(--primary-color);">{job['company']}</span>
                <span style="color: var(--text-secondary);"> • {job['location']}</span>
                {f'<span style="color: var(--success-color); font-weight: 600;"> • {job["salary"]}</span>' if job.get('salary') else ''}
            </p>
            <div class="card-metadata" style="margin-top: 1rem;">
                {tags_html}
                <span class="badge badge-success">Posted {job['posted'].strftime('%Y-%m-%d')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)