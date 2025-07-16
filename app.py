"""
AI TrendPulse - Main Application
Real-time AI news, tools, research, and community insights
"""
import streamlit as st
from datetime import datetime, timedelta, timezone
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import logging
import pytz

# Import custom modules
from config import config
from data_fetchers import (
    EnhancedAINewsAPI, EnhancedRedditAPI, EnhancedHackerNewsAPI,
    EnhancedAIToolsAPI, AIResearchAPI, AIPodcastAPI, AIEventsAPI,
    AIJobsAPI, format_number, time_ago
)
from ui_components import (
    UIComponents, NewsCard, RedditPostCard, HackerNewsCard,
    ResearchPaperCard, PodcastCard, EventCard, JobCard
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize APIs
@st.cache_resource
def initialize_apis():
    """Initialize all API instances"""
    return {
        "news": EnhancedAINewsAPI(),
        "reddit": EnhancedRedditAPI(),
        "hackernews": EnhancedHackerNewsAPI(),
        "tools": EnhancedAIToolsAPI(),
        "research": AIResearchAPI(),
        "podcasts": AIPodcastAPI(),
        "events": AIEventsAPI(),
        "jobs": AIJobsAPI()
    }

# Page configuration
st.set_page_config(
    page_title="🤖 AI TrendPulse - Real-time AI Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/ai-trendpulse',
        'Report a bug': 'https://github.com/yourusername/ai-trendpulse/issues',
        'About': 'AI TrendPulse - Your comprehensive AI intelligence platform'
    }
)

# Inject custom CSS and scripts
UIComponents.inject_custom_css()
UIComponents.inject_analytics()
UIComponents.inject_adsense_script()

# Initialize session state
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

# Initialize APIs
apis = initialize_apis()

# Render header
UIComponents.render_header()

# Header ad space
UIComponents.render_ad_space("header", "banner")

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    # Date range filter
    date_range = st.date_input(
        "📅 Date Range",
        value=(datetime.now() - timedelta(days=7), datetime.now()),
        max_value=datetime.now()
    )
    
    # Content type filter
    st.markdown("### 🔍 Content Types")
    show_news = st.checkbox("📰 AI News", value=True)
    show_reddit = st.checkbox("👽 Reddit Discussions", value=True)
    show_hn = st.checkbox("💻 Hacker News", value=True)
    show_tools = st.checkbox("🔧 AI Tools", value=True)
    show_research = st.checkbox("📄 Research Papers", value=True)
    show_podcasts = st.checkbox("🎙️ Podcasts", value=True)
    show_events = st.checkbox("🎯 Events", value=False)
    show_jobs = st.checkbox("💼 Jobs", value=False)
    
    # Refresh button
    if st.button("🔄 Refresh All Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.session_state.refresh_count += 1
        st.rerun()
    
    # Data sources status
    st.markdown("### 📡 Data Sources Status")
    
    status_indicators = {
        "📰 News Sources": len(config.AI_NEWS_SOURCES),
        "👽 Reddit Subs": 14,
        "💻 HN Integration": "Active",
        "🔧 Tool Sources": "Multiple",
        "📄 Research APIs": 3,
        "🎙️ Podcasts": len(config.AI_PODCAST_SOURCES)
    }
    
    for source, status in status_indicators.items():
        st.markdown(f"**{source}:** ✅ {status}")
    
    st.markdown("---")
    
    # About section
    st.markdown("### ℹ️ About")
    st.markdown("""
    **AI TrendPulse** aggregates real-time AI content from 50+ sources including:
    
    - Major AI news outlets
    - Reddit communities
    - Hacker News
    - GitHub repositories
    - Research papers
    - Podcasts & YouTube
    - Job boards
    - Events & conferences
    
    *Data refreshes every 5 minutes*
    """)
    
    # Sidebar ad
    UIComponents.render_ad_space("sidebar", "square")

# Main content area
# Analytics Overview
st.markdown("## 📊 Analytics Dashboard")

# Fetch initial data for metrics
with st.spinner("Loading real-time AI data..."):
    news_data = apis["news"].get_comprehensive_news() if show_news else []
    reddit_data = apis["reddit"].get_comprehensive_posts() if show_reddit else []
    hn_data = apis["hackernews"].get_ai_stories() if show_hn else []
    tools_data = apis["tools"].get_comprehensive_tools() if show_tools else {}
    research_data = apis["research"].get_latest_papers() if show_research else []
    podcast_data = apis["podcasts"].get_latest_episodes() if show_podcasts else []

# Display metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    UIComponents.render_metric_card("AI News", format_number(len(news_data)))

with col2:
    UIComponents.render_metric_card("Reddit Posts", format_number(len(reddit_data)))

with col3:
    UIComponents.render_metric_card("HN Stories", format_number(len(hn_data)))

with col4:
    total_tools = sum(len(tools) for tools in tools_data.values())
    UIComponents.render_metric_card("AI Tools", format_number(total_tools))

with col5:
    UIComponents.render_metric_card("Research Papers", format_number(len(research_data)))

with col6:
    UIComponents.render_metric_card("Podcasts", format_number(len(podcast_data)))

# Visualizations
if news_data:
    col1, col2 = st.columns(2)
    
    with col1:
        # News sources distribution
        source_counts = {}
        for article in news_data:
            source = article['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        fig_sources = px.pie(
            values=list(source_counts.values()),
            names=list(source_counts.keys()),
            title="AI News Distribution by Source",
            color_discrete_sequence=px.colors.sequential.Purples
        )
        fig_sources.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0',
            showlegend=True
        )
        st.plotly_chart(fig_sources, use_container_width=True)
    
    with col2:
        # Content type distribution
        type_counts = {}
        for article in news_data:
            content_type = article.get('type', 'article')
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        fig_types = px.bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            title="Content Types Distribution",
            color=list(type_counts.values()),
            color_continuous_scale='Viridis'
        )
        fig_types.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0',
            showlegend=False
        )
        st.plotly_chart(fig_types, use_container_width=True)

# Content tabs
tabs = []
tab_names = []

if show_news:
    tabs.append("news")
    tab_names.append("📰 AI News")
    
if show_reddit:
    tabs.append("reddit")
    tab_names.append("👽 Reddit")
    
if show_hn:
    tabs.append("hn")
    tab_names.append("💻 Hacker News")
    
if show_tools:
    tabs.append("tools")
    tab_names.append("🔧 AI Tools")
    
if show_research:
    tabs.append("research")
    tab_names.append("📄 Research")
    
if show_podcasts:
    tabs.append("podcasts")
    tab_names.append("🎙️ Podcasts")
    
if show_events:
    tabs.append("events")
    tab_names.append("🎯 Events")
    
if show_jobs:
    tabs.append("jobs")
    tab_names.append("💼 Jobs")

if tabs:
    tab_objects = st.tabs(tab_names)
    
    for i, (tab_key, tab) in enumerate(zip(tabs, tab_objects)):
        with tab:
            if tab_key == "news":
                UIComponents.render_section_header(
                    "📰", "Latest AI News",
                    f"Real-time updates from {len(config.AI_NEWS_SOURCES)} trusted sources"
                )
                
                # Filter news by date
                filtered_news = []
                for article in news_data:
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        start_date, end_date = date_range
                        # Convert article date to date only for comparison
                        article_date = article['published'].date()
                        if hasattr(article['published'], 'date'):
                            article_date = article['published'].date()
                        if start_date <= article_date <= end_date:
                            filtered_news.append(article)
                
                # Display news with ads
                for idx, article in enumerate(filtered_news[:50]):
                    NewsCard.render(article, show_ad=(idx + 1) % config.MONETIZATION["ad_frequency"] == 0)
                
            elif tab_key == "reddit":
                UIComponents.render_section_header(
                    "👽", "Reddit AI Discussions",
                    "Trending posts from AI communities"
                )
                
                # Time filter
                time_filter = st.selectbox(
                    "Time Period",
                    ["hour", "day", "week", "month", "year"],
                    index=1
                )
                
                # Re-fetch with filter
                reddit_data = apis["reddit"].get_comprehensive_posts(time_filter)
                
                # Display posts
                for idx, post in enumerate(reddit_data[:30]):
                    RedditPostCard.render(post)
                    if (idx + 1) % config.MONETIZATION["ad_frequency"] == 0:
                        UIComponents.render_ad_space("content", "rectangle")
                
            elif tab_key == "hn":
                UIComponents.render_section_header(
                    "💻", "Hacker News AI Stories",
                    "Top AI discussions from the HN community"
                )
                
                # Display stories
                for idx, story in enumerate(hn_data[:30]):
                    HackerNewsCard.render(story)
                    if (idx + 1) % config.MONETIZATION["ad_frequency"] == 0:
                        UIComponents.render_ad_space("content", "rectangle")
                
            elif tab_key == "tools":
                UIComponents.render_section_header(
                    "🔧", "AI Tools Directory",
                    "Discover the latest AI tools and applications"
                )
                
                # Tool categories
                for category, tools in tools_data.items():
                    if tools:
                        st.markdown(f"### {category}")
                        
                        # Create grid layout
                        cols = st.columns(3)
                        for idx, tool in enumerate(tools):
                            with cols[idx % 3]:
                                UIComponents.render_tool_card(tool)
                        
                        # Ad after each category
                        UIComponents.render_ad_space("content", "banner")
                
            elif tab_key == "research":
                UIComponents.render_section_header(
                    "📄", "Latest AI Research",
                    "Recent papers from arXiv and other sources"
                )
                
                # Display papers
                for idx, paper in enumerate(research_data):
                    ResearchPaperCard.render(paper)
                    if (idx + 1) % config.MONETIZATION["ad_frequency"] == 0:
                        UIComponents.render_ad_space("content", "rectangle")
                
            elif tab_key == "podcasts":
                UIComponents.render_section_header(
                    "🎙️", "AI Podcasts",
                    "Latest episodes from top AI podcasts"
                )
                
                # Display episodes
                for idx, episode in enumerate(podcast_data):
                    PodcastCard.render(episode)
                    if (idx + 1) % config.MONETIZATION["ad_frequency"] == 0:
                        UIComponents.render_ad_space("content", "rectangle")
                
            elif tab_key == "events":
                UIComponents.render_section_header(
                    "🎯", "AI Events & Conferences",
                    "Upcoming AI events worldwide"
                )
                
                events_data = apis["events"].get_upcoming_events()
                for event in events_data:
                    EventCard.render(event)
                
            elif tab_key == "jobs":
                UIComponents.render_section_header(
                    "💼", "AI Job Opportunities",
                    "Latest AI and ML job postings"
                )
                
                jobs_data = apis["jobs"].get_ai_jobs()
                for job in jobs_data:
                    JobCard.render(job)

# Footer
st.markdown("---")

# Footer ad
UIComponents.render_ad_space("footer", "banner")

# Footer content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h3 style="color: var(--primary-color);">🤖 AI TrendPulse</h3>
        <p style="color: var(--text-secondary); margin-top: 1rem;">
            Your comprehensive source for real-time AI intelligence
        </p>
        <p style="color: var(--text-secondary); margin-top: 0.5rem;">
            Aggregating content from 50+ trusted sources • Updated every 5 minutes
        </p>
        <p style="margin-top: 2rem; color: var(--text-secondary);">
            <a href="https://github.com/yourusername/ai-trendpulse" target="_blank">GitHub</a> • 
            <a href="https://twitter.com/aitrendpulse" target="_blank">Twitter</a> • 
            <a href="mailto:contact@aitrendpulse.com">Contact</a> • 
            <a href="/privacy">Privacy Policy</a>
        </p>
        <p style="margin-top: 1rem; color: var(--text-secondary); font-size: 0.9rem;">
            © 2025 AI TrendPulse by Arnold Sajjan
        </p>
    </div>
    """, unsafe_allow_html=True)

# Auto-refresh
if st.sidebar.checkbox("⚡ Auto-refresh", value=False):
    import time
    time.sleep(300)  # Refresh every 5 minutes
    st.rerun()