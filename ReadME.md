Twitter Posting Agent
This repository contains a Twitter Posting Agent. The agent automatically generates and posts tweets to X (Twitter) at regular intervals without manual intervention, leveraging an AI-powered tweet generator (distilgpt2) and the X API v2 (tweepy). 
It fulfills all task requirements and includes the optional AI bonus task for dynamic tweet generation.

Project Overview
The Twitter Posting Agent:

Generates concise, engaging tweets using distilgpt2 from predefined topic prompts.
Posts two tweets sequentially. 
Runs autonomously.

Key Features
Automated Posting: Posts tweets at regular intervals without intervention.
Dynamic Content: Uses distilgpt2 to create single-sentence tweets (<280 characters) from a topics.json file.
Robust Design: Includes logging, error handling, and a modular structure for scalability.

Task Requirements
✅ Automatic Posting: Tweets are posted at fixed intervals without manual input.
✅ Predefined Source: Content is sourced from topics.json, enhanced by AI generation.
✅ Continuous Operation: Runs fully autonomously.

Bonus Task
✅ AI-Powered Generation: Integrates distilgpt2 to dynamically generate engaging, context-aware tweets instead of static text.

Evaluation Criteria
Functionality: Successfully posts AI-generated tweets, verified by Tweet IDs in logs.
Code Quality: Well-documented with inline comments, structured with a TweetGenerator class, and uses logging for transparency.
Scalability: Easily extensible—add new topics via add_topic(), adjust intervals in main(), or scale to more tweets.
Efficiency: Uses lightweight distilgpt2 (~82M parameters, ~500MB VRAM) for fast generation, optimized with GPU support.

Prerequisites
Python: 3.8 or higher
Libraries:
transformers (for distilgpt2)
torch (for model inference)
tweepy>=4.10.0 (for X API v2)
X Developer Account: API credentials with read/write access (Free tier or higher)


Copy
API_KEY = "your-api-key"
API_SECRET_KEY = "your-api-secret-key"
BEARER_TOKEN = "your-bearer-token"
ACCESS_TOKEN = "your-access-token"
ACCESS_TOKEN_SECRET = "your-access-token-secret"
CLIENT_ID = "your-client-id"
CLIENT_SECRET = "your-client-secret"
Ensure your X app has write permissions (Free tier allows 500 posts/month as of 2025).

How It Works
Initialization:
Loads distilgpt2 and topics from topics.json (auto-created if missing with 8 default topics).
Tweet Generation:
Uses generate_engaging_tweet() to produce a single-sentence tweet (60-200 characters) from a random topic prompt.
Applies strict cleaning (_clean_tweet) to ensure conciseness and proper format.
Posting:
Posts tweets via tweepy.Client (X API v2) .
Execution:
Runs autonomously, logging each step and confirming completion.
File Structure
twitter_posting_agent.py: Main script with the TweetGenerator class, generation logic, and posting functionality.
topics.json: Auto-generated file storing topic prompts (e.g., "AI is evolving beyond automation, unlocking").
