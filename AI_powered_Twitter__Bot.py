# Ensure required packages are installed before running the script
# !pip install tweepy schedule transformers torch requests

import os  # For file operations
import json  # For handling JSON topic files
import logging  # For logging execution details
import random  # For random topic selection
import torch  # For PyTorch-based model inference
import warnings  # For suppressing unnecessary warnings
import time  # For adding delay between tweets
from typing import List, Dict, Optional, Set  # For type hints
from transformers import pipeline, logging as transformers_logging  # For AI model and logging control
import tweepy  # For X API interaction

# Twitter API Credentials - Replace with your own before running
API_KEY = "lgKfDigC7qbos2zUIQeCvfbph"  # API Key for X authentication
API_SECRET_KEY = "NkkFO9wCVvsvv4m6GOssoXC0rCcE4qUav0rvVvM5xhWuJIkod2"  # API Secret Key for X authentication
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAALbPzwEAAAAA55b7v5kLBGPx1JyibiHmBQpmxoo%3Dhzw1o30lEajXtkuinTH1IAmjnIE4pILWy8e1wklNvWRA6N8j7f"  # Bearer Token for v2 API access
ACCESS_TOKEN = "1674821745340211200-UQzdIVGRCTSt7D5411vpxV9ApP9msO"  # Access Token for user-level auth
ACCESS_TOKEN_SECRET = "EjwT4klBT9nelP3aDonZz6hcruZ6IciO987TpRnnPFkif"  # Access Token Secret for user-level auth
CLIENT_ID = "UFAwM1M1aWs2TVgyZ2F6R3Z4RzU6MTpjaQ"  # Client ID for OAuth 2.0 (unused here but included)
CLIENT_SECRET = "yy0c-4hcra6OJz0RodTF60gWAg2Uj802bWKUl3XErsUzJUb06n"  # Client Secret for OAuth 2.0 (unused here)

# Suppress warnings and non-critical logging from libraries
warnings.filterwarnings("ignore")  # Ignore Python warnings
transformers_logging.set_verbosity_error()  # Show only critical errors from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism warnings

# Configure custom logging for the application
logging.basicConfig(
    level=logging.INFO,  # Log INFO level and above for detailed output
    format='%(asctime)s - %(levelname)s - %(message)s',  # Timestamp, level, message format
    handlers=[logging.StreamHandler()]  # Output logs to console
)
logger = logging.getLogger(__name__)  # Logger instance for this module

class TweetGenerator:
    def __init__(
        self,
        model_name: str = "distilgpt2",  # Lightweight, fast model for tweet generation
        topics_file: str = "topics.json",  # File storing predefined topic prompts
        max_tweet_length: int = 280,  # Maximum tweet length per X rules
        cache_dir: Optional[str] = None  # Optional directory for caching model files (unused)
    ):
        """Initialize the TweetGenerator with model and topic settings."""
        self.topics_file = topics_file  # Path to topics JSON file
        self.max_tweet_length = max_tweet_length  # Max allowed characters per tweet
        self.model_name = model_name  # Name of the AI model to use
        self.cache_dir = cache_dir  # Cache directory (not utilized here)
        self.topics = self._load_static_topics()  # Load or initialize topics
        self.used_topics: Set[int] = set()  # Set to track used topic indices
        self._setup_generator()  # Setup the AI model pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"  # Choose GPU if available
        logger.info(f"Using device: {device}")  # Log the selected device

    def _setup_generator(self):
        """Initialize the distilgpt2 model pipeline for text generation."""
        try:
            logger.info(f"Loading {self.model_name} model...")  # Log model loading start
            with warnings.catch_warnings():  # Suppress warnings during setup
                warnings.simplefilter("ignore")
                self.generator = pipeline(
                    "text-generation",  # Task type for the pipeline
                    model=self.model_name,  # Specific model to load
                    device_map="auto",  # Automatically map to available device
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # Optimize for GPU
                )
            logger.info("Model initialized successfully.")  # Confirm successful initialization
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")  # Log any errors
            raise  # Re-raise exception to stop execution

    def _load_static_topics(self) -> List[Dict[str, str]]:
        """Load topics from file or create defaults if file doesn't exist."""
        try:
            if not os.path.exists(self.topics_file):  # Check if topics file exists
                sample_topics = [  # Default topics with prompts for initial setup
                    {"topic": "Future of AI", "prompt": "AI is evolving beyond automation, unlocking"},
                    {"topic": "Immersive Gaming", "prompt": "Gaming is shifting towards hyper-realistic worlds with"},
                    {"topic": "Decentralized Finance", "prompt": "DeFi is transforming traditional banking by"},
                    {"topic": "Code Optimization", "prompt": "Writing efficient code isn't just about speed, it's about"},
                    {"topic": "Next-Gen Cybersecurity", "prompt": "Tomorrow's digital threats demand proactive security like"},
                    {"topic": "The Rise of Hybrid Work", "prompt": "Blending remote and in-office work successfully depends on"},
                    {"topic": "AI in Marketing", "prompt": "AI-driven marketing isn't just about ads, it's about"},
                    {"topic": "The App Revolution", "prompt": "Smart apps are becoming personal assistants by"}
                ]
                dir_path = os.path.dirname(os.path.abspath(self.topics_file))  # Get directory path
                if dir_path and not os.path.exists(dir_path):  # Create directory if needed
                    os.makedirs(dir_path, exist_ok=True)
                with open(self.topics_file, 'w') as f:  # Save default topics to file
                    json.dump(sample_topics, f, indent=2)
                logger.info(f"Created new topics file with {len(sample_topics)} topics")
            with open(self.topics_file, 'r') as f:  # Load existing topics
                topics = json.load(f)
            logger.info(f"Loaded {len(topics)} topics from {self.topics_file}")
            return topics
        except Exception as e:
            logger.error(f"Failed to load static topics: {str(e)}")
            raise

    def _get_next_topic(self) -> Optional[Dict[str, str]]:
        """Select a random unused topic, resetting if all are used."""
        if not self.topics:  # Check if topics list is empty
            logger.error("No topics available")
            return None
        if len(self.used_topics) >= len(self.topics):  # Reset if all topics used
            logger.info("All topics have been used. Resetting.")
            self.used_topics.clear()
        available_topics = [i for i in range(len(self.topics)) if i not in self.used_topics]  # Get unused indices
        if available_topics:
            topic_index = random.choice(available_topics)  # Pick a random unused topic
            self.used_topics.add(topic_index)  # Mark as used
            return self.topics[topic_index]
        return None

    def _clean_tweet(self, text: str) -> str:
        """Clean text to a single, concise sentence under 280 characters."""
        text = text.strip()  # Remove leading/trailing whitespace
        sentences = text.split('.')  # Split into sentences
        if len(sentences) > 1:  # Take only the first sentence
            text = sentences[0].strip() + '.'
        text = ''.join(c for c in text if c.isalnum() or c in ' .,!?')  # Filter to valid characters
        if text and text[0].islower():  # Capitalize first letter
            text = text[0].upper() + text[1:]
        if text and not text[-1] in ['.', '!', '?']:  # Ensure ending punctuation
            text += '.'
        if len(text) > self.max_tweet_length:  # Truncate if over limit
            text = text[:self.max_tweet_length-1] + '.'
        return text

    def _generate_tweet_content(self, topic: Dict[str, str]) -> Optional[str]:
        """Generate a clean, concise tweet from a topic prompt."""
        try:
            prompt = topic['prompt']  # Use prompt as the starting point
            with warnings.catch_warnings():  # Suppress generation warnings
                warnings.simplefilter("ignore")
                outputs = self.generator(
                    prompt,
                    max_new_tokens=30,  # Limit tokens for concise output
                    temperature=0.7,  # Lower value for coherent text
                    top_p=0.9,  # Tighter sampling for focused output
                    num_return_sequences=1,  # Generate one tweet
                    do_sample=True,  # Enable sampling for variety
                    pad_token_id=50256  # Avoid padding issues
                )
            generated_text = outputs[0]['generated_text']  # Extract generated text
            tweet = self._clean_tweet(generated_text)  # Clean to single sentence
            logger.info(f"Generated tweet content: {tweet} ({len(tweet)} characters)")
            return tweet
        except Exception as e:
            logger.error(f"Failed to generate tweet content: {str(e)}")
            return None

    def generate_tweet(self) -> Optional[str]:
        """Generate a single tweet from a random topic."""
        topic = self._get_next_topic()  # Select a topic
        if not topic:
            logger.warning("No topic available")
            return None
        tweet_content = self._generate_tweet_content(topic)  # Generate tweet
        if not tweet_content:
            logger.warning("Failed to generate tweet content")
            return None
        logger.info(f"Generated tweet ({len(tweet_content)} chars): {tweet_content}")
        return tweet_content
    
    def generate_engaging_tweet(self) -> Optional[str]:
        """Generate an engaging tweet by selecting the best from 3 attempts."""
        best_tweet = None
        best_length = 0
        for _ in range(3):  # Try 3 times to find a good tweet
            topic = self._get_next_topic()
            if not topic:
                continue
            tweet = self._generate_tweet_content(topic)
            if not tweet:
                continue
            tweet_length = len(tweet)
            if (best_tweet is None) or (60 <= tweet_length <= 200 and tweet_length > best_length):  # Prefer concise, engaging length
                best_tweet = tweet
                best_length = tweet_length
        if best_tweet:
            logger.info(f"Selected best tweet: {best_tweet}")
            return best_tweet
        else:
            return self.generate_tweet()  # Fallback to basic generation if needed

    def generate_tweets_batch(self, count: int = 3) -> List[str]:
        """Generate multiple tweets (unused in main flow)."""
        tweets = []
        for _ in range(count):
            tweet = self.generate_tweet()
            if tweet:
                tweets.append(tweet)
        return tweets
    
    def add_topic(self, topic: str, prompt: str) -> bool:
        """Add a new topic to the topics file for scalability."""
        try:
            new_topic = {"topic": topic, "prompt": prompt}  # Create new topic dict
            self.topics.append(new_topic)  # Add to in-memory list
            with open(self.topics_file, 'w') as f:  # Update file
                json.dump(self.topics, f, indent=2)
            logger.info(f"Added new topic: {topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to add topic: {str(e)}")
            return False

def post_tweet(client, tweet):
    """Post a tweet using the X v2 API and log the result."""
    try:
        response = client.create_tweet(text=tweet)  # Post via v2 API
        logger.info(f"Tweet posted successfully with ID: {response.data['id']}")
        print(f"\nSuccessfully posted tweet: {tweet} (Tweet ID: {response.data['id']})")
    except tweepy.TweepyException as e:
        logger.error(f"Failed to post tweet: {str(e)}")
        print(f"\nFailed to post tweet: {str(e)}")

def main():
    """Generate and post two tweets with a 2-minute interval without intervention."""
    print("\n Initializing Tweet Generator...")
    generator = TweetGenerator()  # Create generator instance
    
    # Setup Twitter client once for efficiency
    client = tweepy.Client(
        bearer_token=BEARER_TOKEN,  # v2 API authentication
        consumer_key=API_KEY,
        consumer_secret=API_SECRET_KEY,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_TOKEN_SECRET
    )

    # Generate and post first tweet
    print("\n Generating first engaging tweet...")
    tweet1 = generator.generate_engaging_tweet()  # Get first tweet
    if tweet1:
        print("\n FIRST ENGAGING TWEET ")
        print(f"{tweet1}")
        post_tweet(client, tweet1)  # Post it
    else:
        print("Failed to generate first engaging tweet.")
    
    # Wait 2 minutes before second tweet
    print("\nWaiting before posting the second tweet...")
    time.sleep(120)  # 120 seconds = 2 minutes delay
    
    # Generate and post second tweet
    print("\n Generating second engaging tweet...")
    tweet2 = generator.generate_engaging_tweet()  # Get second tweet
    if tweet2:
        print("\n SECOND ENGAGING TWEET ")
        print(f"{tweet2}")
        post_tweet(client, tweet2)  # Post it
    else:
        print("Failed to generate second engaging tweet.")
    
    # Log and confirm completion
    logger.info("Two-tweet generation and posting process completed.")
    print("\nProcess completed.")

if __name__ == "__main__":
    main()  # Entry point to run the script