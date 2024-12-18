"""
Example script demonstrating Twitter API usage with delayed tweet deletion
Shows how to post a tweet, wait for a specified duration, then delete it
"""

import os
import sys
import time
from twitter_api import TwitterApiLib

def main():
    # Check for required environment variables
    if not all([os.getenv('TWITTER_CLIENT_ID'), os.getenv('TWITTER_CLIENT_SECRET')]):
        print("Error: Missing required Twitter API credentials")
        print("Please set TWITTER_CLIENT_ID and TWITTER_CLIENT_SECRET in .env")
        sys.exit(1)

    # Initialize Twitter API client with verbose logging
    twitter = TwitterApiLib(verbose=True)

    try:
        # Post a tweet
        print("Creating tweet...")
        result = twitter.create_tweet("Hello! This is a test tweet that will be deleted in 5 minutes. #test")
        tweet_id = result["data"]["id"]
        print(f"Created tweet with ID: {tweet_id}")

        # Wait for 5 minutes
        wait_time = 300  # 5 minutes in seconds
        print(f"Waiting for {wait_time} seconds before deletion...")
        time.sleep(wait_time)

        # Delete the tweet
        print("Deleting tweet...")
        response = twitter.delete_tweet(tweet_id)
        print(f"Tweet deletion response: {response}")

    except Exception as e:
        print(f"Error in example script: {e}")

if __name__ == "__main__":
    main()
