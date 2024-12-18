"""
Example script demonstrating Twitter API usage with delayed tweet deletion
Shows how to post a tweet, wait for a specified duration, then delete it
"""

import time
from twitter_api import TwitterApiLib

def main():
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
        deleted = twitter.delete_tweet(tweet_id)
        print(f"Tweet deleted successfully: {deleted}")

    except Exception as e:
        print(f"Error in example script: {e}")

if __name__ == "__main__":
    main()
