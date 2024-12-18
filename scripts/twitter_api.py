"""
Twitter API v2 wrapper for internal usage
Provides basic tweet management functionality using Twitter API v2
"""

import os
import json
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TwitterApiLib:
    """Twitter API v2 wrapper for internal usage"""

    def __init__(self, verbose: bool = False):
        """Initialize Twitter API client with credentials from environment"""
        self.verbose = verbose
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

        if not all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            raise ValueError("Missing required Twitter API credentials in environment")

        self.base_url = "https://api.twitter.com/2"

    def _get_auth_header(self) -> Dict[str, str]:
        """Generate OAuth 1.0a authorization header"""
        return {
            "Authorization": f"OAuth oauth_consumer_key=\"{self.api_key}\", "
                           f"oauth_token=\"{self.access_token}\"",
            "Content-Type": "application/json"
        }

    def create_tweet(self, text: str) -> Dict[str, Any]:
        """
        Create a new tweet

        Args:
            text: The text content of the tweet

        Returns:
            Dict containing the response from Twitter API

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        url = f"{self.base_url}/tweets"

        try:
            response = requests.post(
                url,
                headers=self._get_auth_header(),
                json={"text": text},
                timeout=30
            )

            if self.verbose:
                print(f"Create tweet response: {response.text}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error creating tweet: {e}")
            raise

    def delete_tweet(self, tweet_id: str) -> bool:
        """
        Delete a tweet by ID

        Args:
            tweet_id: The ID of the tweet to delete

        Returns:
            bool: True if deletion was successful, False otherwise

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        url = f"{self.base_url}/tweets/{tweet_id}"

        try:
            response = requests.delete(
                url,
                headers=self._get_auth_header(),
                timeout=30
            )

            if self.verbose:
                print(f"Delete tweet response: {response.text}")

            response.raise_for_status()
            return True

        except requests.exceptions.RequestException as e:
            print(f"Error deleting tweet: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    try:
        twitter = TwitterApiLib(verbose=True)

        # Create a tweet
        result = twitter.create_tweet("Hello from TwitterApiLib!")
        tweet_id = result["data"]["id"]
        print(f"Created tweet with ID: {tweet_id}")

        # Delete the tweet
        deleted = twitter.delete_tweet(tweet_id)
        print(f"Tweet deleted: {deleted}")

    except Exception as e:
        print(f"Error in example: {e}")
