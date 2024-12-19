"""
Twitter API v2 wrapper for internal usage
Implements OAuth 2.0 authentication and basic tweet management functionality
"""

import os
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from current directory or parent directory
load_dotenv()
if not os.getenv('TWITTER_CLIENT_ID'):
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Required OAuth 2.0 credentials
TWITTER_CLIENT_ID = os.getenv('TWITTER_CLIENT_ID')
TWITTER_CLIENT_SECRET = os.getenv('TWITTER_CLIENT_SECRET')
TWITTER_REDIRECT_URI = os.getenv('TWITTER_REDIRECT_URI', 'http://localhost:8080/callback')

class TwitterApiLib:
    """Twitter API v2 wrapper for internal usage"""

    def __init__(self, verbose: bool = False):
        """Initialize Twitter API client with OAuth 2.0 credentials"""
        self.verbose = verbose
        self.client_id = TWITTER_CLIENT_ID
        self.client_secret = TWITTER_CLIENT_SECRET
        self.redirect_uri = TWITTER_REDIRECT_URI

        if not all([self.client_id, self.client_secret]):
            raise ValueError("Missing required Twitter API credentials")

        # Get access token
        self.access_token = self._get_access_token()

    def _get_access_token(self) -> str:
        """
        Get OAuth 2.0 access token using client credentials flow
        Returns the access token as a string
        """
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
        }

        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'client_credentials',
            'scope': 'tweet.write offline.access'
        }

        response = requests.post(
            'https://api.twitter.com/2/oauth2/token',
            headers=headers,
            data=data
        )

        if self.verbose:
            print(f"Token response: {response.text}")

        if response.status_code != 200:
            raise Exception(f"Failed to get access token: {response.text}")

        return response.json()['access_token']

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
        url = 'https://api.twitter.com/2/tweets'
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        data = {'text': text}

        response = requests.post(url, headers=headers, json=data)
        if self.verbose:
            print(f"Create tweet response: {response.text}")

        if response.status_code != 201:
            raise Exception(f"{response.status_code} {response.reason} for url: {url}")

        return response.json()

    def delete_tweet(self, tweet_id: str) -> Dict[str, Any]:
        """
        Delete a tweet
        Args:
            tweet_id: The ID of the tweet to delete
        Returns:
            Dict containing the API response
        """
        url = f'https://api.twitter.com/2/tweets/{tweet_id}'
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

        response = requests.delete(url, headers=headers)
        if self.verbose:
            print(f"Delete tweet response: {response.text}")

        if response.status_code != 200:
            raise Exception(f"{response.status_code} {response.reason} for url: {url}")

        return response.json()

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
