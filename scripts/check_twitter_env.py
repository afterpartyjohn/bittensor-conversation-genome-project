"""
Diagnostic script to verify Twitter API environment variables
"""
import os
from twitter_api import TwitterApiLib

def check_env_variables():
    """Check if all required Twitter API environment variables are set"""
    print('Checking Twitter API environment variables...')
    
    variables = [
        'TWITTER_API_KEY',
        'TWITTER_API_SECRET',
        'TWITTER_ACCESS_TOKEN',
        'TWITTER_ACCESS_TOKEN_SECRET'
    ]
    
    all_present = True
    for var in variables:
        value = os.getenv(var)
        exists = bool(value)
        print(f'{var} exists: {exists}')
        if exists:
            # Only show first/last few characters if value exists
            masked_value = f'{value[:4]}...{value[-4:]}' if value else 'None'
            print(f'{var} value: {masked_value}')
        all_present = all_present and exists
    
    print(f'\nAll required variables present: {all_present}')

if __name__ == '__main__':
    check_env_variables()
