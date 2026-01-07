from chat_downloader import ChatDownloader
import pandas as pd
import datetime

def get_chat_dataframe(url, max_messages=None, progress_callback=None):
    """
    Downloads chat from a Twitch VOD or Clip URL and returns a Pandas DataFrame.
    Apps can provide a progress_callback(current_count) to update UI.
    """
    import sys
    print("DEBUG: Calling get_chat_dataframe...", file=sys.__stdout__)
    
    # Initialize with quiet=True to prevent stdout interference
    downloader = ChatDownloader(quiet=True)
    chats = []
    
    try:
        print(f"DEBUG: Attempting to fetch chat from {url}", file=sys.__stdout__)
        # Get chat generator
        chat_iterator = downloader.get_chat(url)
        print("DEBUG: Got chat iterator!", file=sys.__stdout__)
        
        count = 0
        for message in chat_iterator:
            if count == 0:
                print("DEBUG: First message received!", file=sys.__stdout__)
                
            # Basic info
            timestamp = message.get('time_in_seconds', 0) # Time offset in VOD
            author = message.get('author', {}).get('name', 'Anonymous')
            text = message.get('message', '')
            
            # Store
            chats.append({
                'timestamp': timestamp,
                'author': author,
                'message': text,
                'time_text': str(datetime.timedelta(seconds=int(timestamp)))
            })
            
            count += 1
            if progress_callback and count % 50 == 0:
                progress_callback(count)

            if max_messages and count >= max_messages:
                print("DEBUG: Max messages reached.", file=sys.__stdout__)
                break
                
    except Exception as e:
        print(f"Error downloading chat: {e}", file=sys.__stdout__)
        return pd.DataFrame() # Return empty on error
        
    print(f"DEBUG: Finished downloading {len(chats)} messages.", file=sys.__stdout__)
    return pd.DataFrame(chats)

def parse_chat_log_file(file_obj):
    """
    Parses a raw text file or JSON file uploaded by the user.
    """
    # Simple JSON parser if it matches chat-downloader format
    try:
        df = pd.read_json(file_obj)
        # Normalize columns if needed
        required_cols = ['timestamp', 'author', 'message']
        if not all(col in df.columns for col in required_cols):
             # Fallback for simple CSV/Text? 
             pass
        return df
    except:
        return pd.DataFrame()

def generate_mock_chat_dataframe(num_messages=100):
    """
    Generates fake chat data for debugging UI/Model without network.
    """
    import random
    
    mock_msgs = [
        "POG", "LUL", "This is amazing!", "WTF happened?", "Kappa", 
        "ResidentSleeper", "So boring", "Hype!", "Love this streamer", "Terrible play"
    ]
    
    chats = []
    for i in range(num_messages):
        chats.append({
            'timestamp': i * 2,
            'author': f"User{i}",
            'message': random.choice(mock_msgs),
            'time_text': str(datetime.timedelta(seconds=i*2))
        })
    return pd.DataFrame(chats)
