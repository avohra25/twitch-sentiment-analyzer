from chat_downloader import ChatDownloader
import pandas as pd
import datetime

def get_chat_dataframe(url, max_messages=None, progress_callback=None):
    """
    Downloads chat from a Twitch VOD or Clip URL and returns a Pandas DataFrame.
    Apps can provide a progress_callback(current_count) to update UI.
    """
    downloader = ChatDownloader()
    chats = []
    
    try:
        # Get chat generator
        chat_iterator = downloader.get_chat(url)
        
        count = 0
        for message in chat_iterator:
            
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
                break
                
    except Exception as e:
        print(f"Error downloading chat: {e}")
        return pd.DataFrame() # Return empty on error
        
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
