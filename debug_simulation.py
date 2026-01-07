from src.twitch_utils import get_chat_dataframe
import time

def mock_progress(count):
    print(f"Callback: Downloaded {count} messages...")

def test_simulation():
    url = "https://www.twitch.tv/videos/2663165637"
    print("Starting simulation with URL:", url)
    
    start = time.time()
    # Simulate app params
    df = get_chat_dataframe(url, max_messages=1000, progress_callback=mock_progress)
    
    print(f"Finished. Dataframe shape: {df.shape}")
    print(f"Time taken: {time.time() - start:.2f}s")
    
    if not df.empty:
        print("First few rows:")
        print(df.head())
    else:
        print("Dataframe is empty!")

if __name__ == "__main__":
    test_simulation()
