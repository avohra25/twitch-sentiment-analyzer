from chat_downloader import ChatDownloader
import sys
import time

def test_download():
    # A known specific VOD might be hard to guess, but let's try a very popular channel's recent vod if possible, 
    # or just rely on the library to handle an invalid url gracefully (it should raise error).
    # Since I don't have a URL, I'll use a placeholder and expect the user might need to edit it, 
    # OR I can rely on the fact that I can't really test a real URL without one.
    # actually, I'll try to use a public VOD URL if I can find one. 
    # Let's search for "Twitch VOD" to get a valid URL, or just use the user's URL if I knew it.
    
    # I'll try a generic reliable test URL if I can, but for now let's just use a dummy one 
    # and see if it fails fast or hangs.
    url = "https://www.twitch.tv/videos/2143026857" # Random VOD ID, might be dead.
    
    print(f"Testing download from {url}...")
    downloader = ChatDownloader()
    
    try:
        # We'll set a timeout if possible? No, library doesn't support explicit timeout arg easily.
        # But we can try to get iterator.
        start = time.time()
        chat = downloader.get_chat(url)
        print("Got iterator in", time.time() - start, "seconds")
        
        count = 0
        for msg in chat:
            print(f"Got message: {msg['message']}")
            count += 1
            if count >= 5:
                print("Successfully downloaded 5 messages.")
                break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_download()
