import requests

BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

def send_telegram_alert(message, image_bytes=None):
    if not BOT_TOKEN or "YOUR_TELEGRAM" in BOT_TOKEN:
        print(f"[ALERT LOG] {message}")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto" if image_bytes else f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    try:
        if image_bytes:
            files = {'photo': ('alert.jpg', image_bytes, 'image/jpeg')}
            data = {'chat_id': CHAT_ID, 'caption': message, 'parse_mode': 'HTML'}
            requests.post(url, data=data, files=files, timeout=5)
        else:
            data = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
            requests.post(url, data=data, timeout=5)
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")