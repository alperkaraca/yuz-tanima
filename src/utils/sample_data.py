from __future__ import annotations
import os
from urllib.request import urlretrieve


def download_sample(dest: str = "data/input_videos/sample.mp4"):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # Creative Commons lisanslı bir kısa yüz videosu bulmak ağ gerektirir.
    # Ağ yoksa hata yakalanır; kullanıcı kendi videosunu sağlayabilir.
    url = "https://github.com/opencv/opencv_extra/raw/master/testdata/cv/tracking/faceocc2.webm"
    try:
        urlretrieve(url, dest)
        return dest
    except Exception:
        return None

if __name__ == "__main__":
    path = download_sample()
    print("Downloaded:", path)

