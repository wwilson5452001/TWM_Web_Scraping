import os
import cv2
import numpy as np
import requests
import urllib.parse
import time
import random
import torch
import base64
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from facenet_pytorch import MTCNN, InceptionResnetV1
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160,
    margin=14,
    device=device,
)

mtcnn_all = MTCNN(
    image_size=160,
    margin=14,
    device=device,
    keep_all=True,
)

resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

def check_copyright_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 檢查meta標籤
    meta_tags = soup.find_all('meta')
    for tag in meta_tags:
        if 'copyright' in tag.get('name', '').lower():
            return tag.get('content', 'No copyright information found')

    # 檢查頁腳
    footer = soup.find('footer')
    if footer:
        text = footer.get_text()
        if 'copyright' in text.lower():
            return text

    return 'No copyright information found'

def scrape_google_images(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

        image_urls = []
        num_scrolls = 3
        for _ in range(num_scrolls):
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            image_tags = soup.find_all('img')

            for img in image_tags:
                src = img.get('data-src') or img.get('src')
                if src:
                    if 'http' in src or src.startswith('data:image'):
                        if src not in image_urls:
                            image_urls.append(src)

            driver.execute_script("window.scrollBy(0, 1000);")
            WebDriverWait(driver, 2).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

        driver.quit()
        return image_urls

    except Exception as e:
        print(f"An error occurred: {e}")
        driver.quit()

def distance(embeddings1, embeddings2):
    dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
    norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
    similarity = dot / norm
    dist = np.arccos(similarity) / math.pi
    return dist[0]

def download_images(image_urls, save_folder, prefix):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    this = True
    for i, url in enumerate(image_urls):
        try:
            # 檢查圖片的版權信息
            # copyright_info = check_copyright_info(url)
            # if 'No copyright information found' not in copyright_info:
            #     print(f"==================================================Image {i + 1} skipped - copyright issue detected: {copyright_info}")
            #     continue

            if url.startswith('data:image'):
                # 處理base64編碼的圖片
                header, encoded = url.split(',', 1)
                data = base64.b64decode(encoded)
                image = np.frombuffer(data, np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            else:
                # 處理普通圖片URL
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image_data = np.array(bytearray(response.content), dtype=np.uint8)
                    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                else:
                    print(f"Failed to retrieve image from {url}")
                    continue

            if image is not None:
                height, width, _ = image.shape
                if width >= 160 and height >= 160:
                    faces_all = mtcnn_all(image)
                    if len(faces_all) == 1:
                        with torch.no_grad():
                            face_crop = mtcnn(image)
                            face_crop = face_crop.to(device, dtype=torch.float32)
                            face_crop = face_crop.unsqueeze(0)
                            face_emb = resnet(face_crop)
                            face_emb = face_emb.cpu().numpy()

                        if this:
                            pre_image_emb = face_emb
                            this = False
                        else:
                            dist = distance(pre_image_emb, face_emb)
                            if dist >= 0.4 or dist < 0.01:
                                raise ValueError(f"similarity too large: {dist}")

                        # pre_image_emb = face_emb
                        file_extension = 'jpg' if not url.startswith('data:image') else header.split('/')[1].split(';')[0]
                        file_path = os.path.join(save_folder, f"{prefix}_image_{i + 1}.{file_extension}")
                        if url.startswith('data:image'):
                            with open(file_path, 'wb') as file:
                                file.write(data)
                        else:
                            with open(file_path, 'wb') as file:
                                file.write(response.content)
                        print(f"Image {i + 1} saved to {file_path}")
                    else:
                        print(f"Image {i + 1} skipped - found {len(faces_all)} faces.")
                else:
                    print(f"Image {i + 1} skipped - resolution is {width}x{height}.")
            else:
                print(f"Failed to decode image {i + 1} from {url}")
        except Exception as e:
            print(f"An error occurred while downloading {url}: {e}")

if __name__ == "__main__":
    with open("cleaned_name_list2.txt", "r") as file:
        names = [line.strip() for line in file]
    # names = ['A. K. Antony (India)']

    total = len(names)
    start = 0
    names = names[start:]

    google_path = "https://www.google.com/search?tbm=isch&q={}"

    for i, name in enumerate(names):
        save_folder = os.path.join('google_images', f"{name}")
        if os.path.exists(save_folder) or os.path.exists('google_images_no_check/'+name):
            continue
        encoded_name = urllib.parse.quote(name)

        # Search for young images
        url_young = google_path.format(encoded_name + " young")
        data_young = scrape_google_images(url_young)
        
        if data_young:
            download_images(data_young, save_folder, "young")
            print(f'{i+start+1}/{total} {name} young done')
        else:
            print(f'{i+start+1}/{total} {name} no young data')

        sleep_time = random.uniform(20, 60)
        print(f"Sleeping for {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)

        # Search for old images
        url_old = google_path.format(encoded_name)
        data_old = scrape_google_images(url_old)
        
        if data_old:
            download_images(data_old, save_folder, "old")
            print(f'{i+start+1}/{total} {name} old done')
        else:
            print(f'{i+start+1}/{total} {name} no old data')
        sleep_time = random.uniform(20, 60)  
        print(f"Sleeping for {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)
