import requests
from bs4 import BeautifulSoup
from langdetect import detect
import re
import time

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

def scrap_gs(url, output_file="scholar_results.txt"):
    text = ""
    page = 0  

    while True:  
        print(f"Page {page}")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        for item in soup.select('[data-lid]'):
            try:
                title = item.select_one('h3').get_text()
                title = re.sub(r'\[.*?\]\s*', '', title).strip()
                

                # Filter hanya jurnal berbahasa Inggris
                if is_english(title):
                    text += f"{title}\n"

            except AttributeError:
                continue  

        # Cari tombol halaman berikutnya
        next_page = soup.select_one('.gs_ico_nav_next')
        if next_page:
            page += 1
            url = url.split("&start=")[0] + f"&start={page * 10}"
            time.sleep(2)  # Delay agar tidak diblokir oleh Google Scholar
        else:
            break  

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"Data berhasil disimpan ke {output_file}")

def main():
    url = "https://scholar.google.com/scholar?as_q=&as_epq=&as_oq=&as_eq=&as_occt=any&as_sauthors=&as_publication=&as_ylo=2024&as_yhi=2024&hl=id&as_sdt=0%2C5&start=0"
    scrap_gs(url)

main()
