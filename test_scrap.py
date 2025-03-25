import requests  
from bs4 import BeautifulSoup  
from langdetect import detect  
import re  
import time  
import json  

def is_english(text):  
    """Check if the text is in English."""  
    try:  
        return detect(text) == "en"  
    except:  
        return False  

def scrape_google_scholar(url, output_file="scholar_results.json"):  
    """Scrape Google Scholar to get articles' titles, authors, cited by, and versions."""  
    page = 0  
    data_list = []  # List to store the collected data  

    while True:  
        print(f"Scraping Page {page}")  
        response = requests.get(url)  
        soup = BeautifulSoup(response.content, 'html.parser')  
        
        # Find all the results on the page  
        results = soup.find_all('div', class_='gs_ri')  
        
        for result in results:  
            try:  
                # Title  
                title = result.find('h3', class_='gs_rt').get_text()  
                title = re.sub(r'\[.*?\]\s*', '', title).strip()  

                # Check if the title is in English  
                if not is_english(title):  
                    continue  

                # Authors  
                authors = result.find('div', class_='gs_a').text  
                authors = re.sub(r'\s*-\s*.*$', '', authors)  
                authors = re.sub(r'\d{4}.*$', '', authors)  
                authors = authors.strip()  

                # Cited by  
                cited_by_elem = result.find('div', class_='gs_fl').find_all('a')  
                cited_by = cited_by_elem[2].text if len(cited_by_elem) > 2 else 'Cited by 0'  
                cited_by = re.search(r'\d+', cited_by).group()  

                # Versions  
                versions_elem = result.find('div', class_='gs_fl').find_all('a')  
                versions = versions_elem[4].text if len(versions_elem) > 4 else '0 versions'  
                versions_match = re.search(r'\d+', versions)  
                versions = versions_match.group() if versions_match else '0'  
                
                # Collect data  
                data = {  
                    "Judul": title,  
                    "Penulis": authors,  
                    "Jumlah Dirujuk": cited_by,  
                    "Jumlah Versi": versions  
                }  
                data_list.append(data)  

            except AttributeError as e:  
                print(f"AttributeError: {e}")  
                continue  
        
        # Next Page  
        next_page = soup.select_one('.gs_ico_nav_next')  
        if next_page:  
            page += 1  
            url = url.split("&start=")[0] + f"&start={page * 10}"  
            time.sleep(5)  # Avoid getting blocked by using delay  
        else:  
            break  

    # Save to JSON  
    with open(output_file, 'w', encoding='utf-8') as json_file:  
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)  

    print(f"Data successfully saved to {output_file}")  

def main():  
    url = "https://scholar.google.com/scholar?as_q=&as_epq=&as_oq=&as_eq=&as_occt=any&as_sauthors=&as_publication=&as_ylo=2024&as_yhi=2024&hl=id&as_sdt=0%2C5&start=0"  
    scrape_google_scholar(url)  

main()  