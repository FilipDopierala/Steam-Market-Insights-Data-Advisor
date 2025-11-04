import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import csv
import time

HEADERS = {
    'Accept-Language': 'en-US,en;q=0.5',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
}
COOKIES = {
    'birthtime': '631152001',
    'wants_mature_content': '1',
    'Steam_Language': 'english',
}

def get_top_app_ids(limit=50):
    print(f"--- Step 1: Fetching top {limit} game App IDs from SteamCharts ---")
    app_ids = []
    url = "https://steamcharts.com/top"
    
    try:
        page = requests.get(url, headers=HEADERS)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, "html.parser")
        
        game_rows = soup.select("#top-games tbody tr")[:limit]
        
        for row in game_rows:
            link_tag = row.find('a', href=True)
            if link_tag:
                app_id_match = re.search(r'/app/(\d+)', link_tag['href'])
                if app_id_match:
                    app_ids.append(app_id_match.group(1))
                    
    except requests.RequestException as e:
        print(f"Error fetching SteamCharts page: {e}")
        return []
    
    print(f"Successfully found {len(app_ids)} App IDs.")
    return app_ids

def scrape_game_data(app_id):

    steam_store_url = f"https://store.steampowered.com/app/{app_id}/?cc=us&l=english"
    steamcharts_url = f"https://steamcharts.com/app/{app_id}"
    
    game_data = {
        'App ID': app_id, 'Name': 'N/A', 'Price (USD)': None, 'Is Free': False,
        'Release Date': 'N/A', 'Days Since Release': None, 'Tags': [],
        'Controller Support': False, 'Steam Deck Support': False, 'Languages': [],
        'Review Summary': 'N/A', 'Total Reviews': 0, 'Positive Reviews': 0,
        'All-Time Peak Players': None, 'Avg Players (30 Days)': None
    }
    
    try:
        page = requests.get(steam_store_url, headers=HEADERS, cookies=COOKIES)
        if "agecheck" in page.url:
             print(f"Failed: App ID {app_id} is behind an age gate that could not be bypassed. Skipping.")
             return None
        page.raise_for_status()
        soup = BeautifulSoup(page.content, "html.parser")

        name_div = soup.find('div', {'id': 'appHubAppName'})
        game_data['Name'] = name_div.text.strip() if name_div else 'N/A'
        
        if game_data['Name'] == 'N/A':
            print(f"Failed: Could not find game title for App ID {app_id}. Page might be invalid. Skipping.")
            return None

        purchase_area = soup.find('div', {'class': 'game_area_purchase_game'})
        if purchase_area:
            price_div = purchase_area.find('div', {'class': 'game_purchase_price'})
            if price_div and ('Free' in price_div.text):
                game_data['Is Free'] = True
                game_data['Price (USD)'] = 0.0
            else:
                final_price_div = purchase_area.find('div', {'class': 'discount_final_price'})
                price_str = final_price_div.text if final_price_div else (price_div.text if price_div else "")
                price_match = re.search(r'(\d+[.,]\d{2})', price_str)
                if price_match:
                    game_data['Price (USD)'] = float(price_match.group(1).replace(',', '.'))

        date_div = soup.find('div', {'class': 'date'})
        if date_div:
            game_data['Release Date'] = date_div.text.strip()
            try:
                if ',' in game_data['Release Date']:
                    release_obj = datetime.strptime(game_data['Release Date'], '%b %d, %Y')
                else:
                    release_obj = datetime.strptime(game_data['Release Date'], '%b %Y')
                game_data['Days Since Release'] = (datetime.now() - release_obj).days
            except ValueError:
                pass

        tags_container = soup.find('div', {'class': 'glance_tags popular_tags'})
        if tags_container:
            game_data['Tags'] = ', '.join([tag.text.strip() for tag in tags_container.find_all('a')])

        controller_container = soup.find('div', attrs={'data-featuretarget': 'store-sidebar-controller-support-info'})
        if controller_container and controller_container.has_attr('data-props'):
            try:
                data_props_str = controller_container['data-props']
                controller_data = json.loads(data_props_str)
                if controller_data.get('bFullXboxControllerSupport') is True:
                    game_data['Controller Support'] = True
            except json.JSONDecodeError:
                pass

        deck_block = soup.find('div', attrs={'data-featuretarget': 'deck-verified-results'})
        if deck_block:
            game_data['Steam Deck Support'] = True
        lang_table = soup.find('table', {'class': 'game_language_options'})
        if lang_table:
            langs = [lang.text.strip() for lang in lang_table.select('td.ellipsis')]
            game_data['Languages'] = ', '.join(list(set(langs)))

        reviews_url = f"https://store.steampowered.com/appreviews/{app_id}?json=1"
        reviews_response = requests.get(reviews_url)
        if reviews_response.status_code == 200:
            summary = reviews_response.json().get('query_summary', {})
            game_data['Review Summary'] = summary.get('review_score_desc', 'N/A')
            game_data['Total Reviews'] = summary.get('total_reviews', 0)
            game_data['Positive Reviews'] = summary.get('total_positive', 0)

        charts_page = requests.get(steamcharts_url, headers=HEADERS)
        if charts_page.status_code == 200:
            charts_soup = BeautifulSoup(charts_page.content, "html.parser")
            stats = charts_soup.select('div.app-stat .num')
            if len(stats) >= 3:
                try:
                    game_data['Avg Players (30 Days)'] = float(stats[1].text.replace(',', ''))
                    game_data['All-Time Peak Players'] = int(stats[2].text.replace(',', ''))
                except (ValueError, IndexError):
                    pass
                
        return game_data

    except requests.RequestException as e:
        print(f"Error scraping App ID {app_id}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for App ID {app_id}: {e}")
        return None

if __name__ == "__main__":
    top_app_ids = get_top_app_ids(limit=50)
    
    if not top_app_ids:
        print("Could not retrieve App IDs. Exiting.")
    else:
        all_games_data = []
        
        for i, app_id in enumerate(top_app_ids):
            print(f"\n--- Scraping game {i+1}/{len(top_app_ids)} (App ID: {app_id}) ---")
            data = scrape_game_data(app_id)
            if data:
                all_games_data.append(data)
                print(f"Successfully scraped: {data.get('Name', 'N/A')}")
            else:
                print(f"Skipping App ID: {app_id} due to previous errors.")
            
            time.sleep(1.2)

        if all_games_data:
            output_filename = 'steam_top_50_games.csv'
            print(f"\n--- Step 3: Saving {len(all_games_data)} games to {output_filename} ---")
            
            headers = all_games_data[0].keys()
            
            with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(all_games_data)
            
            print("Data scraping and saving complete.")
        else:
            print("No data was collected. CSV file not created.")