import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime

URL = "https://store.steampowered.com/app/2507950/Delta_Force/?cc=us&l=english"

HEADERS = {
    'Accept-Language': 'en-US,en;q=0.5',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
}
COOKIES = {
    'birthtime': '631152001', 
    'wants_mature_content': '1',
    'Steam_Language': 'english'
}

print("--- Step 1: Downloading main page data (forcing English) ---")
page = requests.get(URL, headers=HEADERS, cookies=COOKIES)
soup = BeautifulSoup(page.content, "html.parser")

game_name = 'N/A'
price_usd = None
is_free = False
release_date_str = 'N/A'
days_since_release = None
tags = []
controller_support = False
steam_deck_support = False
languages = []
percentage_str, review_count, positive_reviews = 'N/A', 0, 0
all_time_peak_players = None
avg_players_last_30_days = None

# 1. Title
name_div = soup.find('div', {'id': 'appHubAppName'})
game_name = name_div.text.strip() if name_div else 'N/A'

# 2. Price
base_game_purchase_area = soup.find('div', {'class': 'game_area_purchase_game'})
if base_game_purchase_area:
    price_div = base_game_purchase_area.find('div', {'class': 'game_purchase_price'})
    if price_div and ('Free To Play' in price_div.text or 'Free' == price_div.text.strip()):
        is_free = True
        price_usd = 0.0
    if not is_free:
        discount_price_div = base_game_purchase_area.find('div', {'class': 'discount_final_price'})
        price_to_parse = discount_price_div.text if discount_price_div else (base_game_purchase_area.find('div', {'class': 'game_purchase_price'}).text if base_game_purchase_area.find('div', {'class': 'game_purchase_price'}) else None)
        if price_to_parse:
            price_match = re.search(r'(\d+\.\d{2})', price_to_parse)
            if price_match: price_usd = float(price_match.group(1))

# 3. Release date and days from release
date_div = soup.find('div', {'class': 'date'})
if date_div:
    release_date_str = date_div.text.strip()
    try:
        release_date_obj = datetime.strptime(release_date_str, '%b %d, %Y')
        days_since_release = (datetime.now() - release_date_obj).days
    except ValueError:
        days_since_release = -1

# 4. Tags
tags_container = soup.find('div', {'class': 'glance_tags popular_tags'})
tags = [tag.text.strip() for tag in tags_container.find_all('a', {'class': 'app_tag'})] if tags_container else []

# 5. Controller Support
controller_support = False
controller_container = soup.find('div', attrs={'data-featuretarget': 'store-sidebar-controller-support-info'})
if controller_container and controller_container.has_attr('data-props'):
    try:
        data_props_str = controller_container['data-props']
        controller_data = json.loads(data_props_str)
        if controller_data.get('bFullXboxControllerSupport') is True:
            controller_support = True
    except json.JSONDecodeError:
        pass

# 6. Steam Deck Compatibility
deck_block = soup.find('div', attrs={'data-featuretarget': 'deck-verified-results'})
if deck_block:
    steam_deck_support = True
    
# 7. Languages
lang_table = soup.find('table', {'class': 'game_language_options'})
if lang_table:
    lang_set = set()
    supported_langs = lang_table.find_all('td', class_='ellipsis')
    for lang_cell in supported_langs:
        if lang_cell.find_next_sibling('td') and '✔' in lang_cell.find_next_sibling('td').text:
            lang_set.add(lang_cell.text.strip())
    languages = list(lang_set)

app_id_match = re.search(r'/app/(\d+)/', URL)
app_id = app_id_match.group(1) if app_id_match else None

# 8. Reviews
if app_id:
    print("--- Step 2: Downloading review data from API ---")
    reviews_url = f"https://store.steampowered.com/appreviews/{app_id}?json=1"
    reviews_response = requests.get(reviews_url)
    if reviews_response.status_code == 200:
        reviews_data = reviews_response.json()
        summary = reviews_data.get('query_summary', {})
        percentage_str = summary.get('review_score_desc', 'N/A')
        review_count = summary.get('total_reviews', 0)
        positive_reviews = summary.get('total_positive', 0)

# 9. Player statistics
if app_id:
    print("--- Step 3: Downloading player stats from Steam Charts ---")
    steamcharts_url = f"https://steamcharts.com/app/{app_id}"
    charts_page = requests.get(steamcharts_url, headers=HEADERS)
    if charts_page.status_code == 200:
        charts_soup = BeautifulSoup(charts_page.content, "html.parser")
        
        peak_stat_div = charts_soup.select_one('div.app-stat:nth-of-type(3)')
        if peak_stat_div:
             num_span = peak_stat_div.find('span', class_='num')
             if num_span: all_time_peak_players = int(num_span.text.replace(',', ''))
        
        if days_since_release is not None and days_since_release >= 30:
            avg_30_days_stat = charts_soup.select_one('div.app-stat:nth-of-type(2) span.num')
            if avg_30_days_stat:
                avg_players_last_30_days = float(avg_30_days_stat.text.strip().replace(',', ''))

print("\n--- FINAL RESULTS ---")
print(f"Name: {game_name}")
print(f"Price (USD): {price_usd}")
print(f"Is Free-to-Play: {is_free}")
print(f"Release Date: {release_date_str}")
print(f"Days Since Release: {days_since_release}")
print(f"Tags: {tags}")
print(f"Controller Support: {controller_support}")
print(f"Steam Deck Support: {steam_deck_support}")
print(f"Supported Languages: {languages}")
print(f"Review Summary: {percentage_str}")
print(f"Total Reviews: {review_count}")
print(f"Positive Reviews: {positive_reviews}")
print(f"All-Time Peak Players (Steam only): {all_time_peak_players}")
print(f"Average Players (Last 30 Days): {avg_players_last_30_days}")