import requests
from bs4 import BeautifulSoup
import csv
import json
import time
import os
from datetime import datetime

output_dir = "app"
os.makedirs(output_dir, exist_ok=True)
base_url = "https://www.shl.com/solutions/products/product-catalog/"
all_products = []

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

for page in range(32):  
    start = page * 12
    url = f"{base_url}?start={start}&type=1"
    
    print(f"\n[Page {page+1}] Fetching: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.find_all("tr")
        if len(rows) <= 1:
            print("No more rows found. Stopping.")
            break
        page_count = 0
        for row in rows[1:]: 
            try:
                cells = row.find_all(["th", "td"])
                if len(cells) == 0:
                    continue
                first_cell = cells[0]
                link = first_cell.find("a")
                
                if link:
                    name = link.text.strip()
                    url_href = link.get("href", "")
                    
                    if not url_href.startswith("http"):
                        url_href = "https://www.shl.com" + url_href
                    
                    all_spans = row.find_all("span", class_="catalogue__circle")
                    
                    remote = "No"
                    adaptive = "No"
                    
                    if len(all_spans) >= 1:
                        if "yes" in str(all_spans[0].get("class", [])).lower():
                            remote = "Yes"
                    
                    if len(all_spans) >= 2:
                        if "yes" in str(all_spans[1].get("class", [])).lower():
                            adaptive = "Yes"
                    
                    test_spans = row.find_all("span", class_="product-catalogue__key")
                    test_types = []
                    test_keys = []
                    for span in test_spans:
                        key = span.text.strip()
                        if key:
                            test_keys.append(key)
                            mapping = {'K': 'Knowledge & Skills', 'P': 'Personality & Behavior',
                                      'B': 'Behavior', 'A': 'Ability', 'S': 'Simulations'}
                            test_types.append(mapping.get(key, key))
                    
                    product = {
                        "Assessment Name": name,
                        "Assessment URL": url_href,
                        "Remote Testing Support": remote,
                        "Adaptive/IRT Support": adaptive,
                        "Test Type": ", ".join(test_types) if test_types else "N/A",
                        "Test Type Keys": test_keys
                    }
                    
                    all_products.append(product)
                    page_count += 1
                    
                    if page_count <= 2:
                        print(f"  ✓ {name[:50]}")
            
            except Exception as e:
                continue
        
        print(f"✓ Extracted {page_count} products from page {page+1}")
        print(f"Total so far: {len(all_products)}")
        
        if page_count == 0:
            print("No products on this page. Stopping.")
            break
        
        time.sleep(0.5)
    
    except Exception as e:
        print(f"Error: {e}")
        break

print(f"\n{'='*80}")
print(f"TOTAL PRODUCTS: {len(all_products)}")
print(f"{'='*80}")

if all_products:
    json_file = f"app/shl_catalog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_products, f, indent=2)
    print(f"✅ Saved JSON: {json_file}")

    csv_file = f"app/shl_catalog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_products[0].keys())
        writer.writeheader()
        writer.writerows(all_products)
    print(f"✅ Saved CSV: {csv_file}")

    if all_products:
        print(f"\nSample Product:")
        for k, v in all_products[0].items():
            print(f"  {k}: {v}")
