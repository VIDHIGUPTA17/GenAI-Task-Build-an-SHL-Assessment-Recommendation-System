# """
# SHL Assessment Details Scraper - Enriches JSON with description, time, job levels, etc.
# Processes each assessment URL and extracts detailed information
# """

# import requests
# from bs4 import BeautifulSoup
# import json
# import time
# import os
# from datetime import datetime
# import logging
# from pathlib import Path

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class SHLAssessmentDetailsScraper:
#     """Scrapes detailed information from each SHL assessment page"""
    
#     def __init__(self, json_input_path, output_dir="app"):
#         self.json_input_path = json_input_path
#         self.output_dir = output_dir
#         self.headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
#         }
#         os.makedirs(output_dir, exist_ok=True)
        
#         self.assessments = []
#         self._load_json()
    
#     def _load_json(self):
#         """Load the initial JSON catalog"""
#         if not os.path.exists(self.json_input_path):
#             raise FileNotFoundError(f"JSON file not found: {self.json_input_path}")
        
#         try:
#             with open(self.json_input_path, 'r', encoding='utf-8') as f:
#                 self.assessments = json.load(f)
#             logger.info(f"✓ Loaded {len(self.assessments)} assessments from JSON")
#         except Exception as e:
#             logger.error(f"Error loading JSON: {e}")
#             raise
    
#     def extract_details_from_page(self, url, assessment_index):
#         """
#         Extract detailed information from an assessment page
#         Returns dict with: Description, Time, Job Levels, Languages, etc.
#         """
#         details = {
#             "Description": None,
#             "Time (Minutes)": None,
#             "Job Levels": None,
#             "Languages": None,
#             "Assessment Type": None,
#             "Use Cases": None,
#             "Sample Tasks": None,
#             "Job Titles": None,
#             "Configurations": None,
#             "Features": None
#         }
        
#         try:
#             logger.info(f"[{assessment_index}] Fetching: {url[:80]}...")
#             response = requests.get(url, headers=self.headers, timeout=15)
#             response.raise_for_status()
            
#             soup = BeautifulSoup(response.text, 'html.parser')
            
#             # METHOD 1: Extract main description
#             description_section = None
            
#             # Try to find introduction/description section
#             intro_div = soup.find("div", class_="product-intro")
#             if intro_div:
#                 description_section = intro_div
            
#             if not description_section:
#                 intro_div = soup.find("section", class_="introduction")
#                 if intro_div:
#                     description_section = intro_div
            
#             if not description_section:
#                 intro_div = soup.find("div", class_="content-block")
#                 if intro_div:
#                     description_section = intro_div
            
#             # Extract description text
#             if description_section:
#                 # Get all paragraph text
#                 paragraphs = description_section.find_all("p")
#                 if paragraphs:
#                     desc_text = " ".join([p.get_text().strip() for p in paragraphs[:2]])
#                     details["Description"] = desc_text[:500]  # First 500 chars
#                 else:
#                     details["Description"] = description_section.get_text().strip()[:500]
            
#             # METHOD 2: Extract completion time
#             time_patterns = [
#                 ("Assessment length", "Approximate Completion Time in minutes"),
#                 ("Time", "minutes"),
#                 ("Duration", "minutes"),
#                 ("Completion Time", "minutes")
#             ]
            
#             page_text = soup.get_text()
            
#             # Look for time information
#             for pattern in time_patterns:
#                 if pattern[0] in page_text:
#                     # Find the line with time
#                     lines = page_text.split('\n')
#                     for line in lines:
#                         if pattern[0] in line and ("minutes" in line or "minute" in line):
#                             # Extract numbers
#                             import re
#                             numbers = re.findall(r'\d+', line)
#                             if numbers:
#                                 details["Time (Minutes)"] = numbers[0]
#                                 break
            
#             # METHOD 3: Extract job levels
#             job_levels = []
#             job_level_keywords = [
#                 "Entry-level", "Junior", "Mid-level", "Mid-Professional",
#                 "Senior", "Professional", "Manager", "Executive", "Leadership"
#             ]
            
#             for keyword in job_level_keywords:
#                 if keyword in page_text:
#                     job_levels.append(keyword)
            
#             if job_levels:
#                 details["Job Levels"] = ", ".join(set(job_levels))
            
#             # METHOD 4: Extract languages
#             languages = []
#             language_keywords = [
#                 "English (USA)", "English (UK)", "Spanish", "French", "German",
#                 "Portuguese", "Dutch", "Chinese", "Japanese", "Multilingual"
#             ]
            
#             for keyword in language_keywords:
#                 if keyword in page_text:
#                     languages.append(keyword)
            
#             if languages:
#                 details["Languages"] = ", ".join(set(languages))
            
#             # METHOD 5: Extract assessment type
#             assessment_types = [
#                 "ability", "personality", "behavior", "knowledge", "skills",
#                 "cognitive", "competency", "situational", "integrity"
#             ]
            
#             found_types = []
#             for atype in assessment_types:
#                 if atype.lower() in page_text.lower():
#                     found_types.append(atype.capitalize())
            
#             if found_types:
#                 details["Assessment Type"] = ", ".join(set(found_types))
            
#             # METHOD 6: Extract job titles
#             job_titles = []
#             lines = page_text.split('\n')
#             for line in lines:
#                 if "job titles" in line.lower() or "positions" in line.lower():
#                     # Get the next line which might have titles
#                     idx = lines.index(line)
#                     if idx + 1 < len(lines):
#                         job_titles_line = lines[idx + 1]
#                         # Extract comma-separated titles
#                         titles = [t.strip() for t in job_titles_line.split(',') if len(t.strip()) > 2]
#                         job_titles.extend(titles[:5])  # Limit to 5
            
#             if job_titles:
#                 details["Job Titles"] = ", ".join(set(job_titles))
            
#             # METHOD 7: Extract sample tasks
#             sample_tasks = []
#             lines = page_text.split('\n')
#             for i, line in enumerate(lines):
#                 if "sample tasks" in line.lower() or "examples" in line.lower():
#                     # Get next few lines
#                     for j in range(i+1, min(i+4, len(lines))):
#                         task_text = lines[j].strip()
#                         if len(task_text) > 10 and task_text not in sample_tasks:
#                             sample_tasks.append(task_text[:100])
            
#             if sample_tasks:
#                 details["Sample Tasks"] = "; ".join(sample_tasks[:3])
            
#             # METHOD 8: Extract use cases
#             use_cases = []
#             if "use case" in page_text.lower() or "use for" in page_text.lower():
#                 lines = page_text.split('\n')
#                 for line in lines:
#                     if ("use" in line.lower() and len(line) > 20) or "ideal for" in line.lower():
#                         use_cases.append(line.strip()[:100])
            
#             if use_cases:
#                 details["Use Cases"] = "; ".join(use_cases[:2])
            
#             # METHOD 9: Extract configurations
#             if "configurations" in page_text.lower():
#                 details["Configurations"] = "Multiple configurations available"
            
#             # METHOD 10: Extract key features from page sections
#             features = []
#             sections = soup.find_all(["section", "div"], class_=["feature", "benefit", "highlight"])
#             for section in sections[:3]:
#                 feature_text = section.get_text().strip()[:100]
#                 if feature_text:
#                     features.append(feature_text)
            
#             if features:
#                 details["Features"] = "; ".join(features)
            
#             logger.info(f"  ✓ Extracted details successfully")
#             return details
        
#         except requests.exceptions.Timeout:
#             logger.warning(f"  ⚠️ Timeout fetching {url}")
#             return details
#         except requests.exceptions.ConnectionError:
#             logger.warning(f"  ⚠️ Connection error for {url}")
#             return details
#         except Exception as e:
#             logger.warning(f"  ⚠️ Error extracting details: {e}")
#             return details
    
#     def enrich_assessments(self, delay=1):
#         """
#         Process each assessment URL and add detailed information
#         """
#         logger.info("="*80)
#         logger.info(f"ENRICHING {len(self.assessments)} ASSESSMENTS WITH DETAILS")
#         logger.info("="*80)
        
#         enriched_assessments = []
        
#         for idx, assessment in enumerate(self.assessments, 1):
#             try:
#                 url = assessment.get("Assessment URL")
                
#                 if not url:
#                     logger.warning(f"[{idx}] No URL found, skipping")
#                     enriched_assessments.append(assessment)
#                     continue
                
#                 # Extract details from the page
#                 details = self.extract_details_from_page(url, idx)
                
#                 # Merge details with original assessment
#                 enriched = {**assessment, **details}
#                 enriched_assessments.append(enriched)
                
#                 # Show progress every 10 items
#                 if idx % 10 == 0:
#                     logger.info(f"✓ Processed {idx}/{len(self.assessments)} assessments")
                
#                 # Rate limiting
#                 time.sleep(delay)
            
#             except Exception as e:
#                 logger.error(f"[{idx}] Error processing assessment: {e}")
#                 enriched_assessments.append(assessment)
        
#         logger.info(f"\n✓ Enrichment complete!")
#         return enriched_assessments
    
#     def save_enriched_json(self, enriched_assessments, filename=None):
#         """Save enriched data to JSON"""
#         if not filename:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = os.path.join(self.output_dir, f"shl_catalog_enriched_{timestamp}.json")
        
#         try:
#             with open(filename, "w", encoding='utf-8') as f:
#                 json.dump(enriched_assessments, f, indent=2, ensure_ascii=False)
            
#             logger.info(f"\n✅ Enriched JSON saved: {filename}")
#             logger.info(f"   Total assessments: {len(enriched_assessments)}")
#             logger.info(f"   File size: {os.path.getsize(filename) / 1024:.2f} KB")
#             return filename
        
#         except Exception as e:
#             logger.error(f"Error saving JSON: {e}")
#             return None
    
#     def save_enriched_csv(self, enriched_assessments, filename=None):
#         """Save enriched data to CSV"""
#         if not filename:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = os.path.join(self.output_dir, f"shl_catalog_enriched_{timestamp}.csv")
        
#         try:
#             import csv
            
#             if not enriched_assessments:
#                 logger.warning("No data to save")
#                 return None
            
#             fieldnames = list(enriched_assessments[0].keys())
            
#             with open(filename, "w", newline='', encoding='utf-8') as f:
#                 writer = csv.DictWriter(f, fieldnames=fieldnames)
#                 writer.writeheader()
#                 writer.writerows(enriched_assessments)
            
#             logger.info(f"✅ Enriched CSV saved: {filename}")
#             logger.info(f"   File size: {os.path.getsize(filename) / 1024:.2f} KB")
#             return filename
        
#         except Exception as e:
#             logger.error(f"Error saving CSV: {e}")
#             return None
    
#     def print_sample(self, enriched_assessments):
#         """Print sample enriched assessment"""
#         if enriched_assessments:
#             logger.info("\n" + "="*80)
#             logger.info("SAMPLE ENRICHED ASSESSMENT")
#             logger.info("="*80)
            
#             sample = enriched_assessments[0]
#             for key, value in sample.items():
#                 if value:
#                     value_preview = str(value)[:100] if isinstance(value, str) else str(value)
#                     logger.info(f"{key}: {value_preview}")
    
#     def run(self):
#         """Run complete enrichment process"""
#         # Enrich assessments
#         enriched = self.enrich_assessments(delay=1)
        
#         # Save results
#         json_file = self.save_enriched_json(enriched)
#         csv_file = self.save_enriched_csv(enriched)
        
#         # Show sample
#         self.print_sample(enriched)
        
#         logger.info("\n" + "="*80)
#         logger.info("ENRICHMENT COMPLETE")
#         logger.info("="*80)
#         logger.info(f"JSON Output: {json_file}")
#         logger.info(f"CSV Output: {csv_file}")
        
#         return json_file, csv_file


# def main():
#     """Main execution"""
    
#     # Find the latest JSON file
#     app_dir = Path("app")
#     json_files = sorted(app_dir.glob("shl_catalog_*.json"))
    
#     if not json_files:
#         logger.error("❌ No JSON catalog found in app/ directory")
#         logger.error("Run shl_scraper_working.py first to generate catalog")
#         return
    
#     latest_json = str(json_files[-1])
#     logger.info(f"Using catalog: {latest_json}")
    
#     # Create scraper and run
#     scraper = SHLAssessmentDetailsScraper(latest_json)
#     scraper.run()


# if __name__ == "__main__":
#     main()






















# """
# SHL Assessment Details Scraper - Enriches JSON with description, time, job levels, etc.
# Processes each assessment URL and extracts detailed information
# WITH FILE-BASED CACHING
# """

# import requests
# from bs4 import BeautifulSoup
# import json
# import time
# import os
# import hashlib
# from datetime import datetime
# import logging
# from pathlib import Path

# # -------------------------------------------------------------------
# # Logging Configuration
# # -------------------------------------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# class SHLAssessmentDetailsScraper:
#     """Scrapes detailed information from each SHL assessment page with caching"""

#     def __init__(self, json_input_path, output_dir="app"):
#         self.json_input_path = json_input_path
#         self.output_dir = output_dir

#         self.headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
#         }

#         # Output & cache directories
#         os.makedirs(self.output_dir, exist_ok=True)
#         self.cache_dir = os.path.join(self.output_dir, "cache_pages")
#         os.makedirs(self.cache_dir, exist_ok=True)

#         self.assessments = []
#         self._load_json()

#     # -------------------------------------------------------------------
#     # JSON Loader
#     # -------------------------------------------------------------------
#     def _load_json(self):
#         if not os.path.exists(self.json_input_path):
#             raise FileNotFoundError(f"JSON file not found: {self.json_input_path}")

#         with open(self.json_input_path, "r", encoding="utf-8") as f:
#             self.assessments = json.load(f)

#         logger.info(f"✓ Loaded {len(self.assessments)} assessments from JSON")

#     # -------------------------------------------------------------------
#     # Cache Helpers
#     # -------------------------------------------------------------------
#     def _cache_path(self, url: str) -> str:
#         key = hashlib.sha256(url.encode("utf-8")).hexdigest()
#         return os.path.join(self.cache_dir, f"{key}.json")

#     def _load_from_cache(self, url: str):
#         path = self._cache_path(url)
#         if os.path.exists(path):
#             try:
#                 with open(path, "r", encoding="utf-8") as f:
#                     return json.load(f)
#             except Exception:
#                 return None
#         return None

#     def _save_to_cache(self, url: str, html: str):
#         path = self._cache_path(url)
#         try:
#             with open(path, "w", encoding="utf-8") as f:
#                 json.dump(
#                     {
#                         "url": url,
#                         "html": html,
#                         "cached_at": datetime.utcnow().isoformat()
#                     },
#                     f,
#                     ensure_ascii=False
#                 )
#         except Exception:
#             pass

#     # -------------------------------------------------------------------
#     # Page Scraper
#     # -------------------------------------------------------------------
#     def extract_details_from_page(self, url, assessment_index):
#         details = {
#             "Description": None,
#         "Time (Minutes)": None,
#         "Job Levels": None,
#         "Languages": None,
#         "Assessment Type": None,
#         "Use Cases": None,
#         "Sample Tasks": None,
#         "Job Titles": None,
#         "Configurations": None,
#         "Features": None,
#     }

#         try:
#             logger.info(f"[{assessment_index}] Fetching: {url[:80]}")

#             cached = self._load_from_cache(url)
#             if cached:
#                 html = cached["html"]
#                 logger.info("  ↺ Loaded from cache")
#             else:
#                 resp = requests.get(url, headers=self.headers, timeout=15)
#                 resp.raise_for_status()
#                 html = resp.text
#                 self._save_to_cache(url, html)
#                 logger.info("  💾 Saved to cache")

#             soup = BeautifulSoup(html, "html.parser")
#             page_text = soup.get_text(separator="\n")

#         # --------- helper: get <p> after given <h4> label ----------
#             def get_p_after_h4(label_text: str):
#                 h4 = soup.find("h4", string=lambda s: s and label_text.lower() in s.lower())
#                 if not h4:
#                     return None
#                 p = h4.find_next("p")
#                 if not p:
#                     return None
#                 return p.get_text(strip=True)

#         # Description
#             desc = get_p_after_h4("Description")
#             if desc:
#                 details["Description"] = desc[:500]

#         # Job Levels
#             job_levels_text = get_p_after_h4("Job levels")
#             if job_levels_text:
#                 details["Job Levels"] = job_levels_text

#         # Languages
#             langs_text = get_p_after_h4("Languages")
#             if langs_text:
#                 details["Languages"] = langs_text

#         # Time (Minutes)
#             import re

#             time_text = get_p_after_h4("Assessment length")
#             if time_text:
#                 nums = re.findall(r"\d+", time_text)
#                 if nums:
#                     details["Time (Minutes)"] = nums[0]
#             else:
#             # fallback to generic search if block not found
#                 for line in page_text.split("\n"):
#                     if any(k in line.lower() for k in ["assessment length", "time", "duration", "completion"]):
#                         nums = re.findall(r"\d+", line)
#                         if nums:
#                             details["Time (Minutes)"] = nums[0]
#                             break

#         # Assessment Type from "Test Type" line near the small text
#         # e.g. "Test Type ... Knowledge & Skills"
#             test_type_block = soup.find("p", class_="product-cataloguesmall-text")
#             if test_type_block and "Test Type" in test_type_block.get_text():
#             # Try to get following span containing keys (K, P, etc.) if needed,
#             # but you already have Test Type from your catalog JSON.
#                 details["Assessment Type"] = None  # or leave as is / map from Test Type

#         # You can keep your old generic heuristics for extra fields as fallback
#         # or remove them if they are too noisy.

#             logger.info("  ✓ Extracted successfully")
#             return details

#         except Exception as e:
#             logger.warning(f"  ⚠️ Error extracting details: {e}")
#         return details


#     # -------------------------------------------------------------------
#     # Enrichment Runner
#     # -------------------------------------------------------------------
#     def enrich_assessments(self, delay=1):
#         enriched = []

#         for idx, assessment in enumerate(self.assessments, 1):
#             url = assessment.get("Assessment URL")
#             if not url:
#                 enriched.append(assessment)
#                 continue

#             details = self.extract_details_from_page(url, idx)
#             enriched.append({**assessment, **details})
#             time.sleep(delay)

#         return enriched

#     # -------------------------------------------------------------------
#     # Save Methods
#     # -------------------------------------------------------------------
#     def save_enriched_json(self, data):
#         filename = os.path.join(
#             self.output_dir,
#             f"shl_catalog_enriched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#         )
#         with open(filename, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)

#         logger.info(f"✅ JSON saved: {filename}")
#         return filename

#     def save_enriched_csv(self, data):
#         import csv
#         filename = os.path.join(
#             self.output_dir,
#             f"shl_catalog_enriched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#         )
#         with open(filename, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=data[0].keys())
#             writer.writeheader()
#             writer.writerows(data)

#         logger.info(f"✅ CSV saved: {filename}")
#         return filename

#     # -------------------------------------------------------------------
#     # Run
#     # -------------------------------------------------------------------
#     def run(self):
#         enriched = self.enrich_assessments()
#         self.save_enriched_json(enriched)
#         self.save_enriched_csv(enriched)
#         logger.info("🎉 ENRICHMENT COMPLETE")


# # -------------------------------------------------------------------
# # Main
# # -------------------------------------------------------------------
# def main():
#     app_dir = Path("app")
#     json_files = sorted(app_dir.glob("shl_catalog_*.json"))

#     if not json_files:
#         logger.error("❌ No JSON catalog found")
#         return

#     scraper = SHLAssessmentDetailsScraper(str(json_files[-1]))
#     scraper.run()


# if __name__ == "__main__":
#     main()





import requests
from bs4 import BeautifulSoup
import json
import time
import os
import hashlib
from datetime import datetime
import logging
from pathlib import Path

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SHLAssessmentDetailsScraper:
    """Scrapes detailed information from each SHL assessment page with caching"""

    def __init__(self, json_input_path, output_dir="app", max_assessments=None):
        self.json_input_path = json_input_path
        self.output_dir = output_dir
        self.max_assessments = max_assessments  # Limit how many assessments to process (for testing)

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        # Output & cache directories
        os.makedirs(self.output_dir, exist_ok=True)
        self.cache_dir = os.path.join(self.output_dir, "cache_pages")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.assessments = []
        self._load_json()

    # -------------------------------------------------------------------
    # JSON Loader
    # -------------------------------------------------------------------
    def _load_json(self):
        if not os.path.exists(self.json_input_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_input_path}")

        with open(self.json_input_path, "r", encoding="utf-8") as f:
            self.assessments = json.load(f)

        logger.info(f"✓ Loaded {len(self.assessments)} assessments from JSON")

        if self.max_assessments is not None:
            self.assessments = self.assessments[:self.max_assessments]
            logger.info(f"  ➡️ Limited processing to first {self.max_assessments} assessments (for testing)")

    # -------------------------------------------------------------------
    # Cache Helpers
    # -------------------------------------------------------------------
    def _cache_path(self, url: str) -> str:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.json")

    def _load_from_cache(self, url: str):
        path = self._cache_path(url)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, url: str, html: str):
        path = self._cache_path(url)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "url": url,
                        "html": html,
                        "cached_at": datetime.utcnow().isoformat()
                    },
                    f,
                    ensure_ascii=False
                )
        except Exception:
            pass

    # -------------------------------------------------------------------
    # Page Scraper
    # -------------------------------------------------------------------
    def extract_details_from_page(self, url, assessment_index):
        details = {
            "Description": None,
            "Time (Minutes)": None,
            "Job Levels": None,
            "Languages": None,
        }

        try:
            logger.info(f"[{assessment_index}] Fetching: {url[:80]}")

            cached = self._load_from_cache(url)
            if cached:
                html = cached["html"]
                logger.info("  ↺ Loaded from cache")
            else:
                resp = requests.get(url, headers=self.headers, timeout=15)
                resp.raise_for_status()
                html = resp.text
                self._save_to_cache(url, html)
                logger.info("  💾 Saved to cache")

            soup = BeautifulSoup(html, "html.parser")
            page_text = soup.get_text(separator="\n")

            # --------- helper: get <p> after given <h4> label ----------
            def get_p_after_h4(label_text: str):
                h4 = soup.find("h4", string=lambda s: s and label_text.lower() in s.lower())
                if not h4:
                    return None
                p = h4.find_next("p")
                if not p:
                    return None
                return p.get_text(strip=True)

            # Description
            desc = get_p_after_h4("Description")
            if desc:
                details["Description"] = desc[:500]

            # Job Levels
            job_levels_text = get_p_after_h4("Job levels")
            if job_levels_text:
                details["Job Levels"] = job_levels_text

            # Languages
            langs_text = get_p_after_h4("Languages")
            if langs_text:
                details["Languages"] = langs_text

            # Time (Minutes)
            import re
            time_text = get_p_after_h4("Assessment length")
            if time_text:
                nums = re.findall(r"\d+", time_text)
                if nums:
                    details["Time (Minutes)"] = nums[0]
            else:
                # fallback to generic search
                for line in page_text.split("\n"):
                    if any(k in line.lower() for k in ["assessment length", "time", "duration", "completion"]):
                        nums = re.findall(r"\d+", line)
                        if nums:
                            details["Time (Minutes)"] = nums[0]
                            break

            logger.info("  ✓ Extracted successfully")
            return details

        except Exception as e:
            logger.warning(f"  ⚠️ Error extracting details: {e}")
            return details

    # -------------------------------------------------------------------
    # Enrichment Runner
    # -------------------------------------------------------------------
    def enrich_assessments(self, delay=1):
        enriched = []

        for idx, assessment in enumerate(self.assessments, 1):
            url = assessment.get("Assessment URL")
            if not url:
                enriched.append(assessment)
                continue

            details = self.extract_details_from_page(url, idx)
            enriched.append({**assessment, **details})
            time.sleep(delay)

        return enriched

    # -------------------------------------------------------------------
    # Save Methods
    # -------------------------------------------------------------------
    def save_enriched_json(self, data):
        filename = os.path.join(
            self.output_dir,
            f"shl_catalog_enriched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ JSON saved: {filename}")
        return filename

    def save_enriched_csv(self, data):
        import csv
        filename = os.path.join(
            self.output_dir,
            f"shl_catalog_enriched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        logger.info(f"✅ CSV saved: {filename}")
        return filename

    # -------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------
    def run(self):
        enriched = self.enrich_assessments()
        self.save_enriched_json(enriched)
        self.save_enriched_csv(enriched)
        logger.info("🎉 ENRICHMENT COMPLETE")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    app_dir = Path("app")
    json_files = sorted(app_dir.glob("shl_catalog_20251218_001752.json"))

    if not json_files:
        logger.error("❌ No JSON catalog found")
        return

    # Process ONLY the first 9 assessments for testing
    scraper = SHLAssessmentDetailsScraper(str(json_files[-1]), max_assessments=None)
    scraper.run()


if __name__ == "__main__":
    main()

