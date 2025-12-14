"""
Web Crawler Module
Crawls websites up to 2 levels deep, extracting content while filtering out
scripts, images, ads, and non-HTML content.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import validators
import time
from typing import Dict, List, Set, Optional
import re


class WebCrawler:
    def __init__(self, max_depth: int = 2, timeout: int = 10, max_pages: int = 50):
        self.max_depth = max_depth
        self.timeout = timeout
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.crawled_data: List[Dict] = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and crawlable"""
        if not url or not validators.url(url):
            return False
        parsed = urlparse(url)
        if parsed.scheme not in ['http', 'https']:
            return False
        excluded_extensions = [
            '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.zip', '.rar',
            '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.css', '.js', '.json', '.xml'
        ]
        path_lower = parsed.path.lower()
        for ext in excluded_extensions:
            if path_lower.endswith(ext):
                return False
        return True
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and trailing slashes"""
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if normalized.endswith('/') and len(parsed.path) > 1:
            normalized = normalized[:-1]
        return normalized
    
    def is_same_domain(self, url: str, base_url: str) -> bool:
        """Check if URL belongs to the same domain as base URL"""
        base_domain = urlparse(base_url).netloc
        url_domain = urlparse(url).netloc
        return url_domain == base_domain or url_domain.endswith('.' + base_domain)
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract relevant content from HTML"""
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 
                         'aside', 'iframe', 'noscript', 'form', 'button',
                         'input', 'select', 'textarea', 'img', 'video',
                         'audio', 'canvas', 'svg', 'object', 'embed']):
            tag.decompose()
        
        for ad_class in ['ad', 'ads', 'advertisement', 'banner', 'popup', 
                         'modal', 'cookie', 'newsletter', 'subscribe']:
            for element in soup.find_all(class_=re.compile(ad_class, re.I)):
                element.decompose()
            for element in soup.find_all(id=re.compile(ad_class, re.I)):
                element.decompose()
        
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        headings = []
        for h_tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            heading_text = h_tag.get_text(strip=True)
            if heading_text:
                headings.append({
                    'level': h_tag.name,
                    'text': heading_text
                })
        
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            paragraphs = main_content.find_all(['p', 'li', 'td', 'th', 'span', 'div'])
            text_content = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and len(text) > 20:
                    text_content.append(text)
            visible_text = '\n'.join(text_content)
        else:
            visible_text = soup.get_text(separator='\n', strip=True)
        
        visible_text = re.sub(r'\n\s*\n', '\n\n', visible_text)
        visible_text = re.sub(r' +', ' ', visible_text)
        
        return {
            'url': url,
            'title': title,
            'headings': headings,
            'content': visible_text
        }
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract valid links from the page"""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            normalized = self.normalize_url(full_url)
            
            if (self.is_valid_url(normalized) and 
                self.is_same_domain(normalized, base_url) and
                normalized not in self.visited_urls):
                links.append(normalized)
        
        return list(set(links))
    
    def crawl_page(self, url: str, base_url: str, depth: int, 
                   progress_callback=None) -> List[str]:
        """Crawl a single page and return found links"""
        if depth > self.max_depth:
            return []
        
        if len(self.visited_urls) >= self.max_pages:
            return []
        
        normalized_url = self.normalize_url(url)
        if normalized_url in self.visited_urls:
            return []
        
        self.visited_urls.add(normalized_url)
        
        try:
            response = requests.get(
                url, 
                headers=self.headers, 
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type.lower():
                return []
            
            soup = BeautifulSoup(response.text, 'lxml')
            page_data = self.extract_content(soup, url)
            
            if page_data['content'] and len(page_data['content']) > 50:
                self.crawled_data.append(page_data)
                if progress_callback:
                    progress_callback(len(self.visited_urls), url, page_data['title'])
            
            if depth < self.max_depth:
                links = self.extract_links(soup, base_url)
                return links
            
        except requests.exceptions.Timeout:
            if progress_callback:
                progress_callback(len(self.visited_urls), url, "Timeout - skipped")
        except requests.exceptions.HTTPError as e:
            if progress_callback:
                progress_callback(len(self.visited_urls), url, f"HTTP Error: {e.response.status_code}")
        except requests.exceptions.RequestException as e:
            if progress_callback:
                progress_callback(len(self.visited_urls), url, f"Error: {str(e)[:50]}")
        except Exception as e:
            if progress_callback:
                progress_callback(len(self.visited_urls), url, f"Parse error: {str(e)[:50]}")
        
        return []
    
    def crawl(self, start_url: str, progress_callback=None) -> List[Dict]:
        """Main crawl method - crawls website up to max_depth levels"""
        self.visited_urls = set()
        self.crawled_data = []
        
        if not self.is_valid_url(start_url):
            raise ValueError(f"Invalid URL: {start_url}")
        
        base_url = start_url
        current_level_urls = [start_url]
        
        for depth in range(self.max_depth + 1):
            if not current_level_urls:
                break
                
            if len(self.visited_urls) >= self.max_pages:
                break
            
            next_level_urls = []
            
            for url in current_level_urls:
                if len(self.visited_urls) >= self.max_pages:
                    break
                
                found_links = self.crawl_page(url, base_url, depth, progress_callback)
                next_level_urls.extend(found_links)
                
                time.sleep(0.3)
            
            current_level_urls = list(set(next_level_urls))
        
        return self.crawled_data
    
    def get_crawl_summary(self) -> Dict:
        """Get summary of crawl results"""
        total_chars = sum(len(page['content']) for page in self.crawled_data)
        return {
            'pages_crawled': len(self.crawled_data),
            'urls_visited': len(self.visited_urls),
            'total_characters': total_chars,
            'pages': [{'url': p['url'], 'title': p['title']} for p in self.crawled_data]
        }
