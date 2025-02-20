import requests
import xml.etree.ElementTree as ET
from datetime import datetime

class NeoGamesKnowledge:
    def __init__(self, base_url: str = "https://www.neogames.online", sitemap_url: str = "https://www.neogames.online/sitemap.xml"):
        self.base_url = base_url
        self.sitemap_url = sitemap_url

    def fetch_sitemap(self):
        """Busca e processa o sitemap para obter URLs e datas de modificação"""
        try:
            response = requests.get(self.sitemap_url)
            response.raise_for_status()
            
            # Processa o conteúdo XML do sitemap
            tree = ET.fromstring(response.content)
            ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            
            # Variável para armazenar as URLs com suas datas
            urls_and_dates = []

            # Itera pelas entradas do sitemap
            for url_elem in tree.findall("ns:url", ns):
                try:
                    loc = url_elem.find("ns:loc", ns).text.strip()  # URL da página
                    lastmod_elem = url_elem.find("ns:lastmod", ns)  # Data de modificação
                    
                    # Tenta obter a data de modificação
                    lastmod = None
                    if lastmod_elem is not None and lastmod_elem.text:
                        try:
                            # Converte para datetime
                            lastmod = datetime.fromisoformat(lastmod_elem.text.replace('Z', '+00:00'))
                        except ValueError:
                            print(f"Erro no formato da data para {loc}")
                    
                    # Se a data não for encontrada, usa a data atual
                    if not lastmod:
                        lastmod = datetime.now()
                    
                    # Adiciona a URL e a data ao resultado
                    urls_and_dates.append((loc, lastmod))
                    
                except Exception as e:
                    print(f"Erro ao processar entrada do sitemap: {e}")
                    continue
            
            # Retorna as URLs com as datas de modificação
            return urls_and_dates
        
        except Exception as e:
            print(f"Erro ao buscar sitemap: {e}")
            return []

# Exemplo de uso
neo_games = NeoGamesKnowledge()

# Buscando as URLs e as datas de modificação
urls_and_dates = neo_games.fetch_sitemap()

# Exibindo as URLs e suas datas de modificação
for url, date in urls_and_dates:
    print(f"URL: {url}, Data de Modificação: {date}")
