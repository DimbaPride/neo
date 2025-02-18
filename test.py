from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
import logging
import json
from datetime import datetime
import time

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URL do ranking de guild
GUILD_RANKING_URL = "https://www.neogames.online/ranking/guild"

class GuildRankingExtractor:
    def __init__(self):
        """
        Inicializa o extrator de ranking de guilds com os cabeçalhos exatos do site.
        """
        # Cabeçalhos exatos do site
        self.headers = {
            'position': '#',
            'name': 'Nome',
            'power': 'Poder',
            'members': 'Membros',
            'war_points': 'Pontos de Guerra',
            'war_kills': 'Abates na Guerra'
        }

    def fetch_ranking_data(self):
        """
        Busca os dados do ranking de guild usando Playwright.
        """
        logger.info("Iniciando coleta de dados do ranking de guild")
        
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(
                    headless=True,
                    args=['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage']
                )
                
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
                )
                
                page = context.new_page()
                page.set_default_navigation_timeout(60000)
                
                response = page.goto(GUILD_RANKING_URL, wait_until="networkidle")
                
                if not response or response.status != 200:
                    raise Exception(f"Falha ao carregar página: Status {response.status if response else 'N/A'}")
                
                # Espera pela tabela de ranking
                page.wait_for_selector('table', timeout=30000)
                
                # Captura o conteúdo após carregar
                content = page.content()
                
                browser.close()
                return content
                
            except Exception as e:
                logger.error(f"Erro ao buscar ranking de guild: {e}")
                if 'browser' in locals():
                    browser.close()
                raise

    def parse_value(self, value_str):
        """
        Converte valores string para formato apropriado.
        """
        try:
            # Remove caracteres não numéricos exceto pontos e vírgulas
            clean_value = ''.join(c for c in value_str if c.isdigit() or c in '.,')
            # Remove pontos de milhar e troca vírgula por ponto
            clean_value = clean_value.replace('.', '').replace(',', '.')
            return int(float(clean_value)) if clean_value else 0
        except:
            return 0

    def parse_guild_data(self, html_content):
        """
        Analisa o HTML para extrair dados das guilds, ignorando o cabeçalho na contagem.
        """
        logger.info("Analisando dados do ranking de guild")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        guild_data = []
        
        try:
            # Procura a tabela de ranking
            rows = soup.find_all('tr', class_=lambda x: x and 'border-b' in x)
            
            position = 1  # Começa do 1 e só incrementa para linhas válidas
            
            for row in rows:
                # Pula o cabeçalho sem contar na posição
                if row.find('th'):
                    continue
                
                try:
                    # Extrai as células da linha
                    cells = row.find_all(['td'])
                    
                    if len(cells) >= 6:  # Verifica se tem todas as colunas necessárias
                        guild_entry = {
                            'position': position,  # Usa a posição correta
                            'name': cells[1].get_text(strip=True),
                            'power': self.parse_value(cells[2].get_text(strip=True)),
                            'members': self.parse_value(cells[3].get_text(strip=True)),
                            'war_points': self.parse_value(cells[4].get_text(strip=True)),
                            'war_kills': self.parse_value(cells[5].get_text(strip=True))
                        }
                        guild_data.append(guild_entry)
                        position += 1  # Só incrementa quando encontra uma guild válida
                    
                except Exception as e:
                    logger.warning(f"Erro ao processar guild na posição {position}: {e}")
                    continue
            
            return guild_data
            
        except Exception as e:
            logger.error(f"Erro ao analisar ranking de guild: {e}")
            raise

    def save_ranking_data(self, data):
        """
        Salva os dados do ranking de guild em arquivo JSON.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = f"guild_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            output_data = {
                'timestamp': timestamp,
                'total_guilds': len(data),
                'headers': self.headers,
                'rankings': data
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Ranking de guild salvo em: {filename}")
            
            # Mostra exemplo das primeiras guilds
            if data:
                logger.info("\nTop 3 Guilds:")
                for entry in data[:3]:
                    logger.info(
                        f"#{entry['position']}: {entry['name']} - "
                        f"Poder: {entry['power']:,} - "
                        f"Membros: {entry['members']} - "
                        f"Pontos de Guerra: {entry['war_points']:,} - "
                        f"Abates: {entry['war_kills']:,}"
                    )
        
        except Exception as e:
            logger.error(f"Erro ao salvar ranking de guild: {e}")

def main():
    """
    Função principal para extrair o ranking de guilds.
    """
    try:
        extractor = GuildRankingExtractor()
        
        # Tenta coletar dados com retry
        max_retries = 3
        html_content = None
        
        for attempt in range(max_retries):
            try:
                html_content = extractor.fetch_ranking_data()
                if html_content:
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Tentativa {attempt + 1} falhou, tentando novamente...")
                time.sleep(5)
        
        if html_content:
            guild_data = extractor.parse_guild_data(html_content)
            
            if guild_data:
                extractor.save_ranking_data(guild_data)
                logger.info(f"Processamento concluído: {len(guild_data)} guilds encontradas")
            else:
                logger.warning("Nenhum dado de guild encontrado")
        
    except Exception as e:
        logger.error(f"Falha ao processar ranking de guild: {e}")

if __name__ == "__main__":
    main()