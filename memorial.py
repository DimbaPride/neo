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

# URL do ranking memorial
MEMORIAL_RANKING_URL = "https://www.neogames.online/ranking/memorial"

# Mapeamento de classes do Cabal
CLASS_MAPPING = {
    'icon-wa': 'Warrior',
    'icon-bl': 'Blader',
    'icon-wz': 'Wizard',
    'icon-fs': 'Force Shielder',
    'icon-fb': 'Force Blader',
    'icon-gl': 'Gladiator',
    'icon-fg': 'Force Gunner',
    'icon-fa': 'Force Archer',
    'icon-dm': 'Dark Mage',
    'icon-aa': 'Archon Archer'
}

# Mapeamento de nações
NATION_MAPPING = {
    'icon-procyon': 'Procyon',
    'icon-capella': 'Capella'
}

class MemorialRankingExtractor:
    def __init__(self):
        self.headers = {
            'position': '#',
            'character_name': 'Nome do Personagem',
            'character_class': 'Classe',
            'guild_name': 'Guild',
            'nation': 'Nação'
        }

    def fetch_ranking_data(self):
        """
        Busca os dados do ranking memorial usando Playwright com espera explícita.
        """
        logger.info("Iniciando coleta de dados do ranking memorial")
        
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
                
                # Navega para a página
                logger.info("Navegando para a página do ranking")
                response = page.goto(MEMORIAL_RANKING_URL)
                
                if not response:
                    raise Exception("Falha ao carregar a página - sem resposta")
                
                # Espera pelo container principal
                logger.info("Aguardando carregamento do container principal")
                page.wait_for_selector('.grid.grid-cols-1', timeout=30000)
                
                # Espera adicional para garantir que todos os cards sejam carregados
                logger.info("Aguardando carregamento dos cards")
                time.sleep(5)
                
                # Avalia a quantidade de cards presentes
                card_count = page.evaluate('''() => {
                    return document.querySelectorAll('.rounded-md.border-2').length;
                }''')
                
                logger.info(f"Encontrados {card_count} cards no total")
                
                if card_count == 0:
                    raise Exception("Nenhum card encontrado na página")
                
                # Extrai o conteúdo da página
                content = page.content()
                
                browser.close()
                return content
                
            except Exception as e:
                logger.error(f"Erro ao buscar ranking memorial: {e}")
                if 'browser' in locals():
                    browser.close()
                raise

    def parse_memorial_data(self, html_content):
        """
        Analisa o HTML para extrair dados dos possuidores da espada memorial.
        """
        logger.info("Analisando dados do ranking memorial")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        memorial_data = []
        
        try:
            # Encontra todos os cards usando a estrutura correta
            cards = soup.select('div.rounded-md.border-2.text-card-foreground')
            
            logger.info(f"Encontrados {len(cards)} cards para processar")
            
            for position, card in enumerate(cards, 1):
                try:
                    # Extrai o nome do personagem
                    character_name_elem = card.select_one('h2.font-bold')
                    character_name = character_name_elem.get_text(strip=True) if character_name_elem else "Unknown"
                    
                    # Extrai o nome da guild
                    guild_name_elem = card.select_one('p.text-muted-foreground')
                    guild_name = guild_name_elem.get_text(strip=True) if guild_name_elem else "No Guild"
                    
                    # Extrai a classe do personagem
                    class_icon = card.select_one('img[alt^="Icon"]')
                    class_src = class_icon['srcset'].split(' ')[0] if class_icon and 'srcset' in class_icon.attrs else ''
                    
                    character_class = 'Unknown'
                    for class_key, class_name in CLASS_MAPPING.items():
                        if class_key in class_src:
                            character_class = class_name
                            break
                    
                    # Extrai a nação
                    nation_icon = card.select_one('img[alt="Icon Nation"]')
                    nation_src = nation_icon['srcset'].split(' ')[0] if nation_icon and 'srcset' in nation_icon.attrs else ''
                    
                    nation = 'Unknown'
                    for nation_key, nation_name in NATION_MAPPING.items():
                        if nation_key in nation_src:
                            nation = nation_name
                            break
                    
                    memorial_entry = {
                        'position': position,
                        'character_name': character_name,
                        'character_class': character_class,
                        'guild_name': guild_name,
                        'nation': nation
                    }
                    
                    memorial_data.append(memorial_entry)
                    logger.info(f"Processado: {character_name} ({character_class}) - {guild_name}")
                    
                except Exception as e:
                    logger.warning(f"Erro ao processar card {position}: {e}")
                    continue
            
            if len(memorial_data) < 9:
                logger.warning(f"Atenção: Encontrados apenas {len(memorial_data)} possuidores de 9 esperados")
            
            return memorial_data
            
        except Exception as e:
            logger.error(f"Erro ao analisar ranking memorial: {e}")
            raise

    def save_ranking_data(self, data):
        """
        Salva os dados do ranking memorial em arquivo JSON.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = f"memorial_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            output_data = {
                'timestamp': timestamp,
                'total_holders': len(data),
                'headers': self.headers,
                'rankings': data
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Ranking memorial salvo em: {filename}")
            
            # Mostra os possuidores encontrados
            if data:
                logger.info("\nPossuidores da Espada Memorial:")
                for entry in data:
                    logger.info(
                        f"#{entry['position']}: {entry['character_name']} "
                        f"({entry['character_class']}) - "
                        f"Guild: {entry['guild_name']} - "
                        f"Nação: {entry['nation']}"
                    )
            
        except Exception as e:
            logger.error(f"Erro ao salvar ranking memorial: {e}")

def main():
    """
    Função principal para extrair o ranking memorial.
    """
    try:
        extractor = MemorialRankingExtractor()
        
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
            memorial_data = extractor.parse_memorial_data(html_content)
            
            if memorial_data:
                extractor.save_ranking_data(memorial_data)
                logger.info(f"Processamento concluído: {len(memorial_data)} possuidores encontrados")
            else:
                logger.warning("Nenhum possuidor encontrado")
        
    except Exception as e:
        logger.error(f"Falha ao processar ranking memorial: {e}")

if __name__ == "__main__":
    main()