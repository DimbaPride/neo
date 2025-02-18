#power.py
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

# URLs de ranking
BASE_URL = "https://www.neogames.online/ranking"
GUILD_RANKING_URL = f"{BASE_URL}/guild"
MEMORIAL_RANKING_URL = f"{BASE_URL}/memorial"
POWER_RANKING_URL = f"{BASE_URL}/power"

# Mapeamento atualizado de classes com nomes em português e inglês
# Mapeamento atualizado de classes usando os alt das imagens
CLASS_MAPPING = {
    1: {
        'name': 'Warrior',
        'name_pt': 'Guerreiro',
        'short': 'GU',
        'icon': 'wa',
        'alt': 'Icon WA'
    },
    2: {
        'name': 'Blader',
        'name_pt': 'Duelista',
        'short': 'DU',
        'icon': 'bl',
        'alt': 'Icon BL'
    },
    3: {
        'name': 'Wizard',
        'name_pt': 'Mago',
        'short': 'MA',
        'icon': 'wz',
        'alt': 'Icon WZ'
    },
    4: {
        'name': 'Force Archer',
        'name_pt': 'Arqueiro Arcano',
        'short': 'AA',
        'icon': 'fa',
        'alt': 'Icon FA'
    },
    5: {
        'name': 'Force Shielder',
        'name_pt': 'Guardião Arcano',
        'short': 'GA',
        'icon': 'fs',
        'alt': 'Icon FS'
    },
    6: {
        'name': 'Force Blader',
        'name_pt': 'Espadachim Arcano',
        'short': 'EA',
        'icon': 'fb',
        'alt': 'Icon FB'
    },
    7: {
        'name': 'Gladiator',
        'name_pt': 'Gladiador',
        'short': 'GL',
        'icon': 'gl',
        'alt': 'Icon GL'
    },
    8: {
        'name': 'Force Gunner',
        'name_pt': 'Atirador',
        'short': 'AT',
        'icon': 'fg',
        'alt': 'Icon FG'
    },
    9: {
        'name': 'Dark Mage',
        'name_pt': 'Mago Negro',
        'short': 'MN',
        'icon': 'dm',
        'alt': 'Icon DM'
    }
}

def get_class_info(self, class_text_or_icon):
    """
    Identifica a classe baseado no texto ou ícone.
    Agora inclui verificação do atributo alt da imagem.
    """
    if not class_text_or_icon:
        return None
        
    class_text_lower = class_text_or_icon.lower()
    
    # Primeiro tenta encontrar pelo ID da classe no texto
    for class_id, info in CLASS_MAPPING.items():
        if str(class_id) in class_text_or_icon:
            return info
            
    # Depois tenta pelos outros identificadores
    for _, info in CLASS_MAPPING.items():
        if any(identifier.lower() in class_text_lower for identifier in [
            info['name'].lower(),
            info['name_pt'].lower(),
            info['short'].lower(),
            info['icon'].lower(),
            info['alt'].lower()
        ]):
            return info
            
    # Se não encontrou, retorna classe desconhecida
    return {
        'name': 'Unknown',
        'name_pt': 'Desconhecida',
        'short': 'UNK',
        'icon': 'unknown'
    }

# Mapeamento atualizado de nações
NATION_MAPPING = {
    'icon-procyon': {
        'name': 'Procyon',
        'name_pt': 'Procion',
        'icon_alt': 'Logo nation',
        'icon_src': 'icon-procyon.png'
    },
    'icon-capella': {
        'name': 'Capella',
        'name_pt': 'Capella',
        'icon_alt': 'Logo nation',
        'icon_src': 'icon-capella.png'
    }
}
class RankingExtractor:
    def __init__(self):
        """
        Inicializa o extrator de rankings.
        """
        self.guild_headers = {
            'position': '#',
            'name': 'Nome',
            'power': 'Poder',
            'members': 'Membros',
            'war_points': 'Pontos de Guerra',
            'war_kills': 'Abates na Guerra'
        }
        
        self.memorial_headers = {
            'position': '#',
            'character_name': 'Nome do Personagem',
            'character_class': 'Classe',
            'guild_name': 'Guild',
            'nation': 'Nação'
        }
        
        self.power_headers = {
            'position': '#',
            'class': 'Classe',
            'name': 'Nome',
            'guild': 'Guilda',
            'attack_power': 'Poder de Ataque',
            'defense_power': 'Poder de Defesa',
            'total_power': 'Poder total',
            'nation': 'Nação'
        }

    def fetch_page_content(self, url, wait_selector='table', timeout=30000):
        """
        Busca o conteúdo de uma página usando Playwright.
        """
        logger.info(f"Acessando URL: {url}")
        
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
                
                response = page.goto(url)
                
                if not response or response.status != 200:
                    raise Exception(f"Falha ao carregar página: Status {response.status if response else 'N/A'}")
                
                # Espera pelo seletor específico
                page.wait_for_selector(wait_selector, timeout=timeout)
                
                # Espera adicional para carregamento dinâmico
                time.sleep(3)
                
                content = page.content()
                browser.close()
                return content
                
            except Exception as e:
                logger.error(f"Erro ao buscar página {url}: {e}")
                if 'browser' in locals():
                    browser.close()
                raise

        def get_nation_info(self, cell_or_src):
            """
            Identifica a nação baseado na célula da tabela ou src da imagem.
            """
            try:
                # Se for uma célula BeautifulSoup
                if hasattr(cell_or_src, 'find'):
                    img = cell_or_src.find('img')
                    if img and 'srcset' in img.attrs:
                        srcset = img['srcset']
                        
                        # Procura por procyon ou capella no srcset
                        if 'icon-procyon.png' in srcset:
                            return NATION_MAPPING['icon-procyon']
                        elif 'icon-capella.png' in srcset:
                            return NATION_MAPPING['icon-capella']
                
                # Se for uma string (src ou texto)
                elif isinstance(cell_or_src, str):
                    text_lower = cell_or_src.lower()
                    if 'procyon' in text_lower:
                        return NATION_MAPPING['icon-procyon']
                    elif 'capella' in text_lower:
                        return NATION_MAPPING['icon-capella']
                    
            except Exception as e:
                logger.warning(f"Erro ao identificar nação: {e}")
            
            # Se não encontrou, retorna nação desconhecida
            return {
                'name': 'Unknown',
                'name_pt': 'Desconhecida',
                'icon_alt': 'Unknown'
            }        

    def parse_value(self, value_str):
        """
        Converte valores string para formato numérico.
        """
        try:
            clean_value = ''.join(c for c in value_str if c.isdigit() or c in '.,')
            clean_value = clean_value.replace('.', '').replace(',', '.')
            return int(float(clean_value))
        except:
            return 0

    def get_class_info(self, class_name_or_icon):
        """
        Identifica a classe baseado no nome ou ícone.   
        """
        class_name_lower = class_name_or_icon.lower()
        
        for class_id, info in CLASS_MAPPING.items():
            if (info['name'].lower() in class_name_lower or 
                info['name_pt'].lower() in class_name_lower or 
                info['short'].lower() in class_name_lower or 
                info['icon'].lower() in class_name_lower):
                return info
        return None

    def get_nation_info(self, nation_src_or_name):
        """
        Identifica a nação baseado no src da imagem ou nome.
        """
        nation_lower = nation_src_or_name.lower()
        
        for icon, info in NATION_MAPPING.items():
            if (icon.lower() in nation_lower or 
                info['name'].lower() in nation_lower or 
                info['name_pt'].lower() in nation_lower):
                return info
        return None

    def parse_guild_ranking(self, html_content):
        """
        Analisa o HTML para extrair dados do ranking de guild.
        """
        logger.info("Analisando dados do ranking de guild")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        guild_data = []
        
        try:
            rows = soup.find_all('tr')[1:]  # Pula o cabeçalho
            
            for position, row in enumerate(rows, 1):
                try:
                    cells = row.find_all(['td'])
                    if len(cells) >= 6:
                        guild_entry = {
                            'position': position,
                            'name': cells[1].get_text(strip=True),
                            'power': self.parse_value(cells[2].get_text(strip=True)),
                            'members': self.parse_value(cells[3].get_text(strip=True)),
                            'war_points': self.parse_value(cells[4].get_text(strip=True)),
                            'war_kills': self.parse_value(cells[5].get_text(strip=True))
                        }
                        guild_data.append(guild_entry)
                except Exception as e:
                    logger.warning(f"Erro ao processar guild {position}: {e}")
                    continue
            
            return guild_data
            
        except Exception as e:
            logger.error(f"Erro ao analisar ranking de guild: {e}")
            raise

    def parse_memorial_ranking(self, html_content):
        """
        Analisa o HTML para extrair dados do ranking memorial.
        """
        logger.info("Analisando dados do ranking memorial")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        memorial_data = []
        
        try:
            cards = soup.select('div.rounded-md.border-2.text-card-foreground')
            
            for position, card in enumerate(cards, 1):
                try:
                    character_name = card.select_one('h2.font-bold').get_text(strip=True)
                    guild_name = card.select_one('p.text-muted-foreground').get_text(strip=True)
                    
                    class_icon = card.select_one('img[alt^="Icon"]')
                    class_src = class_icon['srcset'].split(' ')[0] if class_icon else ''
                    
                    class_info = None
                    for _, info in CLASS_MAPPING.items():
                        if info['icon'] in class_src:
                            class_info = info
                            break
                    
                    if not class_info:
                        class_info = {'name': 'Unknown', 'name_pt': 'Desconhecida', 'short': 'UNK'}
                    
                    nation_icon = card.select_one('img[alt="Icon Nation"]')
                    nation_src = nation_icon['srcset'].split(' ')[0] if nation_icon else ''
                    
                    nation_info = None
                    for icon, info in NATION_MAPPING.items():
                        if icon in nation_src:
                            nation_info = info
                            break
                    
                    if not nation_info:
                        nation_info = {'name': 'Unknown', 'name_pt': 'Desconhecida'}
                    
                    memorial_entry = {
                        'position': position,
                        'character_name': character_name,
                        'character_class': {
                            'en': class_info['name'],
                            'pt': class_info['name_pt'],
                            'short': class_info['short']
                        },
                        'guild_name': guild_name,
                        'nation': {
                            'en': nation_info['name'],
                            'pt': nation_info['name_pt']
                        }
                    }
                    memorial_data.append(memorial_entry)
                    
                except Exception as e:
                    logger.warning(f"Erro ao processar memorial {position}: {e}")
                    continue
            
            return memorial_data
            
        except Exception as e:
            logger.error(f"Erro ao analisar ranking memorial: {e}")
            raise

    def parse_power_ranking(self, html_content):
        """
        Analisa o HTML para extrair dados do ranking de power.
        """
        logger.info("Analisando dados do ranking de power")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        power_data = []
        
        try:
            rows = soup.find_all('tr')[1:]  # Pula o cabeçalho
            
            for position, row in enumerate(rows, 1):
                try:
                    cells = row.find_all(['td'])
                    if len(cells) >= 7:
                        # Identifica a classe
                        class_cell = cells[1]
                        class_info = None
                        
                        # Tenta encontrar a imagem da classe
                        class_img = class_cell.find('img')
                        
                        if class_img:
                            # Tenta identificar pelo srcset
                            if 'srcset' in class_img.attrs:
                                srcset = class_img['srcset']
                                for class_id, info in CLASS_MAPPING.items():
                                    icon_pattern = f"/ranking/icon-{info['icon']}.png"
                                    if icon_pattern in srcset:
                                        class_info = info
                                        break
                            
                            # Se não achou pelo srcset, tenta pelo alt
                            if not class_info and 'alt' in class_img.attrs:
                                alt_text = class_img['alt']
                                for class_id, info in CLASS_MAPPING.items():
                                    if info['alt'] == alt_text:
                                        class_info = info
                                        break
                        
                        # Se não encontrou pela imagem, tenta pelo texto da célula
                        if not class_info:
                            class_text = class_cell.get_text(strip=True)
                            for class_id, info in CLASS_MAPPING.items():
                                if (info['name_pt'].lower() in class_text.lower() or 
                                    info['name'].lower() in class_text.lower() or 
                                    info['short'].lower() in class_text.lower()):
                                    class_info = info
                                    break
                        
                        # Se ainda não encontrou, tenta pelo conteúdo da célula
                        if not class_info:
                            cell_html = str(class_cell)
                            for class_id, info in CLASS_MAPPING.items():
                                icon_pattern = f"icon-{info['icon']}"
                                if icon_pattern in cell_html:
                                    class_info = info
                                    break
                        
                        # Se ainda não encontrou, usa valor padrão
                        if not class_info:
                            class_info = {
                                'name': 'Unknown',
                                'name_pt': 'Desconhecida',
                                'short': 'UNK'
                            }
                            # Log para debug quando não encontra a classe
                            logger.debug(f"Classe não identificada para posição {position}. HTML da célula: {class_cell}")
                        
                        # Identifica a nação
                        nation_cell = cells[7] if len(cells) >= 8 else None
                        nation_info = None
                        
                        if nation_cell:
                            nation_img = nation_cell.find('img')
                            if nation_img and 'srcset' in nation_img.attrs:
                                srcset = nation_img['srcset']
                                if 'icon-procyon.png' in srcset:
                                    nation_info = NATION_MAPPING['icon-procyon']
                                elif 'icon-capella.png' in srcset:
                                    nation_info = NATION_MAPPING['icon-capella']
                        
                        if not nation_info:
                            nation_info = {
                                'name': 'Unknown',
                                'name_pt': 'Desconhecida'
                            }
                        
                        power_entry = {
                            'position': position,
                            'class': {
                                'en': class_info['name'],
                                'pt': class_info['name_pt'],
                                'short': class_info['short']
                            },
                            'name': cells[2].get_text(strip=True),
                            'guild': cells[3].get_text(strip=True),
                            'attack_power': self.parse_value(cells[4].get_text(strip=True)),
                            'defense_power': self.parse_value(cells[5].get_text(strip=True)),
                            'total_power': self.parse_value(cells[6].get_text(strip=True)),
                            'nation': {
                                'en': nation_info['name'],
                                'pt': nation_info['name_pt']
                            }
                        }
                        
                        power_data.append(power_entry)
                        
                        # Log detalhado para debug
                        logger.debug(
                            f"Processado: {power_entry['name']} - "
                            f"Classe: {power_entry['class']['pt']} ({power_entry['class']['short']}) - "
                            f"Nação: {power_entry['nation']['pt']} - "
                            f"Power: {power_entry['total_power']:,}"
                        )
                        
                except Exception as e:
                    logger.warning(f"Erro ao processar power {position}: {e}")
                    continue
            
            return power_data
            
        except Exception as e:
            logger.error(f"Erro ao analisar ranking de power: {e}")
            raise

    def save_ranking_data(self, data, ranking_type, class_id=None):
        """
        Salva os dados do ranking em arquivo JSON.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if class_id:
            class_info = CLASS_MAPPING[class_id]
            filename = f"{ranking_type}_ranking_{class_info['short'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            filename = f"{ranking_type}_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            headers = getattr(self, f"{ranking_type}_headers")
            
            output_data = {
                'timestamp': timestamp,
                'total_entries': len(data),
                'class_info': CLASS_MAPPING[class_id] if class_id else None,
                'headers': headers,
                'rankings': data
            }
            
            # Salva o arquivo JSON
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Ranking salvo em: {filename}")
            
            # Mostra exemplo dos primeiros colocados
            if data:
                class_name = CLASS_MAPPING[class_id]['name_pt'] if class_id else 'Geral'
                logger.info(f"\nTop 3 {class_name}:")
                for entry in data[:3]:
                    if ranking_type == 'power':
                        logger.info(
                            f"#{entry['position']}: {entry['name']} "
                            f"({entry['class']['pt']}) - "
                            f"Guild: {entry['guild']} - "
                            f"Power Total: {entry['total_power']:,} - "
                            f"Nação: {entry['nation']['pt']}"
                        )
                    elif ranking_type == 'memorial':
                        logger.info(
                            f"#{entry['position']}: {entry['character_name']} "
                            f"({entry['character_class']['pt']}) - "
                            f"Guild: {entry['guild_name']} - "
                            f"Nação: {entry['nation']['pt']}"
                        )
                    else:  # guild
                        logger.info(
                            f"#{entry['position']}: {entry['name']} - "
                            f"Poder: {entry['power']:,} - "
                            f"Membros: {entry['members']} - "
                            f"Pontos Guerra: {entry['war_points']:,}"
                        )
        
        except Exception as e:
            logger.error(f"Erro ao salvar ranking: {e}")

def main():
    """
    Função principal para extrair todos os rankings.
    """
    try:
        extractor = RankingExtractor()
        
        # 1. Extrai ranking de guild
        logger.info("\n=== Extraindo Ranking de Guilds ===")
        html_content = extractor.fetch_page_content(GUILD_RANKING_URL)
        if html_content:
            guild_data = extractor.parse_guild_ranking(html_content)
            if guild_data:
                extractor.save_ranking_data(guild_data, 'guild')
        
        # 2. Extrai ranking memorial
        logger.info("\n=== Extraindo Ranking Memorial ===")
        html_content = extractor.fetch_page_content(
            MEMORIAL_RANKING_URL,
            wait_selector='div.grid.grid-cols-1'
        )
        if html_content:
            memorial_data = extractor.parse_memorial_ranking(html_content)
            if memorial_data:
                extractor.save_ranking_data(memorial_data, 'memorial')
        
        # 3. Extrai rankings de power (geral e por classe)
        logger.info("\n=== Extraindo Rankings de Power ===")
        
        # Ranking geral primeiro
        logger.info("Extraindo ranking geral de power")
        html_content = extractor.fetch_page_content(POWER_RANKING_URL)
        if html_content:
            power_data = extractor.parse_power_ranking(html_content)
            if power_data:
                extractor.save_ranking_data(power_data, 'power')
        
        # Rankings por classe
        for class_id in CLASS_MAPPING.keys():
            class_info = CLASS_MAPPING[class_id]
            logger.info(f"\nExtraindo ranking de power para {class_info['name_pt']} ({class_info['short']})")
            
            url = f"{POWER_RANKING_URL}?classId={class_id}"
            html_content = extractor.fetch_page_content(url)
            
            if html_content:
                power_data = extractor.parse_power_ranking(html_content)
                if power_data:
                    extractor.save_ranking_data(power_data, 'power', class_id)
            
            # Pequena pausa entre requisições
            time.sleep(1)
        
        logger.info("\n=== Extração de rankings concluída com sucesso! ===")
        
    except Exception as e:
        logger.error(f"Falha ao processar rankings: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExtração interrompida pelo usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        raise