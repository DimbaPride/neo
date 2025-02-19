import os
import asyncio
import logging
from enum import Enum
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import time
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URLs base
BASE_URL = "https://www.neogames.online/ranking"
GUILD_RANKING_URL = f"{BASE_URL}/guild"
MEMORIAL_RANKING_URL = f"{BASE_URL}/memorial"
POWER_RANKING_URL = f"{BASE_URL}/power"
WAR_RANKING_URL = f"{BASE_URL}/war"


RANKING_TYPE_POWER = "power"
RANKING_TYPE_GUILD = "guild"
RANKING_TYPE_MEMORIAL = "memorial"
RANKING_TYPE_WAR = "war"



# Mapeamento atualizado de classes com nomes em português e inglês
# Mapeamento de classes
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

# Mapeamento de nações
NATION_MAPPING = {
    'icon-procyon': {
        'name': 'Procyon',
        'name_pt': 'Procyon',
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

class NeoGamesRankings:
    def __init__(self, base_dir: str = "knowledge_base/ranking"):
        self.base_dir = base_dir
        self.embeddings = OpenAIEmbeddings()
        
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
        self.war_headers = {
        'position': '#',
        'name': 'Nome',
        'guild': 'Guild',
        'kills': 'Abates',
        'deaths': 'Mortes',
        'kd_ratio': 'K/D',
        'nation': 'Nação'
    }        
        self._setup_directories()

    def _setup_directories(self):
        """Cria a estrutura de diretórios necessária para os rankings."""
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Cria diretórios base para cada tipo de ranking
        for ranking_type in ['power', 'guild', 'memorial', 'war']:
            path = os.path.join(self.base_dir, ranking_type)
            os.makedirs(path, exist_ok=True)
            
            # Apenas para o ranking de power, cria subpastas
            if ranking_type == 'power':
                os.makedirs(os.path.join(path, "general"), exist_ok=True)
                for class_info in CLASS_MAPPING.values():
                    os.makedirs(os.path.join(path, class_info['short'].lower()), exist_ok=True)


    async def fetch_page_content(self, url: str, wait_selector='table', timeout=30000) -> str:
        """
        Busca o conteúdo de uma página usando Playwright de forma assíncrona.
        
        Args:
            url (str): URL da página a ser buscada
            wait_selector (str): Seletor CSS para aguardar carregar
            timeout (int): Tempo máximo de espera em ms
            
        Returns:
            str: Conteúdo HTML da página
        """
        logger.info(f"Acessando URL: {url}")
        
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage']
                )
                
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
                )
                
                page = await context.new_page()
                page.set_default_timeout(60000)
                
                response = await page.goto(url)
                
                if not response or response.status != 200:
                    raise Exception(f"Falha ao carregar página: Status {response.status if response else 'N/A'}")
                
                await page.wait_for_selector(wait_selector, timeout=timeout)
                await page.wait_for_timeout(3000)  # Espera adicional para carregamento dinâmico
                
                content = await page.content()
                await browser.close()
                return content
                
            except Exception as e:
                logger.error(f"Erro ao buscar página {url}: {e}")
                if 'browser' in locals():
                    await browser.close()
                raise

    def parse_value(self, value_str: str) -> int:
        """
        Converte valores string para formato numérico.
        
        Args:
            value_str (str): String contendo o valor a ser convertido
            
        Returns:
            int: Valor numérico convertido
        """
        try:
            clean_value = ''.join(c for c in value_str if c.isdigit() or c in '.,')
            clean_value = clean_value.replace('.', '').replace(',', '.')
            return int(float(clean_value))
        except:
            return 0

    def get_class_info(self, class_text_or_icon: str) -> Dict:
        """
        Identifica a classe baseado no texto ou ícone.
        
        Args:
            class_text_or_icon (str): Texto ou ícone da classe
            
        Returns:
            Dict: Informações da classe identificada
        """
        if not class_text_or_icon:
            return {
                'name': 'Unknown',
                'name_pt': 'Desconhecida',
                'short': 'UNK',
                'icon': 'unknown'
            }
            
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

    def get_nation_info(self, cell_or_src) -> Dict:
        """
        Identifica a nação baseado na célula da tabela ou src da imagem.
        
        Args:
            cell_or_src: Célula BeautifulSoup ou string contendo informação da nação
            
        Returns:
            Dict: Informações da nação identificada
        """
        try:
            # Se for uma célula BeautifulSoup
            if hasattr(cell_or_src, 'find'):
                img = cell_or_src.find('img')
                if img and 'srcset' in img.attrs:
                    srcset = img['srcset']
                    
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
        
        return {
            'name': 'Unknown',
            'name_pt': 'Desconhecida',
            'icon_alt': 'Unknown'
        }

        # ... (código anterior permanece igual)

    def parse_power_ranking(self, html_content: str) -> List[Dict]:
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
                        if class_img and 'srcset' in class_img.attrs:
                            srcset = class_img['srcset']
                            for class_id, info in CLASS_MAPPING.items():
                                if f"icon-{info['icon']}" in srcset:
                                    class_info = info
                                    break
                        
                        # Se não encontrou a classe, usa valor padrão
                        if not class_info:
                            class_info = {
                                'name': 'Unknown',
                                'name_pt': 'Desconhecida',
                                'short': 'UNK'
                            }
                            logger.debug(f"Classe não identificada para posição {position}. HTML da célula: {class_cell}")
                        
                        # Identifica a nação
                        nation_cell = cells[7] if len(cells) >= 8 else None
                        nation_info = self.get_nation_info(nation_cell) if nation_cell else {
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
                        
                except Exception as e:
                    logger.warning(f"Erro ao processar power {position}: {e}")
                    continue
            
            return power_data
            
        except Exception as e:
            logger.error(f"Erro ao analisar ranking de power: {e}")
            raise

    def parse_guild_ranking(self, html_content: str) -> List[Dict]:
        """
        Analisa o HTML para extrair dados do ranking de guild.
        
        Args:
            html_content (str): Conteúdo HTML da página de ranking
            
        Returns:
            List[Dict]: Lista de dicionários com os dados do ranking de guild
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

    def parse_memorial_ranking(self, html_content: str) -> List[Dict]:
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
                    
                    # Nova lógica de identificação de classe
                    class_icon = card.select_one('img[alt^="Icon"]')
                    class_info = None
                    
                    if class_icon:
                        # Tenta identificar pelo srcset
                        if 'srcset' in class_icon.attrs:
                            srcset = class_icon['srcset']
                            for class_id, info in CLASS_MAPPING.items():
                                icon_pattern = f"icon-{info['icon']}"
                                if icon_pattern in srcset:
                                    class_info = info
                                    break
                        
                        # Se não achou pelo srcset, tenta pelo alt
                        if not class_info and 'alt' in class_icon.attrs:
                            alt_text = class_icon['alt']
                            for class_id, info in CLASS_MAPPING.items():
                                if info['alt'] == alt_text:
                                    class_info = info
                                    break
                    
                    # Se ainda não encontrou a classe, usa valor padrão
                    if not class_info:
                        class_info = {
                            'name': 'Unknown',
                            'name_pt': 'Desconhecida',
                            'short': 'UNK'
                        }
                        logger.debug(f"Classe não identificada para {character_name}. HTML do ícone: {class_icon}")
                    
                    # Usando a mesma lógica do power.py para nação
                    nation_img = card.select_one('img[srcset*="procyon.png"]')
                    nation_info = None
                    
                    if nation_img:
                        nation_info = NATION_MAPPING['icon-procyon']
                    else:
                        nation_img = card.select_one('img[srcset*="capella.png"]')
                        if nation_img:
                            nation_info = NATION_MAPPING['icon-capella']
                    
                    if not nation_info:
                        nation_info = {
                            'name': 'Unknown',
                            'name_pt': 'Desconhecida',
                            'icon_alt': 'Unknown',
                            'icon_src': 'unknown'
                        }
                    
                    memorial_entry = {
                        'position': position,
                        'character_name': character_name,
                        'character_class': {
                            'name': class_info['name'],
                            'name_pt': class_info['name_pt'],
                            'short': class_info['short']
                        },
                        'guild_name': guild_name,
                        'nation': {
                            'en': nation_info['name'],
                            'pt': nation_info['name_pt']
                        }
                    }
                    memorial_data.append(memorial_entry)
                    
                    # Log para debug da identificação de classe
                    logger.debug(
                        f"Memorial #{position}: {character_name} - "
                        f"Classe: {class_info['name_pt']} ({class_info['short']})"
                    )
                    
                except Exception as e:
                    logger.warning(f"Erro ao processar memorial {position}: {e}")
                    continue
            
            return memorial_data
            
        except Exception as e:
            logger.error(f"Erro ao analisar ranking memorial: {e}")
            raise

    def parse_war_ranking(self, html_content: str) -> Dict[str, List[Dict]]:
        """
        Analisa o HTML para extrair dados do ranking de war e pontuação semanal.
        """
        logger.info("Analisando dados do ranking de war e pontuação semanal")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        war_roles_data = []
        weekly_scores_data = []
        
        try:
            tables = soup.find_all('table', class_='w-full')
            logger.debug(f"Encontradas {len(tables)} tabelas")
            
            for table in tables:
                rows = table.find_all('tr')
                if not rows:
                    continue
                    
                # Verifica o tipo de tabela pelo número de células no cabeçalho
                header_cells = rows[0].find_all(['th'])
                data_rows = rows[1:]  # Pula o cabeçalho
                
                if len(header_cells) == 4:  # Tabela de Guardiões/Portadores
                    for row in data_rows:
                        try:
                            cells = row.find_all(['td'])
                            if len(cells) >= 4:  # Classe, Nome, Guild, Tipo
                                # Identifica a classe
                                class_img = cells[0].find('img')
                                class_info = None
                                
                                if class_img and 'srcset' in class_img.attrs:
                                    srcset = class_img['srcset']
                                    for class_id, info in CLASS_MAPPING.items():
                                        if f"icon-{info['icon']}" in srcset:
                                            class_info = info
                                            break
                                
                                if not class_info:
                                    class_info = {
                                        'name': 'Unknown',
                                        'name_pt': 'Desconhecida',
                                        'short': 'UNK'
                                    }
                                
                                # Determina o tipo (Portador ou Guardião)
                                type_cell = cells[3]
                                role_type = 'Portador' if 'text-brand' in type_cell.get('class', []) else 'Guardião'
                                
                                # Nova lógica de detecção de nação usando a mesma abordagem do memorial
                                nation_img = table.find_previous('img', srcset=lambda x: 'procyon-main.png' in x if x else False)
                                nation = None
                                
                                if nation_img:
                                    nation = NATION_MAPPING['icon-procyon']
                                else:
                                    nation_img = table.find_previous('img', srcset=lambda x: 'capella-main.png' in x if x else False)
                                    if nation_img:
                                        nation = NATION_MAPPING['icon-capella']

                                if not nation:
                                    nation = {
                                        'name': 'Unknown',
                                        'name_pt': 'Desconhecida'
                                    }
                                
                                entry = {
                                    'name': cells[1].get_text(strip=True),
                                    'class': {
                                        'name': class_info['name'],
                                        'name_pt': class_info['name_pt'],
                                        'short': class_info['short']
                                    },
                                    'guild': cells[2].get_text(strip=True),
                                    'role': role_type,
                                    'nation': {
                                        'en': nation['name'],
                                        'pt': nation['name_pt']
                                    }
                                }
                                war_roles_data.append(entry)
                        except Exception as e:
                            logger.error(f"Erro ao processar linha de roles: {e}")
                            continue
                            
                elif len(header_cells) == 7:  # Tabela de pontuação semanal
                    for row in data_rows:
                        try:
                            cells = row.find_all(['td'])
                            if len(cells) >= 7:  # Posição, Classe, Nome, Guild, Pontos, Abates, Nação
                                position = int(cells[0].get_text(strip=True))
                                
                                # Classe
                                class_img = cells[1].find('img')
                                class_info = None
                                if class_img and 'srcset' in class_img.attrs:
                                    srcset = class_img['srcset']
                                    for class_id, info in CLASS_MAPPING.items():
                                        if f"icon-{info['icon']}" in srcset:
                                            class_info = info
                                            break
                                
                                if not class_info:
                                    class_info = {
                                        'name': 'Unknown',
                                        'name_pt': 'Desconhecida',
                                        'short': 'UNK'
                                    }
                                
                                # Nação
                                nation_cell = cells[6]
                                nation_img = nation_cell.find('img')
                                nation = None
                                if nation_img and 'srcset' in nation_img.attrs:
                                    srcset = nation_img['srcset']
                                    if 'icon-capella.png' in srcset:
                                        nation = NATION_MAPPING['icon-capella']
                                    elif 'icon-procyon.png' in srcset:
                                        nation = NATION_MAPPING['icon-procyon']
                                
                                if not nation:
                                    nation = {
                                        'name': 'Unknown',
                                        'name_pt': 'Desconhecida'
                                    }
                                
                                entry = {
                                    'position': position,
                                    'name': cells[2].get_text(strip=True),
                                    'class': {
                                        'name': class_info['name'],
                                        'name_pt': class_info['name_pt'],
                                        'short': class_info['short']
                                    },
                                    'guild': cells[3].get_text(strip=True),
                                    'points': self.parse_value(cells[4].get_text(strip=True)),
                                    'kills': self.parse_value(cells[5].get_text(strip=True)),
                                    'nation': {
                                        'en': nation['name'],
                                        'pt': nation['name_pt']
                                    }
                                }
                                weekly_scores_data.append(entry)
                        except Exception as e:
                            logger.error(f"Erro ao processar linha semanal: {e}")
                            continue
                            
            total_roles = len(war_roles_data)
            total_scores = len(weekly_scores_data)
            logger.info(f"Total processado - Roles: {total_roles}, Scores: {total_scores}")
            
            return {
                'war_roles': war_roles_data,
                'weekly_scores': weekly_scores_data
            }
            
        except Exception as e:
            logger.error(f"Erro ao analisar rankings de war: {e}")
            raise

    def save_ranking_data(self, data: Union[List[Dict], Dict[str, List[Dict]]], ranking_type: str, class_id: Optional[int] = None):
        """
        Salva os dados do ranking apenas em JSON.
        Mantém apenas um arquivo JSON atualizado para cada tipo de ranking.
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Define o diretório de saída
            if ranking_type == 'power':
                if class_id:
                    class_info = CLASS_MAPPING.get(class_id, {
                        'name': 'Unknown',
                        'name_pt': 'Desconhecida',
                        'short': 'UNK'
                    })
                    subfolder = class_info['short'].lower()
                else:
                    subfolder = "general"
                out_dir = os.path.join(self.base_dir, ranking_type, subfolder)
            else:
                out_dir = os.path.join(self.base_dir, ranking_type)
            
            os.makedirs(out_dir, exist_ok=True)
            
            # Tratamento especial para o ranking de war que tem dois tipos de dados
            if ranking_type == 'war' and isinstance(data, dict):
                # Salva os dados de roles (Guardiões/Portadores)
                if 'war_roles' in data:
                    roles_data = {
                        'timestamp': timestamp,
                        'total_entries': len(data['war_roles']),
                        'rankings': data['war_roles']
                    }
                    roles_path = os.path.join(out_dir, 'ranking_roles.json')
                    with open(roles_path, 'w', encoding='utf-8') as f:
                        json.dump(roles_data, f, ensure_ascii=False, indent=2)
                
                # Salva os dados de pontuação semanal
                if 'weekly_scores' in data:
                    weekly_data = {
                        'timestamp': timestamp,
                        'total_entries': len(data['weekly_scores']),
                        'rankings': data['weekly_scores']
                    }
                    weekly_path = os.path.join(out_dir, 'ranking_weekly.json')
                    with open(weekly_path, 'w', encoding='utf-8') as f:
                        json.dump(weekly_data, f, ensure_ascii=False, indent=2)
            else:
                # Nome do arquivo JSON baseado no tipo e classe
                if ranking_type == 'power' and class_id:
                    json_filename = f"ranking_{class_info['short'].lower()}.json"
                else:
                    json_filename = f"ranking_{ranking_type}.json"
                
                json_path = os.path.join(out_dir, json_filename)
                
                # Prepara e salva dados em JSON
                output_data = {
                    'timestamp': timestamp,
                    'total_entries': len(data),
                    'rankings': data
                }
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Dados JSON atualizados em: {json_path}")
                
        except Exception as e:
            logger.error(f"Erro ao salvar ranking: {e}")
            raise
        
    def _log_top_entries(self, entries: List[Dict], ranking_type: str):
        """
        Função auxiliar para logar os primeiros colocados.
        
        Args:
            entries (List[Dict]): Lista de entradas do ranking
            ranking_type (str): Tipo do ranking
        """
        for entry in entries:
            if ranking_type == 'power':
                logger.info(
                    f"#{entry['position']}: {entry['name']} "
                    f"({entry['class']['name_pt']}) - "
                    f"Guild: {entry['guild']} - "
                    f"Power Total: {entry['total_power']:,}"
                )
            elif ranking_type == 'memorial':
                logger.info(
                    f"#{entry['position']}: {entry['character_name']} "
                    f"({entry['character_class']['name_pt']}) - "
                    f"Guild: {entry['guild_name']}"
                )
            elif ranking_type == 'war':
                logger.info(
                    f"#{entry['position']}: {entry['name']} - "
                    f"Guild: {entry['guild']} - "
                    f"K/D: {entry['kd_ratio']:.2f} "
                    f"(Abates: {entry['kills']:,}, Mortes: {entry['deaths']:,})"
                )
            else:  # guild
                logger.info(
                    f"#{entry['position']}: {entry['name']} - "
                    f"Poder: {entry['power']:,} - "
                    f"Membros: {entry['members']}"
                )
                
    def query(self, question: str, ranking_types: Optional[List[str]] = None, k: int = 3, class_abbr: Optional[str] = None) -> str:
        """
        Query flexível para rankings - o agente decide como usar baseado na pergunta
        """
        try:
            # Se não especificou tipo, usa todos
            if ranking_types is None:
                ranking_types = ['power', 'guild', 'memorial', 'war']

            responses = []
            for ranking_type in ranking_types:
                # Pega o JSON correto
                json_path = self._get_json_path(ranking_type, class_abbr)
                
                # Se existe o arquivo
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Pega os rankings
                    rankings = data.get('rankings', [])
                    if rankings:
                        response = self.format_ranking_response(rankings, ranking_type)
                        if response:
                            responses.append(response)

            # Se encontrou algum ranking, retorna
            if responses:
                return "\n\n".join(responses)
            
            return ""  # Retorna vazio para o agente tentar outra ferramenta

        except Exception as e:
            logger.error(f"Erro consultando rankings: {e}")
            return ""

    def format_ranking_response(self, rankings: List[Dict], ranking_type: str) -> str:
        """Formata os rankings de forma amigável"""
        try:
            # 1. Ranking de War (Portadores e Guardiões)
            if ranking_type == 'war':
                # Filtra os dados por nação
                capella = [r for r in rankings if r['nation'].get('pt') == 'Capella']
                procyon = [r for r in rankings if r['nation'].get('pt') == 'Procyon']
                
                response = []
                
                # Capella
                response.append("=== CAPELLA ===")
                portadores = [r for r in capella if r['role'] == 'Portador']
                guardioes = [r for r in capella if r['role'] == 'Guardião']
                
                if portadores:
                    response.append("PORTADOR:")
                    for p in portadores:
                        response.append(f"• {p['name']} ({p['class']['pt']}) - Guild: {p['guild']}")
                
                if guardioes:
                    response.append("\nGUARDIÕES:")
                    for g in guardioes:
                        response.append(f"• {g['name']} ({g['class']['pt']}) - Guild: {g['guild']}")
                
                # Procyon
                response.append("\n=== PROCYON ===")
                portadores = [r for r in procyon if r['role'] == 'Portador']
                guardioes = [r for r in procyon if r['role'] == 'Guardião']
                
                if portadores:
                    response.append("PORTADOR:")
                    for p in portadores:
                        response.append(f"• {p['name']} ({p['class']['pt']}) - Guild: {p['guild']}")
                
                if guardioes:
                    response.append("\nGUARDIÕES:")
                    for g in guardioes:
                        response.append(f"• {g['name']} ({g['class']['pt']}) - Guild: {g['guild']}")
                
                return "\n".join(response)

            # 2. Ranking de Guild
            elif ranking_type == 'guild':
                response = ["=== RANKING DE GUILDS ==="]
                for r in rankings:
                    response.append(
                        f"#{r['position']} - {r['name']}\n"
                        f"• Power: {r['power']:,}\n"
                        f"• Membros: {r['members']}\n"
                        f"• Pontos Guerra: {r['war_points']:,}\n"
                        f"• Abates Guerra: {r['war_kills']:,}"
                    )
                return "\n\n".join(response)

            # 3. Ranking de Power
            elif ranking_type == 'power':
                response = ["=== POWER RANKING ==="]
                for r in rankings:
                    response.append(
                        f"#{r['position']} - {r['name']} ({r['class']['pt']})\n"
                        f"• Guild: {r['guild']}\n"
                        f"• Power Total: {r['total_power']:,}\n"
                        f"• ATK: {r['attack_power']:,} | DEF: {r['defense_power']:,}\n"
                        f"• Nação: {r['nation']['pt']}"
                    )
                return "\n\n".join(response)

            # 4. Ranking Memorial
            elif ranking_type == 'memorial':
                # Organiza por nação
                capella = [r for r in rankings if r['nation'].get('pt') == 'Capella']
                procyon = [r for r in rankings if r['nation'].get('pt') == 'Procyon']
                
                response = []
                
                if capella:
                    response.append("=== MEMORIAL CAPELLA ===")
                    for r in capella:
                        response.append(
                            f"#{r['position']} - {r['character_name']} "
                            f"({r['character_class']['name_pt']}) "
                            f"- Guild: {r['guild_name']}"
                        )
                
                if procyon:
                    response.append("\n=== MEMORIAL PROCYON ===")
                    for r in procyon:
                        response.append(
                            f"#{r['position']} - {r['character_name']} "
                            f"({r['character_class']['name_pt']}) "
                            f"- Guild: {r['guild_name']}"
                        )
                
                return "\n".join(response)

            return "Tipo de ranking não reconhecido"

        except Exception as e:
            logger.error(f"Erro formatando ranking {ranking_type}: {e}")
            return "Erro ao formatar o ranking. Por favor, tente novamente."

    def _get_json_path(self, ranking_type: str, class_abbr: Optional[str] = None) -> str:
        """Retorna o caminho correto do arquivo JSON baseado no tipo e classe."""
        if ranking_type == 'power':
            subfolder = "general"
            if class_abbr:
                class_abbr = class_abbr.upper()
                for _, info in CLASS_MAPPING.items():
                    if info['short'] == class_abbr:
                        subfolder = class_abbr.lower()
                        break
            return os.path.join(self.base_dir, ranking_type, subfolder, f"ranking_{subfolder}.json")
        return os.path.join(self.base_dir, ranking_type, f"ranking_{ranking_type}.json")

    def _filter_rankings(self, rankings: List[Dict], question: str, patterns: Dict[str, bool]) -> List[Dict]:
        """Filtra os rankings baseado nos padrões identificados na pergunta."""
        try:
            # Busca por nome específico
            if patterns['player_search']:
                player_name = self._extract_name(question)
                if player_name:
                    return [r for r in rankings if player_name.lower() in r.get('name', '').lower()]

            # Busca por guild específica
            if patterns['guild_search']:
                guild_name = self._extract_name(question)
                if guild_name:
                    return [r for r in rankings if guild_name.lower() in r.get('guild', '').lower()]

            # Busca por range de posições
            if patterns['range']:
                start, end = self._extract_range(question)
                if start is not None and end is not None:
                    return [r for r in rankings if start <= r.get('position', 0) <= end]

            # Busca por posição específica
            if patterns['specific_position']:
                position = self._extract_position(question)
                if position:
                    return [r for r in rankings if r.get('position') == position]

            # Papéis específicos da guerra
            if patterns['war_roles']:
                return [r for r in rankings if r.get('role') in ['Guardião', 'Portador']]

            # Top N (padrão)
            if patterns['top_n']:
                n = self._extract_number(question) or 3
                return rankings[:n]

            # Se nenhum padrão específico foi identificado, retorna top 3
            return rankings[:3]

        except Exception as e:
            logger.error(f"Erro ao filtrar rankings: {e}")
            return rankings[:3]

    def _extract_name(self, question: str) -> Optional[str]:
        """Extrai nome de player ou guild da pergunta."""
        # Implementar lógica de extração de nome
        return None

    def _extract_range(self, question: str) -> Tuple[Optional[int], Optional[int]]:
        """Extrai range de posições da pergunta."""
        # Implementar lógica de extração de range
        return None, None

    def _extract_position(self, question: str) -> Optional[int]:
        """Extrai posição específica da pergunta."""
        # Implementar lógica de extração de posição
        return None

    def _extract_number(self, question: str) -> Optional[int]:
        """Extrai número (ex: top N) da pergunta."""
        # Implementar lógica de extração de número
        return None

    def _format_header(self, ranking_type: str, class_abbr: Optional[str] = None) -> str:
        """Formata o cabeçalho da resposta."""
        header = f"[{ranking_type.upper()}"
        if ranking_type == 'power' and class_abbr:
            header += f" - {class_abbr.upper()}"
        header += "]"
        return header

    async def process_ranking(self, ranking_type: str):
        try:
            if ranking_type == RANKING_TYPE_POWER:
                # Processa ranking geral primeiro
                logger.info("Processando ranking geral de power")
                html_content = await self.fetch_page_content(POWER_RANKING_URL)
                if html_content:
                    power_data = self.parse_power_ranking(html_content)
                    if power_data:
                        # Passando None como class_id para o ranking geral
                        self.save_ranking_data(power_data, ranking_type, class_id=None)
                
                # Processa rankings por classe
                for class_id in CLASS_MAPPING.keys():
                    class_info = CLASS_MAPPING[class_id]
                    logger.info(f"Processando ranking de power para {class_info['name_pt']} ({class_info['short']})")
                    
                    url = f"{POWER_RANKING_URL}?classId={class_id}"
                    html_content = await self.fetch_page_content(url)
                    
                    if html_content:
                        power_data = self.parse_power_ranking(html_content)
                        if power_data:
                            # Passando class_id explicitamente
                            self.save_ranking_data(power_data, ranking_type, class_id=class_id)
                    
                    await asyncio.sleep(1)
                    
            elif ranking_type == RANKING_TYPE_GUILD:
                html_content = await self.fetch_page_content(GUILD_RANKING_URL)
                if html_content:
                    guild_data = self.parse_guild_ranking(html_content)
                    if guild_data:
                        # Guild não tem class_id, então passa None
                        self.save_ranking_data(guild_data, ranking_type, class_id=None)
                    
            elif ranking_type == RANKING_TYPE_MEMORIAL:
                html_content = await self.fetch_page_content(
                    MEMORIAL_RANKING_URL,
                    wait_selector='div.grid.grid-cols-1'
                )
                if html_content:
                    memorial_data = self.parse_memorial_ranking(html_content)
                    if memorial_data:
                        # Memorial não tem class_id, então passa None
                        self.save_ranking_data(memorial_data, ranking_type, class_id=None)

            elif ranking_type == RANKING_TYPE_WAR:
                logger.info("Processando ranking de war")
                # Ajustando seletor para esperar tanto a tabela quanto o título
                html_content = await self.fetch_page_content(
                    WAR_RANKING_URL,
                    wait_selector='h2.text-3xl, table.w-full',  # Espera o título da nação ou uma tabela
                    timeout=60000  # Aumenta o timeout para 60 segundos
                )
                if html_content:
                    war_data = self.parse_war_ranking(html_content)
                    if war_data:
                        self.save_ranking_data(war_data, ranking_type, class_id=None)
            
        except Exception as e:
            logger.error(f"Erro ao processar ranking {ranking_type}: {e}")
            raise

    async def initialize(self):
        """
        Inicializa o módulo e processa todos os rankings.
        Este método deve ser chamado após instanciar a classe.
        """
        try:
            logger.info("Iniciando processamento dos rankings...")
            
            # Cria diretórios necessários
            self._setup_directories()
            
            # Lista de rankings para processar
            rankings = ['power', 'guild', 'memorial', 'war']
            
            # Processa cada tipo de ranking
            for ranking_type in rankings:
                try:
                    await self.process_ranking(ranking_type)
                    logger.info(f"Ranking {ranking_type} processado com sucesso")
                except Exception as e:
                    logger.error(f"Erro ao processar ranking {ranking_type}: {e}")
            
            logger.info("Inicialização concluída com sucesso")
            
        except Exception as e:
            logger.error(f"Erro durante a inicialização: {e}")
            raise

    async def update_rankings(self, ranking_types: Optional[List[str]] = None):
        """
        Atualiza rankings específicos ou todos os rankings.
        
        Args:
            ranking_types (Optional[List[str]]): Lista de tipos de ranking para atualizar.
                                               Se None, atualiza todos.
        """
        try:
            if ranking_types is None:
                ranking_types = ['power', 'guild', 'memorial','war']
            
            logger.info(f"Iniciando atualização dos rankings: {', '.join(ranking_types)}")
            
            for ranking_type in ranking_types:
                try:
                    await self.process_ranking(ranking_type)
                    logger.info(f"Ranking {ranking_type} atualizado com sucesso")
                except Exception as e:
                    logger.error(f"Erro ao atualizar ranking {ranking_type}: {e}")
            
            logger.info("Atualização concluída")
            
        except Exception as e:
            logger.error(f"Erro durante a atualização: {e}")
            raise