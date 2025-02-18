import os
import asyncio
import logging
from enum import Enum
from typing import List, Dict, Optional
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
        for ranking_type in ['power', 'guild', 'memorial']:
            path = os.path.join(self.base_dir, ranking_type)
            os.makedirs(path, exist_ok=True)
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
                    
                    # Extrai e processa informação da classe
                    class_icon = card.select_one('img[alt^="Icon"]')
                    class_src = class_icon['srcset'].split(' ')[0] if class_icon else ''
                    class_info = self.get_class_info(class_src)
                    
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
                    
                except Exception as e:
                    logger.warning(f"Erro ao processar memorial {position}: {e}")
                    continue
            
            return memorial_data
            
        except Exception as e:
            logger.error(f"Erro ao analisar ranking memorial: {e}")
            raise

    def parse_war_ranking(self, html_content: str) -> List[Dict]:
        """
        Analisa o HTML para extrair dados do ranking de war.
        """
        logger.info("Analisando dados do ranking de war")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        war_data = []
        
        try:
            rows = soup.find_all('tr')[1:]  # Pula o cabeçalho
            
            for position, row in enumerate(rows, 1):
                try:
                    cells = row.find_all(['td'])
                    if len(cells) >= 6:
                        # Identifica a nação
                        nation_cell = cells[6] if len(cells) >= 7 else None
                        nation_info = self.get_nation_info(nation_cell) if nation_cell else {
                            'name': 'Unknown',
                            'name_pt': 'Desconhecida'
                        }
                        
                        # Calcula K/D ratio
                        kills = self.parse_value(cells[3].get_text(strip=True))
                        deaths = self.parse_value(cells[4].get_text(strip=True))
                        kd_ratio = round(kills / deaths, 2) if deaths > 0 else kills
                        
                        war_entry = {
                            'position': position,
                            'name': cells[1].get_text(strip=True),
                            'guild': cells[2].get_text(strip=True),
                            'kills': kills,
                            'deaths': deaths,
                            'kd_ratio': kd_ratio,
                            'nation': {
                                'en': nation_info['name'],
                                'pt': nation_info['name_pt']
                            }
                        }
                        
                        war_data.append(war_entry)
                        
                except Exception as e:
                    logger.warning(f"Erro ao processar war rank {position}: {e}")
                    continue
            
            return war_data
            
        except Exception as e:
            logger.error(f"Erro ao analisar ranking de war: {e}")
            raise        

    def save_ranking_data(self, data: List[Dict], ranking_type: str, class_id: Optional[int] = None):
        """
        Salva os dados do ranking em JSON e cria índices FAISS.
        Mantém apenas um arquivo JSON atualizado para cada tipo de ranking.
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Define o diretório de saída
            if ranking_type == 'power' and class_id:
                class_info = CLASS_MAPPING.get(class_id, {
                    'name': 'Unknown',
                    'name_pt': 'Desconhecida',
                    'short': 'UNK'
                })
                subfolder = class_info['short'].lower()
            else:
                subfolder = "general"
                
            out_dir = os.path.join(self.base_dir, ranking_type, subfolder)
            os.makedirs(out_dir, exist_ok=True)
            
            # Nome fixo do arquivo JSON baseado no tipo e classe (se aplicável)
            if ranking_type == 'power' and class_id:
                json_filename = f"ranking_{class_info['short'].lower()}.json"
            else:
                json_filename = f"ranking_{ranking_type}.json"
                
            json_path = os.path.join(out_dir, json_filename)
            
            # Prepara dados para JSON
            output_data = {
                'timestamp': timestamp,
                'total_entries': len(data),
                'rankings': data
            }
            
            # Adiciona info da classe se necessário
            if class_id and ranking_type == 'power':
                class_info = CLASS_MAPPING.get(class_id)
                if class_info:
                    output_data['class_info'] = {
                        'id': class_id,
                        'name': class_info['name'],
                        'name_pt': class_info['name_pt'],
                        'short': class_info['short']
                    }
            
            # Salva JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Dados JSON atualizados em: {json_path}")
            
            # Prepara documentos para FAISS
            docs = []
            for entry in data:
                if ranking_type == 'power':
                    content = (
                        f"Rank: {entry['position']}\n"
                        f"Player: {entry['name']}\n"
                        f"Classe: {entry['class']['pt']} ({entry['class']['short']})\n"
                        f"Guild: {entry['guild']}\n"
                        f"Poder Total: {entry['total_power']:,}\n"
                        f"Poder de Ataque: {entry['attack_power']:,}\n"
                        f"Poder de Defesa: {entry['defense_power']:,}\n"
                        f"Nação: {entry['nation']['pt']}"
                    )
                elif ranking_type == 'guild':
                    content = (
                        f"Rank: {entry['position']}\n"
                        f"Guild: {entry['name']}\n"
                        f"Poder: {entry['power']:,}\n"
                        f"Membros: {entry['members']}\n"
                        f"Pontos de Guerra: {entry['war_points']:,}\n"
                        f"Abates na Guerra: {entry['war_kills']:,}"
                    )
                elif ranking_type == 'war':
                    content = (
                        f"Rank: {entry['position']}\n"
                        f"Player: {entry['name']}\n"
                        f"Guild: {entry['guild']}\n"
                        f"Abates: {entry['kills']:,}\n"
                        f"Mortes: {entry['deaths']:,}\n"
                        f"K/D: {entry['kd_ratio']:.2f}\n"
                        f"Nação: {entry['nation']['pt']}"
                    )
                else:  # memorial
                    content = (
                        f"Rank: {entry['position']}\n"
                        f"Player: {entry['character_name']}\n"
                        f"Classe: {entry['character_class']['name_pt']} ({entry['character_class']['short']})\n"
                        f"Guild: {entry['guild_name']}\n"
                        f"Nação: {entry['nation']['pt']}"
                    )
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'timestamp': timestamp,
                        'ranking_type': ranking_type,
                        'position': entry['position'],
                        'class_id': class_id if class_id else None
                    }
                )
                docs.append(doc)
            
            # Cria e salva vectorstore
            if docs:
                vectorstore = FAISS.from_documents(docs, self.embeddings)
                vectorstore.save_local(out_dir)
                logger.info(f"Índice FAISS atualizado em: {out_dir}")
            
            # Log de exemplo dos primeiros colocados
            if data:
                if class_id:
                    class_name = CLASS_MAPPING[class_id]['name_pt']
                else:
                    class_name = 'Geral'
                    
                logger.info(f"\nTop 3 {class_name}:")
                for entry in data[:3]:
                    if ranking_type == 'power':
                        logger.info(
                            f"#{entry['position']}: {entry['name']} "
                            f"({entry['class']['pt']}) - "
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
                
    def query(
        self,
        question: str,
        ranking_types: Optional[List[str]] = None,
        k: int = 3,
        class_abbr: Optional[str] = None
    ) -> str:
        """
        Consulta rankings usando FAISS para busca semântica.
        
        Args:
            question (str): Pergunta ou consulta do usuário
            ranking_types (Optional[List[str]]): Lista de tipos de ranking para consultar
            k (int): Número de resultados a retornar
            class_abbr (Optional[str]): Abreviação da classe para filtrar (power ranking)
            
        Returns:
            str: Resultados formatados da consulta
        """
        if ranking_types is None:
            ranking_types = ['power', 'guild', 'memorial']

        responses = []
        for ranking_type in ranking_types:
            try:
                # Define o caminho do índice
                if ranking_type == 'power':
                    subfolder = "general"
                    if class_abbr:
                        class_abbr = class_abbr.upper()
                        for _, info in CLASS_MAPPING.items():
                            if info['short'] == class_abbr:
                                subfolder = class_abbr.lower()
                                break
                    store_path = os.path.join(self.base_dir, ranking_type, subfolder)
                else:
                    store_path = os.path.join(self.base_dir, ranking_type)

                # Verifica se existe o índice
                if not os.path.exists(store_path):
                    continue

                # Carrega e busca
                vectorstore = FAISS.load_local(
                    store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                docs = vectorstore.similarity_search(question, k=k)

                # Formata resposta
                if docs:
                    header = f"[{ranking_type.upper()}"
                    if ranking_type == 'power' and class_abbr:
                        header += f" - {class_abbr}"
                    header += "]"
                    
                    content = []
                    for doc in docs:
                        if doc.page_content.strip():
                            content.append(doc.page_content.strip())
                    
                    if content:
                        responses.append(f"{header}\n" + "\n\n".join(content))

            except Exception as e:
                logger.error(f"Erro consultando {ranking_type}: {e}")
                continue

        if responses:
            return "\n\n".join(responses)
        return "Dados do ranking não disponíveis no momento. Por favor, tente novamente mais tarde."    

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
                html_content = await self.fetch_page_content(WAR_RANKING_URL)
                if html_content:
                    war_data = self.parse_war_ranking(html_content)
                    if war_data:
                        # War não tem class_id, então passa None
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