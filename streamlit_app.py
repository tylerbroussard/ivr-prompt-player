import streamlit as st
import pandas as pd
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModuleGraph:
    """Class to handle module connectivity analysis"""
    def __init__(self, root: ET.Element):
        self.graph = self._build_graph(root)
        self.reachable_modules = set()
        self.disconnected_modules = self._find_disconnected_modules(root)
        self._find_incoming_call(root)

    def _build_graph(self, root: ET.Element) -> Dict[str, Set[str]]:
        """Build directed graph of module connections"""
        graph = defaultdict(set)
        
        for module in root.findall('.//modules/*'):
            module_id = module.find('moduleId')
            if module_id is not None:
                module_id = module_id.text
                # Add connections
                for tag in ['singleDescendant', 'exceptionalDescendant']:
                    for conn in module.findall(f'./{tag}'):
                        if conn.text:
                            graph[module_id].add(conn.text)
                        
                # Add reverse connections from ascendants
                for conn in module.findall('./ascendants'):
                    if conn.text:
                        if module_id not in graph:
                            graph[module_id] = set()
                        if conn.text not in graph:
                            graph[conn.text] = set()
                        graph[conn.text].add(module_id)
        
        return graph

    def _find_disconnected_modules(self, root: ET.Element) -> set:
        """Find modules that are disconnected (all connections point to self)"""
        disconnected = set()
        
        for module in root.findall('.//modules/*'):
            if module.find('moduleId') is not None:
                module_id = module.find('moduleId').text
                
                # Get all connections
                all_connections = []
                for tag in ['ascendants', 'exceptionalDescendant', 'singleDescendant']:
                    connections = module.findall(f'./{tag}')
                    all_connections.extend([conn.text for conn in connections if conn.text])
                
                # If all connections point to self, module is disconnected
                if all_connections and all(conn == module_id for conn in all_connections):
                    disconnected.add(module_id)
        
        return disconnected

    def _find_incoming_call(self, root: ET.Element) -> None:
        """Find IncomingCall module and compute reachable modules"""
        incoming_call = root.find('.//modules/incomingCall/moduleId')
        if incoming_call is not None:
            self._compute_reachable_modules(incoming_call.text)

    def _compute_reachable_modules(self, start_module: str) -> None:
        """Compute all reachable modules using DFS"""
        stack = [start_module]
        while stack:
            current = stack.pop()
            if current not in self.reachable_modules and current not in self.disconnected_modules:
                self.reachable_modules.add(current)
                for neighbor in self.graph[current]:
                    if neighbor not in self.reachable_modules and neighbor not in self.disconnected_modules:
                        stack.append(neighbor)

    def is_module_reachable(self, module_id: str) -> bool:
        """Check if a module is reachable (not disconnected)"""
        return module_id not in self.disconnected_modules

class PromptAnalyzer:
    """Class to handle prompt analysis"""
    def __init__(self, module_graph: ModuleGraph):
        self.module_graph = module_graph
        self.prompts = {}
        self.announcement_prompts = {}
        
    def process_module(self, module: ET.Element):
        """Process prompts in a module"""
        module_name = module.find('moduleName')
        module_id = module.find('moduleId')
        
        if module_name is not None and module_id is not None:
            module_name = module_name.text
            module_id = module_id.text
            is_reachable = self.module_graph.is_module_reachable(module_id)
            is_skill_transfer = module.tag == 'skillTransfer'
            
            # First, find all announcement prompts and their enabled status
            for announcement in module.findall('.//announcements'):
                enabled = announcement.find('enabled')
                prompt = announcement.find('prompt')
                if prompt is not None and enabled is not None:
                    prompt_id = prompt.find('id')
                    if prompt_id is not None:
                        # For skill transfer modules, if the module is reachable and announcement is enabled,
                        # the prompt is in use regardless of disconnected status
                        announcement_enabled = enabled.text.lower() == 'true'
                        self.announcement_prompts[prompt_id.text] = {
                            'enabled': announcement_enabled,
                            'module_id': module_id,
                            'module_name': module_name,
                            'is_reachable': is_reachable or (is_skill_transfer and announcement_enabled),
                            'is_skill_transfer': is_skill_transfer
                        }
            
            # Process prompts in different possible locations
            prompt_locations = [
                './/filePrompt/promptData/prompt',
                './/announcements/prompt',
                './/prompt/filePrompt/promptData/prompt',
                './/compoundPrompt/filePrompt/promptData/prompt',
                './/promptData/prompt'
            ]
            
            for location in prompt_locations:
                for prompt_elem in module.findall(location):
                    self._add_prompt(prompt_elem, module_name, module_id, is_reachable, is_skill_transfer)
    
    def _add_prompt(self, prompt_elem: ET.Element, module_name: str, module_id: str, is_reachable: bool, is_skill_transfer: bool):
        """Add a prompt to the prompts dictionary"""
        prompt_id = prompt_elem.find('id')
        prompt_name = prompt_elem.find('name')
        
        if prompt_id is not None and prompt_name is not None:
            key = (module_id, prompt_id.text, prompt_name.text)
            
            # Check if this is an announcement prompt
            announcement_info = self.announcement_prompts.get(prompt_id.text)
            is_announcement = announcement_info is not None
            
            # For announcements, use the info from when we found it in the announcements tag
            if is_announcement:
                enabled = announcement_info['enabled']
                module_name = announcement_info['module_name']
                module_id = announcement_info['module_id']
                is_reachable = announcement_info['is_reachable']
                is_skill_transfer = announcement_info['is_skill_transfer']
            else:
                enabled = is_reachable
            
            # Add module type suffix if needed
            module_type = None
            if prompt_elem.find('../../recoEvents') is not None or \
               prompt_elem.find('../../../recoEvents') is not None or \
               prompt_elem.find('../../../../recoEvents') is not None:
                module_type = "Event"
            elif is_announcement:
                module_type = "Announcement"
            elif is_skill_transfer:
                module_type = "Queue"
                
            module_display = f"{module_name} ({module_type})" if module_type else module_name
            
            # For skill transfer announcements that are enabled, show as "In Use"
            if is_skill_transfer and is_announcement and enabled:
                status = '✅ In Use'
            else:
                status = '✅ Enabled' if enabled and is_announcement \
                        else '❌ Disabled' if not enabled and is_announcement \
                        else '✅ In Use' if enabled \
                        else '❌ Not In Use'
            
            self.prompts[key] = {
                'ID': prompt_id.text,
                'Name': prompt_name.text,
                'Module': module_display,
                'ModuleID': module_id,
                'Type': 'Announcement' if is_announcement else 'Play',
                'Status': status,
                'Source File': ''  # Will be filled in later
            }
    
    def get_results(self) -> pd.DataFrame:
        """Convert prompts dictionary to a DataFrame"""
        if not self.prompts:
            return pd.DataFrame()
        
        return pd.DataFrame(list(self.prompts.values()))

def process_ivr_file(file_path: str) -> pd.DataFrame:
    """Process a single IVR file and return a DataFrame of prompts"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Build module graph and analyze prompts
        module_graph = ModuleGraph(root)
        analyzer = PromptAnalyzer(module_graph)
        
        # Process each module
        for module in root.findall('.//modules/*'):
            analyzer.process_module(module)
        
        # Get results and add source file
        results = analyzer.get_results()
        if not results.empty:
            results['Source File'] = os.path.basename(file_path)
        return results
        
    except ET.ParseError as e:
        st.error(f"Error parsing {file_path}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing {file_path}: {str(e)}")
        return pd.DataFrame()

def load_mapping_file() -> Optional[pd.DataFrame]:
    """Load and process the campaign mapping CSV file"""
    try:
        df = pd.read_csv("prompt_campaign_mapping.csv")
        df['Associated Campaigns'] = df['Associated Campaigns'].str.split(',')
        return df
    except FileNotFoundError:
        st.error("prompt_campaign_mapping.csv not found in the current directory")
        return None
    except Exception as e:
        st.error(f"Error reading mapping file: {str(e)}")
        return None

def get_unique_campaigns(df: pd.DataFrame) -> List[str]:
    """Extract unique campaigns from the mapping dataframe"""
    if df is None:
        return []
    all_campaigns = []
    for campaigns in df['Associated Campaigns']:
        if isinstance(campaigns, list):
            all_campaigns.extend(campaigns)
    return sorted(list(set(campaign.strip() for campaign in all_campaigns)))

def create_audio_player(prompt_name: str) -> None:
    """Create an audio player for the given prompt"""
    audio_dir = "./prompts"
    
    # Try both space and underscore versions of the filename
    filenames = [
        f"{prompt_name}.wav",
        f"{prompt_name.replace(' ', '_')}.wav"
    ]
    
    for filename in filenames:
        audio_path = os.path.join(audio_dir, filename)
        if os.path.exists(audio_path):
            with open(audio_path, 'rb') as audio_file:
                st.audio(audio_file.read(), format='audio/wav')
            return
    
    st.warning("⚠️ Audio file not found")
    st.text(f"Looking for: {filenames[0]} or {filenames[1]}")

def main():
    st.set_page_config(page_title="IVR Prompt Player", layout="wide")
    st.title("IVR Prompt Player")
    
    # Load campaign mapping data
    mapping_df = load_mapping_file()
    if mapping_df is None:
        return
    
    # Process IVR files to get prompt statuses
    ivr_dir = "./IVRs"
    prompt_status_df = pd.DataFrame()
    
    try:
        ivr_files = list(Path(ivr_dir).glob('*.five9ivr')) + list(Path(ivr_dir).glob('*.xml'))
        
        if ivr_files:
            for file_path in ivr_files:
                df = process_ivr_file(str(file_path))
                if df is not None:
                    prompt_status_df = pd.concat([prompt_status_df, df], ignore_index=True)
        
        if prompt_status_df.empty:
            st.warning("No IVR files found or processed successfully")
    except Exception as e:
        st.error(f"Error processing IVR files: {str(e)}")
        logger.error("Error in IVR processing", exc_info=True)
    
    # Get unique campaigns
    campaigns = get_unique_campaigns(mapping_df)
    
    # Campaign selector
    selected_campaign = st.selectbox(
        "Select Campaign",
        options=campaigns,
        help="Choose a campaign to view its associated prompts"
    )
    
    # Filter prompts for selected campaign
    campaign_prompts = mapping_df[mapping_df['Associated Campaigns'].apply(
        lambda x: selected_campaign in x if isinstance(x, list) else False
    )]
    
    # Display statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prompts in Selected Campaign", len(campaign_prompts))
    with col2:
        if not prompt_status_df.empty:
            # Since we are now capturing multiple rows per prompt (due to module-based key),
            # "Inactive Prompts" might need more careful grouping logic if desired.
            # For simplicity, we can still count how many are in "❌ Not In Use" overall:
            inactive_count = len(prompt_status_df[prompt_status_df['Status'] == '❌ Not In Use'])
            st.metric("Inactive Prompts", inactive_count)
    
    # Display campaign details
    st.markdown("### Campaign Details")
    st.markdown(f"**Selected Campaign:** {selected_campaign}")
    
    # Display prompts with audio players and status
    st.markdown("### Campaign Prompts")
    
    # Get the IVR files for the selected campaign based on the campaign name in the file name
    campaign_name_parts = selected_campaign.lower().split()
    campaign_ivr_files = []
    for file in prompt_status_df['Source File'].unique():
        file_lower = file.lower()
        # Check if all significant parts of the campaign name are in the file name
        if all(part in file_lower for part in campaign_name_parts if len(part) > 2):
            campaign_ivr_files.append(file)
    
    for idx, row in campaign_prompts.iterrows():
        prompt_name = row['Prompt Name']
        
        # Get status info for this prompt, but only from the campaign's IVR files
        relevant_prompt_rows = prompt_status_df[
            (prompt_status_df['Name'] == prompt_name) & 
            (prompt_status_df['Source File'].isin(campaign_ivr_files))
        ]
        
        if not relevant_prompt_rows.empty:
            # Status depends on module reachability in the campaign's IVR
            is_in_use = any(status == '✅ In Use' for status in relevant_prompt_rows['Status'])
            status_info = '✅ In Use' if is_in_use else '❌ Not In Use'
            
            with st.expander(f"{prompt_name} ({status_info})", expanded=False):
                create_audio_player(prompt_name)
        else:
            with st.expander(f"{prompt_name} (No Status Found)", expanded=False):
                create_audio_player(prompt_name)

if __name__ == "__main__":
    main()
