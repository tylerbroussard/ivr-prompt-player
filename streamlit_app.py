import streamlit as st
import pandas as pd
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Optional, Tuple
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptAnalyzer:
    """Class to handle prompt analysis"""
    def __init__(self, root: ET.Element):
        self.root = root
        self.prompts = {}
        # First find all moduleIds and their connections
        self.module_connections = self._build_module_connections()
        # Then find announcement prompts
        self.announcement_prompts = self._find_announcement_prompts()
        
    def _build_module_connections(self) -> dict:
        """Build a map of module connections"""
        connections = {}
        
        for module in self.root.findall('.//modules/*'):
            module_id = module.find('moduleId')
            if module_id is None:
                continue
                
            module_id = module_id.text
            connections[module_id] = {
                'ascendants': set(),
                'descendants': set()
            }
            
            # Add ascendants
            for asc in module.findall('./ascendants'):
                if asc.text:
                    connections[module_id]['ascendants'].add(asc.text)
                    
            # Add descendants from both single and exceptional paths
            for tag in ['singleDescendant', 'exceptionalDescendant']:
                desc = module.find(f'./{tag}')
                if desc is not None and desc.text:
                    connections[module_id]['descendants'].add(desc.text)
        
        return connections

    def _is_module_disconnected(self, module_id: str) -> bool:
        """Check if a module is disconnected (all connections point to self)"""
        if module_id not in self.module_connections:
            return True
            
        connections = self.module_connections[module_id]
        all_connections = list(connections['ascendants']) + list(connections['descendants'])
        
        # If no connections or all connections point to self, module is disconnected
        return not all_connections or all(conn == module_id for conn in all_connections)
    
    def _find_parent_module(self, element: ET.Element) -> Tuple[Optional[str], Optional[str]]:
        """Find the parent module's ID and name for an element by searching up the XML tree"""
        for module in self.root.findall('.//modules/*'):
            for elem in module.iter():
                if elem == element:
                    module_id = module.find('moduleId')
                    module_name = module.find('moduleName')
                    return (
                        module_id.text if module_id is not None else None,
                        module_name.text if module_name is not None else None
                    )
        return None, None

    def _find_announcement_prompts(self) -> Dict[str, Dict]:
        """Find all announcement prompts and their enabled status"""
        announcements = {}
        
        for announcement in self.root.findall('.//announcements'):
            enabled = announcement.find('enabled')
            prompt = announcement.find('prompt')
            if prompt is not None and enabled is not None:
                prompt_id = prompt.find('id')
                if prompt_id is not None:
                    # Find the parent module information
                    module_id, module_name = self._find_parent_module(announcement)
                    
                    announcements[prompt_id.text] = {
                        'enabled': enabled.text.lower() == 'true',
                        'module_id': module_id,
                        'module_name': module_name
                    }
        
        return announcements
    
    def process_module(self, module: ET.Element):
        """Process prompts in a module"""
        module_id = module.find('moduleId')
        module_name = module.find('moduleName')
        
        if module_name is not None and module_id is not None:
            module_name = module_name.text
            module_id = module_id.text
            is_disconnected = self._is_module_disconnected(module_id)
            
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
                    self._add_prompt(prompt_elem, module_name, module_id, is_disconnected)
    
    def _add_prompt(self, prompt_elem: ET.Element, module_name: str, module_id: str, is_disconnected: bool):
        """Add a prompt to the prompts dictionary"""
        prompt_id = prompt_elem.find('id')
        prompt_name = prompt_elem.find('name')
        
        if prompt_id is not None and prompt_name is not None:
            prompt_id_text = prompt_id.text
            
            # Check if this is an announcement prompt
            announcement_info = self.announcement_prompts.get(prompt_id_text)
            is_announcement = announcement_info is not None
            
            if is_announcement:
                # For announcement prompts, use enabled status from announcements section
                enabled = announcement_info['enabled']
                status = '✅ Enabled' if enabled else '❌ Disabled'
                prompt_type = 'Announcement'
                # Use the module name from where the announcement was defined
                if announcement_info['module_name']:
                    module_name = announcement_info['module_name']
            else:
                # For regular prompts, use module connectivity
                enabled = not is_disconnected
                status = '✅ In Use' if enabled else '❌ Not In Use'
                prompt_type = 'Play'
            
            self.prompts[prompt_id_text] = {
                'ID': prompt_id_text,
                'Name': prompt_name.text,
                'Module': module_name,
                'Type': prompt_type,
                'Status': status,
                'Source File': ''
            }

def process_ivr_file(file_path: str) -> pd.DataFrame:
    """Process a single IVR file and return a DataFrame of prompts"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Analyze prompts
        analyzer = PromptAnalyzer(root)
        
        # Process each module
        for module in root.findall('.//modules/*'):
            analyzer.process_module(module)
        
        # Get results and add source file
        results = pd.DataFrame(analyzer.prompts.values())
        if not results.empty:
            results['Source File'] = os.path.basename(file_path)
        return results
        
    except ET.ParseError as e:
        logger.error(f"Error parsing {file_path}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
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

def main():
    st.set_page_config(page_title="IVR Prompt Player", layout="wide")
    st.title("IVR Prompt Player")
    
    # Load campaign mapping data
    mapping_df = load_mapping_file()
    if mapping_df is None:
        return
    
    # Process IVR files from the repository
    ivr_dir = "./IVRs"
    prompt_status_df = pd.DataFrame()
    
    try:
        ivr_files = list(Path(ivr_dir).glob('*.five9ivr')) + list(Path(ivr_dir).glob('*.xml'))
        
        if ivr_files:
            for file_path in ivr_files:
                df = process_ivr_file(str(file_path))
                if not df.empty:
                    prompt_status_df = pd.concat([prompt_status_df, df], ignore_index=True)
        
        if prompt_status_df.empty:
            st.warning("No IVR files found or processed successfully")
            return
            
    except Exception as e:
        st.error(f"Error processing IVR files: {str(e)}")
        logger.error("Error in IVR processing", exc_info=True)
        return

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
        st.metric("Total Prompts", len(campaign_prompts))
    with col2:
        inactive_count = len(prompt_status_df[
            prompt_status_df['Status'].isin(['❌ Not In Use', '❌ Disabled'])
        ])
        st.metric("Inactive Prompts", inactive_count)
    
    # Display campaign prompts
    st.markdown("### Campaign Prompts")
    
    # Get the IVR files for the selected campaign
    campaign_name_parts = selected_campaign.lower().split()
    relevant_ivr_files = [
        file for file in prompt_status_df['Source File'].unique()
        if all(part in file.lower() for part in campaign_name_parts if len(part) > 2)
    ]
    
    for _, row in campaign_prompts.iterrows():
        prompt_name = row['Prompt Name']
        
        # Get status info for this prompt from relevant IVR files
        prompt_status = prompt_status_df[
            (prompt_status_df['Name'] == prompt_name) & 
            (prompt_status_df['Source File'].isin(relevant_ivr_files))
        ]
        
        if not prompt_status.empty:
            # Get the status - prefer statuses in order: In Use > Enabled > Disabled > Not In Use
            status_order = {
                '✅ In Use': 0,
                '✅ Enabled': 1,
                '❌ Disabled': 2,
                '❌ Not In Use': 3
            }
            
            statuses = prompt_status['Status'].unique()
            status = min(statuses, key=lambda x: status_order.get(x, 4))
            
            with st.expander(f"{prompt_name} ({status})", expanded=False):
                create_audio_player(prompt_name)
        else:
            with st.expander(f"{prompt_name} (No Status Found)", expanded=False):
                create_audio_player(prompt_name)

if __name__ == "__main__":
    main()
