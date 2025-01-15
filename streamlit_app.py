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
        self._find_incoming_call(root)

    def _build_graph(self, root: ET.Element) -> Dict[str, Set[str]]:
        """Build directed graph of module connections"""
        graph = defaultdict(set)
        
        # Initialize all modules
        for module in root.findall('.//modules/*'):
            module_id = module.find('moduleId')
            if module_id is not None:
                graph[module_id.text] = set()
        
        # Build connections
        for module in root.findall('.//modules/*'):
            module_id = module.find('moduleId')
            if module_id is not None:
                current_id = module_id.text
                
                # Add branch descendants from branches/entry/value/desc
                for branch in module.findall('.//branches/entry/value/desc'):
                    if branch is not None and branch.text:
                        graph[current_id].add(branch.text)
                
                # Add direct descendants
                for tag in ['singleDescendant', 'exceptionalDescendant']:
                    for descendant in module.findall(f'./{tag}'):
                        if descendant is not None and descendant.text:
                            graph[current_id].add(descendant.text)
                            
                # Add ascendants for reverse lookup
                for ascendant in module.findall('.//ascendants'):
                    if ascendant is not None and ascendant.text:
                        graph[ascendant.text].add(current_id)
        
        return graph

    def _find_incoming_call(self, root: ET.Element) -> None:
        """Find IncomingCall module and compute reachable modules"""
        incoming_call = root.find('.//modules/incomingCall/moduleId')
        if incoming_call is not None:
            self._compute_reachable_modules(incoming_call.text)
        else:
            logger.warning("No IncomingCall module found")

    def _compute_reachable_modules(self, start_module: str) -> None:
        """Compute all reachable modules using DFS"""
        stack = [start_module]
        while stack:
            current = stack.pop()
            if current not in self.reachable_modules:
                self.reachable_modules.add(current)
                for neighbor in self.graph[current]:
                    if neighbor not in self.reachable_modules:
                        stack.append(neighbor)

    def is_module_reachable(self, module_id: str) -> bool:
        """Check if a module is reachable from IncomingCall"""
        return module_id in self.reachable_modules

class PromptAnalyzer:
    """Class to handle prompt analysis"""
    def __init__(self, module_graph: ModuleGraph):
        self.module_graph = module_graph
        self.prompts = {}

    def process_module(self, module: ET.Element) -> None:
        """Process prompts in a module"""
        module_id = module.find('moduleId')
        module_name = module.find('moduleName')
        
        if module_id is None or module_name is None:
            return
            
        is_reachable = self.module_graph.is_module_reachable(module_id.text)
        
        # Process menu-specific prompts with special handling for recoEvents
        if module.tag == 'menu':
            self._process_menu_prompts(module, module_name.text, module_id.text, is_reachable)
        else:
            self._process_standard_prompts(module, module_name.text, module_id.text, is_reachable)

    def _process_menu_prompts(self, module: ET.Element, module_name: str, module_id: str, 
                            is_reachable: bool) -> None:
        """Process prompts specific to menu modules"""
        # If module is not reachable, all prompts are not in use
        if not is_reachable:
            for prompt_elem in module.findall('.//promptData/prompt'):
                self._add_prompt(prompt_elem, module_name, module_id, False)
            return
            
        # Process main menu prompts
        for prompt in module.findall('.//prompts/prompt/filePrompt/promptData/prompt'):
            self._add_prompt(prompt, module_name, module_id, True)
            
        # Track seen prompts and their recoEvents context
        prompt_contexts = {}
        
        # First pass: collect all prompts and their contexts
        for reco_event in module.findall('.//recoEvents'):
            event_type = reco_event.find('event')
            event_count = reco_event.find('count')
            action = reco_event.find('action')
            
            for prompt in reco_event.findall('.//promptData/prompt'):
                prompt_id = prompt.find('id')
                if prompt_id is not None:
                    context = {
                        'event': event_type.text if event_type is not None else 'unknown',
                        'count': int(event_count.text) if event_count is not None else 0,
                        'action': action.text if action is not None else 'unknown'
                    }
                    if prompt_id.text not in prompt_contexts:
                        prompt_contexts[prompt_id.text] = []
                    prompt_contexts[prompt_id.text].append(context)
        
        # Second pass: process prompts with their full context
        for prompt_id, contexts in prompt_contexts.items():
            # Find the prompt element
            prompt_elem = module.find(f'.//promptData/prompt[id="{prompt_id}"]')
            if prompt_elem is not None:
                # A prompt is considered "in use" if:
                # 1. It's part of the main flow (count == 1 and action == 'REPROMPT')
                # 2. It's a transfer prompt (regardless of count)
                # 3. It's an invalid selection prompt (regardless of count)
                prompt_name = prompt_elem.find('name')
                is_active = any(
                    (context['count'] == 1 and context['action'] == 'REPROMPT') or
                    (prompt_name is not None and ('Transfer Prompt' in prompt_name.text or 'Invalid' in prompt_name.text))
                    for context in contexts
                )
                self._add_prompt(prompt_elem, module_name, module_id, is_active)

    def _process_standard_prompts(self, module: ET.Element, module_name: str, module_id: str, 
                                is_reachable: bool) -> None:
        """Process prompts in standard modules"""
        for prompt in module.findall('.//prompt/filePrompt/promptData/prompt'):
            self._add_prompt(prompt, module_name, module_id, is_reachable)

    def _add_prompt(self, prompt_elem: ET.Element, module_name: str, module_id: str, 
                   is_active: bool) -> None:
        """Add a prompt to the prompts dictionary"""
        prompt_id = prompt_elem.find('id')
        prompt_name = prompt_elem.find('name')
        
        if prompt_id is not None and prompt_name is not None:
            key = (prompt_id.text, prompt_name.text)
            status = '✅ In Use' if is_active else '❌ Not In Use'
            
            # Only update if not exists or if new status is "in use"
            if key not in self.prompts or status == '✅ In Use':
                self.prompts[key] = {
                    'ID': prompt_id.text,
                    'Name': prompt_name.text,
                    'Module': module_name,
                    'ModuleID': module_id,
                    'Type': 'Play',
                    'Status': status
                }

    def get_results(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        return pd.DataFrame(list(self.prompts.values()))

def analyze_ivr_file(file_path: str) -> Optional[pd.DataFrame]:
    """Analyze an IVR file and return prompt information"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Create module graph
        module_graph = ModuleGraph(root)
        
        # Create prompt analyzer
        analyzer = PromptAnalyzer(module_graph)
        
        # Process all modules
        for module in root.findall('.//modules/*'):
            analyzer.process_module(module)
        
        # Get results
        results = analyzer.get_results()
        
        # Add source file information
        results['Source File'] = Path(file_path).name
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

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
    st.set_page_config(page_title="Campaign Prompt Player", layout="wide")
    st.title("Campaign Prompt Player")
    
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
                df = analyze_ivr_file(str(file_path))
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
            inactive_count = len(prompt_status_df[prompt_status_df['Status'].str.contains('❌')])
            st.metric("Inactive Prompts", inactive_count)
    
    # Display campaign details
    st.markdown("### Campaign Details")
    st.markdown(f"**Selected Campaign:** {selected_campaign}")
    
    # Display prompts with audio players and status
    st.markdown("### Campaign Prompts")
    
    for idx, row in campaign_prompts.iterrows():
        prompt_name = row['Prompt Name']
        
        # Get status from prompt_status_df
        status_info = "Status Unknown"
        if not prompt_status_df.empty:
            prompt_status = prompt_status_df[prompt_status_df['Name'] == prompt_name]
            if not prompt_status.empty:
                status_info = prompt_status.iloc[0]['Status']
        
        with st.expander(f"{prompt_name} ({status_info})", expanded=True):
            create_audio_player(prompt_name)

if __name__ == "__main__":
    main()
