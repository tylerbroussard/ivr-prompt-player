import streamlit as st
import pandas as pd
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Set

def build_module_graph(root: ET.Element) -> Dict[str, List[str]]:
    """Build a graph of module connections."""
    graph = {}
    
    # First pass: initialize all modules
    for module in root.findall('.//modules/*'):
        module_id = module.find('moduleId')
        if module_id is not None:
            graph[module_id.text] = set()  # Using set to avoid duplicates
            
    # Second pass: build connections
    for module in root.findall('.//modules/*'):
        module_id = module.find('moduleId')
        if module_id is not None:
            module_id = module_id.text
            
            # Add descendants from branches in case modules
            branches = module.findall('.//branches/entry/value/desc')
            for branch in branches:
                if branch is not None and branch.text:
                    graph[module_id].add(branch.text)
            
            # Add direct descendants
            for tag in ['singleDescendant', 'exceptionalDescendant']:
                for descendant in module.findall(f'./{tag}'):
                    if descendant.text:
                        graph[module_id].add(descendant.text)
    
    # Convert sets to lists for final output
    return {k: list(v) for k, v in graph.items()}

def find_reachable_modules(graph: Dict[str, List[str]], start_module: str) -> Set[str]:
    """Find all modules reachable from the start module using DFS."""
    reachable = set()
    stack = [start_module]
    
    while stack:
        current = stack.pop()
        if current not in reachable:
            reachable.add(current)
            if current in graph:  # Check if the module exists in the graph
                for neighbor in graph[current]:
                    if neighbor not in reachable:
                        stack.append(neighbor)
    
    return reachable

def is_module_disconnected(module: ET.Element, reachable_modules: Set[str]) -> bool:
    """Determine if a module is disconnected by checking if it's reachable from IncomingCall."""
    module_id = module.find('moduleId')
    if module_id is None:
        return True
    
    return module_id.text not in reachable_modules

def find_all_prompts_in_module(module: ET.Element, module_name: str, module_id: str, is_disconnected: bool) -> List[Dict]:
    """Extract all prompts from a module, including those in recoEvents"""
    prompts = []
    
    # Find all prompt elements, regardless of their location in the XML
    for elem in module.findall('.//'):
        if elem.tag == 'prompt' and elem.find('id') is not None and elem.find('name') is not None:
            prompt_id = elem.find('id')
            prompt_name = elem.find('name')
            
            # Check if this is an announcement prompt
            is_announcement = False
            parent = elem.getparent()
            while parent is not None:
                if parent.tag == 'announcements':
                    is_announcement = True
                    enabled_elem = parent.find('enabled')
                    enabled = enabled_elem is not None and enabled_elem.text.lower() == 'true'
                    break
                parent = parent.getparent()
            
            if is_announcement:
                if is_disconnected:
                    enabled = False
            else:
                enabled = not is_disconnected
                
            prompts.append({
                'ID': prompt_id.text,
                'Name': prompt_name.text,
                'Module': module_name,
                'ModuleID': module_id,
                'Type': 'Announcement' if is_announcement else 'Play',
                'Status': ('✅ Enabled' if enabled else '❌ Disabled') if is_announcement 
                         else ('✅ In Use' if enabled else '❌ Not In Use')
            })
    
    return prompts

def extract_prompts_from_xml(file_path):
    """Extract prompts and their status from XML content"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        prompts_list = []
        
        # Find IncomingCall module and build reachability graph
        incoming_call = root.find('.//modules/incomingCall/moduleId')
        if incoming_call is not None:
            graph = build_module_graph(root)
            reachable_modules = find_reachable_modules(graph, incoming_call.text)
        else:
            st.warning(f"No incoming call module found in {file_path}")
            reachable_modules = set()

        # Process all modules to find prompts
        for module in root.findall('.//modules/*'):
            # Check if this module is reachable from IncomingCall
            is_disconnected = is_module_disconnected(module, reachable_modules)
            module_id = module.find('moduleId')
            module_name = module.find('moduleName')
            
            if module_name is not None and module_id is not None:
                module_name = module_name.text
                module_id = module_id.text
                
# Process standard prompts
                prompt_locations = [
                    # Main prompts
                    './/prompt/filePrompt/promptData/prompt',
                    # Menu prompts
                    './/prompts/prompt/filePrompt/promptData/prompt',
                    # Compound prompts
                    './/compoundPrompt/filePrompt/promptData/prompt',
                    # Announcement prompts
                    './/announcements/prompt'
                ]
                
                # Process menu reco events separately
                if module.tag == 'menu':
                    for reco_event in module.findall('.//recoEvents'):
                        event_count = reco_event.find('count')
                        event_action = reco_event.find('action')
                        
                        # Process all prompts within this reco event
                        for prompt_elem in reco_event.findall('.//promptData/prompt'):
                            prompt_id = prompt_elem.find('id')
                            prompt_name = prompt_elem.find('name')
                            
                            if prompt_id is not None and prompt_name is not None:
                                prompts_list.append({
                                    'ID': prompt_id.text,
                                    'Name': prompt_name.text,
                                    'Module': module_name,
                                    'ModuleID': module_id,
                                    'Type': 'Play',
                                    'Status': '❌ Not In Use' if is_disconnected else '✅ In Use'
                                })
                
                for location in prompt_locations:
                    for prompt_elem in module.findall(location):
                        prompt_id = prompt_elem.find('id')
                        prompt_name = prompt_elem.find('name')
                        if prompt_id is not None and prompt_name is not None:
                            # For announcement prompts, check enabled status
                            is_announcement = module.tag == 'skillTransfer' and 'announcements' in location
                            if is_announcement:
                                enabled_elem = prompt_elem.find('../enabled')
                                enabled = enabled_elem is not None and enabled_elem.text.lower() == 'true'
                                # Even if enabled, check if the module is reachable
                                if is_disconnected:
                                    enabled = False
                            else:
                                enabled = not is_disconnected
                            
                            prompts_list.append({
                                'ID': prompt_id.text,
                                'Name': prompt_name.text,
                                'Module': module_name,
                                'ModuleID': module_id,
                                'Type': 'Announcement' if is_announcement else 'Play',
                                'Status': ('✅ Enabled' if enabled else '❌ Disabled') if is_announcement 
                                         else ('✅ In Use' if enabled else '❌ Not In Use')
                            })
        
        return pd.DataFrame(prompts_list)
    except Exception as e:
        st.error(f"Error processing XML file {file_path}: {str(e)}")
        return None

def load_mapping_file():
    """Load and process the campaign mapping CSV file from local directory"""
    try:
        df = pd.read_csv("prompt_campaign_mapping.csv")
        # Split campaigns if they're in a comma-separated format
        df['Associated Campaigns'] = df['Associated Campaigns'].str.split(',')
        return df
    except FileNotFoundError:
        st.error("prompt_campaign_mapping.csv not found in the current directory")
        return None
    except Exception as e:
        st.error(f"Error reading mapping file: {str(e)}")
        return None

def get_unique_campaigns(df):
    """Extract unique campaigns from the dataframe"""
    if df is None:
        return []
    all_campaigns = []
    for campaigns in df['Associated Campaigns']:
        if isinstance(campaigns, list):
            all_campaigns.extend(campaigns)
    return sorted(list(set(campaign.strip() for campaign in all_campaigns)))

def get_audio_path(prompt_name):
    """Get the path for an audio file based on the prompt name"""
    audio_dir = "./prompts"
    filename_with_spaces = f"{prompt_name}.wav"
    path_with_spaces = os.path.join(audio_dir, filename_with_spaces)
    filename_with_underscores = f"{prompt_name.replace(' ', '_')}.wav"
    path_with_underscores = os.path.join(audio_dir, filename_with_underscores)
    
    if os.path.exists(path_with_spaces):
        return path_with_spaces
    elif os.path.exists(path_with_underscores):
        return path_with_underscores
    return path_with_spaces

def create_audio_player(prompt_name):
    """Create an audio player for the given prompt"""
    audio_path = get_audio_path(prompt_name)
    if os.path.exists(audio_path):
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        return st.audio(audio_bytes, format='audio/wav')
    return None

def main():
    st.set_page_config(
        page_title="Campaign Prompt Player",
        layout="wide"
    )
    
    st.title("Campaign Prompt Player")
    
    # Load mapping data
    df = load_mapping_file()
    
    # Read IVR files from repository but don't display the table
    ivr_dir = "./IVRs"
    xml_data = pd.DataFrame()
    
    try:
        # Get all XML and five9ivr files from the directory
        ivr_files = []
        for ext in ['*.xml', '*.five9ivr']:
            ivr_files.extend(Path(ivr_dir).glob(ext))
        
        if ivr_files:
            for file_path in ivr_files:
                df_xml = extract_prompts_from_xml(file_path)
                if df_xml is not None:
                    df_xml['Source File'] = file_path.name
                    xml_data = pd.concat([xml_data, df_xml], ignore_index=True)
            
            if not xml_data.empty:
                # Group by prompt ID to combine any duplicates
                final_data = []
                for name, group in xml_data.groupby(['ID', 'Name']):
                    prompt_id, prompt_name = name
                    is_announcement = (group['Type'] == 'Announcement').any()
                    has_enabled = any('✅' in status for status in group['Status'])
                    status = ('✅ Enabled' if has_enabled else '❌ Disabled') if is_announcement else ('✅ In Use' if has_enabled else '❌ Not In Use')
                    
                    final_data.append({
                        'ID': prompt_id,
                        'Name': prompt_name,
                        'Type': 'Announcement' if is_announcement else 'Play',
                        'Status': status,
                    })
                    
                xml_data = pd.DataFrame(final_data)
        
        if ivr_files and not xml_data.empty:
            inactive_prompts_count = len(xml_data[xml_data['Status'].str.contains('❌')])
        else:
            inactive_prompts_count = 0
            st.warning("No IVR files found in the repository. Please ensure IVR files are in the ./IVRs directory.")
    except Exception as e:
        st.error(f"Error reading IVR files: {str(e)}")
        inactive_prompts_count = 0
    
    if df is not None:
        # Get unique campaigns
        campaigns = get_unique_campaigns(df)
        
        # Campaign selector
        selected_campaign = st.selectbox(
            "Select Campaign",
            options=campaigns,
            help="Choose a campaign to view its associated prompts"
        )
        
        # Filter prompts for selected campaign
        campaign_prompts = df[df['Associated Campaigns'].apply(
            lambda x: selected_campaign in x if isinstance(x, list) else False
        )]
        
        # Display statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prompts in Selected Campaign", len(campaign_prompts))
        with col2:
            st.metric("Inactive Prompts", inactive_prompts_count)
        
        # Display associated campaigns
        st.markdown("### Campaign Details")
        st.markdown(f"**Selected Campaign:** {selected_campaign}")
        
        # Display prompts with audio players and status
        st.markdown("### Campaign Prompts")
        
        for idx, row in campaign_prompts.iterrows():
            prompt_name = row['Prompt Name']
            
            # Get status from xml_data
            status_info = "Status Unknown"
            if not xml_data.empty:
                prompt_status = xml_data[xml_data['Name'] == prompt_name]
                if not prompt_status.empty:
                    status_info = prompt_status.iloc[0]['Status']
                    status_type = prompt_status.iloc[0]['Type']
            
            with st.expander(f"{prompt_name} ({status_info})", expanded=True):
                audio_path = get_audio_path(prompt_name)
                if os.path.exists(audio_path):
                    create_audio_player(prompt_name)
                else:
                    st.warning("⚠️ Audio not found")
                    st.text(f"Looking for: {os.path.basename(audio_path)}")

if __name__ == "__main__":
    main()
