import streamlit as st
import pandas as pd
import os
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_ivr_flow(xml_content):
    """Parse IVR XML and extract flow information"""
    root = ET.fromstring(xml_content)
    
    # Dictionary to store module information
    modules = {}
    prompt_locations = {}
    
    # First pass: collect all modules
    for module in root.findall(".//modules/*"):
        module_id = module.find("moduleId")
        if module_id is not None:
            module_data = {
                'type': module.tag,
                'name': module.find("moduleName").text if module.find("moduleName") is not None else module.tag,
                'id': module_id.text,
                'descendants': [],
                'prompt': None
            }
            
            # Check for prompts
            prompt_elem = module.find(".//prompt/filePrompt/promptData/prompt")
            if prompt_elem is not None:
                prompt_name = prompt_elem.find("name").text
                module_data['prompt'] = prompt_name
                if prompt_name not in prompt_locations:
                    prompt_locations[prompt_name] = []
                prompt_locations[prompt_name].append({
                    'module_name': module_data['name'],
                    'module_type': module_data['type']
                })
            
            # Store module
            modules[module_id.text] = module_data
            
            # Get descendants
            for descendant in module.findall(".//singleDescendant"):
                if descendant.text:
                    modules[module_id.text]['descendants'].append(descendant.text)

    return modules, prompt_locations

def get_audio_path(prompt_name, audio_dir):
    """Get the path for an audio file"""
    # Try with spaces (original filename)
    filename_with_spaces = f"{prompt_name}.wav"
    path_with_spaces = os.path.join(audio_dir, filename_with_spaces)
    
    # Try with underscores
    filename_with_underscores = f"{prompt_name.replace(' ', '_')}.wav"
    path_with_underscores = os.path.join(audio_dir, filename_with_underscores)
    
    if os.path.exists(path_with_spaces):
        return path_with_spaces
    elif os.path.exists(path_with_underscores):
        return path_with_underscores
    return path_with_spaces

def create_audio_player(prompt_name, audio_dir):
    """Create an audio player for the given prompt"""
    audio_path = get_audio_path(prompt_name, audio_dir)
    if os.path.exists(audio_path):
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        return st.audio(audio_bytes, format='audio/wav')
    return None

def display_flow_path(modules, module_id, visited=None, depth=0):
    """Display the flow path in a hierarchical structure"""
    if visited is None:
        visited = set()
    
    if module_id in visited or depth > 10:  # Prevent infinite loops
        return ""
    
    visited.add(module_id)
    output = []
    
    if module_id in modules:
        module = modules[module_id]
        prefix = "  " * depth + "→ " if depth > 0 else ""
        output.append(f"{prefix}{module['name']} ({module['type']})")
        
        for descendant in module['descendants']:
            output.extend(display_flow_path(modules, descendant, visited, depth + 1))
    
    return output

def main():
    st.set_page_config(page_title="IVR Prompt Flow Visualizer", layout="wide")
    
    st.title("IVR Prompt Flow Visualizer")
    
    # Config in sidebar
    with st.sidebar:
        audio_dir = st.text_input(
            "Audio Directory",
            value="./prompts",
            help="Directory containing audio files (.wav)"
        )
        ivr_dir = st.text_input(
            "IVR Directory",
            value="./IVRs",
            help="Directory containing IVR scripts"
        )
    
    # File uploader for mapping CSV
    mapping_file = st.file_uploader(
        "Upload Campaign-Prompt Mapping CSV",
        type=['csv'],
        help="CSV file containing campaign to prompt mappings"
    )
    
    if mapping_file:
        df = pd.read_csv(mapping_file)
        # Extract unique campaigns (handle comma-separated values)
        campaigns = sorted(set(campaign.strip() 
                             for campaigns in df['Associated Campaigns'].dropna() 
                             for campaign in campaigns.split(',')))
        
        # Campaign selector
        selected_campaign = st.selectbox(
            "Select Campaign",
            campaigns,
            help="Choose a campaign to view its prompts"
        )
        
        # Find relevant IVR file
        if os.path.exists(ivr_dir):
            ivr_files = [f for f in os.listdir(ivr_dir) if f.endswith(('.five9ivr', '.xml'))]
            
            for ivr_file in ivr_files:
                ivr_path = os.path.join(ivr_dir, ivr_file)
                try:
                    with open(ivr_path, 'r', encoding='utf-8') as f:
                        ivr_content = f.read()
                        modules, prompt_locations = parse_ivr_flow(ivr_content)
                        
                        # Filter prompts for selected campaign
                        campaign_prompts = df[df['Associated Campaigns'].str.contains(selected_campaign, na=False)]
                        
                        if not campaign_prompts.empty:
                            # Display prompts with flow visualization
                            st.markdown(f"### Prompts in {selected_campaign}")
                            
                            for idx, row in campaign_prompts.iterrows():
                                prompt_name = row['Prompt Name']
                                with st.expander(f"{prompt_name}", expanded=True):
                                    cols = st.columns([3, 1])
                                    
                                    with cols[0]:
                                        if prompt_name in prompt_locations:
                                            st.markdown("#### Module Locations")
                                            for location in prompt_locations[prompt_name]:
                                                st.markdown(f"**{location['module_name']}** ({location['module_type']})")
                                                
                                                # Find and display the flow path for this module
                                                for module_id, module in modules.items():
                                                    if module['name'] == location['module_name']:
                                                        st.markdown("**Flow Path:**")
                                                        flow_paths = display_flow_path(modules, module_id)
                                                        for path in flow_paths:
                                                            st.text(path)
                                                        st.markdown("---")
                                        else:
                                            st.warning("Prompt not found in IVR flow")
                                    
                                    with cols[1]:
                                        st.markdown("**Audio Preview:**")
                                        audio_preview = create_audio_player(prompt_name, audio_dir)
                                        if not audio_preview:
                                            st.warning("⚠️ Audio file not found")
                                            st.text(f"Expected: {prompt_name}.wav")
                                    
                except Exception as e:
                    st.error(f"Error processing IVR file {ivr_file}: {str(e)}")
                    continue
        else:
            st.error(f"IVR directory {ivr_dir} not found!")

if __name__ == "__main__":
    main()
