import streamlit as st
import pandas as pd
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import base64
import re

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
                prompt_locations[prompt_name].append(module_data['name'])
            
            # Store module
            modules[module_id.text] = module_data
            
            # Get descendants
            for descendant in module.findall(".//singleDescendant"):
                if descendant.text:
                    modules[module_id.text]['descendants'].append(descendant.text)

    return modules, prompt_locations

def create_mermaid_diagram(modules, highlighted_modules=None):
    """Create Mermaid diagram markup for the IVR flow"""
    if highlighted_modules is None:
        highlighted_modules = []
        
    diagram = ["graph TD"]
    
    # Add nodes
    for module_id, module in modules.items():
        node_style = "style " + module_id + " fill:#ff9,stroke:#666" if module['name'] in highlighted_modules else ""
        node_label = f"{module_id}[{module['name']}]"
        if node_style:
            diagram.append(node_style)
        diagram.append(node_label)
        
        # Add connections
        for descendant in module['descendants']:
            diagram.append(f"{module_id} --> {descendant}")
    
    return "\n".join(diagram)

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

def main():
    st.set_page_config(page_title="IVR Prompt Flow Visualizer", layout="wide")
    
    st.title("IVR Prompt Flow Visualizer")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")
        audio_dir = st.text_input(
            "Audio Directory Path",
            value="./prompts",
            help="Directory containing prompt audio files (.wav format)"
        )
    
    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        mapping_file = st.file_uploader(
            "Upload Campaign-Prompt Mapping CSV",
            type=['csv'],
            help="CSV file with prompt mappings"
        )
    
    with col2:
        ivr_file = st.file_uploader(
            "Upload IVR Script",
            type=['five9ivr', 'xml'],
            help="Five9 IVR script file"
        )
    
    if mapping_file and ivr_file:
        # Load mapping data
        df = pd.read_csv(mapping_file)
        
        # Parse IVR flow
        ivr_content = ivr_file.read().decode('utf-8')
        modules, prompt_locations = parse_ivr_flow(ivr_content)
        
        # Get unique campaigns
        campaigns = df['Associated Campaigns'].str.split(',').explode().unique()
        selected_campaign = st.selectbox("Select Campaign", campaigns)
        
        # Filter prompts for selected campaign
        campaign_prompts = df[df['Associated Campaigns'].str.contains(selected_campaign, na=False)]
        
        # Display prompts with flow visualization
        st.markdown("### Campaign Prompts and Their Locations")
        
        for idx, row in campaign_prompts.iterrows():
            prompt_name = row['Prompt Name']
            with st.expander(f"{prompt_name}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if prompt_name in prompt_locations:
                        st.markdown("**Used in modules:**")
                        for location in prompt_locations[prompt_name]:
                            st.markdown(f"- {location}")
                        
                        # Create Mermaid diagram highlighting this prompt's modules
                        diagram = create_mermaid_diagram(modules, prompt_locations[prompt_name])
                        st.markdown("**Flow Diagram:**")
                        st.mermaid(diagram)
                    else:
                        st.warning("Prompt not found in IVR flow")
                
                with col2:
                    st.markdown("**Audio Preview:**")
                    audio_path = get_audio_path(prompt_name, audio_dir)
                    if os.path.exists(audio_path):
                        create_audio_player(prompt_name, audio_dir)
                    else:
                        st.warning("⚠️ Audio file not found")
                        st.text(f"Expected: {os.path.basename(audio_path)}")

if __name__ == "__main__":
    main()
