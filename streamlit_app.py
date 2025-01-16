import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import os
from pathlib import Path

def load_campaign_data():
    """Load campaign association data."""
    try:
        return pd.read_csv("ivrcampaignassociation.csv")
    except Exception as e:
        st.error(f"Error loading campaign data: {str(e)}")
        return None

def extract_prompts(xml_content):
    """Extract prompts from XML content."""
    root = ET.fromstring(xml_content)
    prompts_list = []
    
    # First, find all announcement prompts and their enabled status
    announcement_prompts = {}
    for announcement in root.findall('.//announcements'):
        enabled = announcement.find('enabled')
        prompt = announcement.find('prompt')
        if prompt is not None and enabled is not None:
            prompt_id = prompt.find('id')
            if prompt_id is not None:
                announcement_prompts[prompt_id.text] = {
                    'enabled': enabled.text.lower() == 'true'
                }
    
    # Iterate through each module
    for module in root.findall('.//modules/*'):
        module_name = module.find('moduleName')
        
        if module_name is not None:
            module_name = module_name.text
            
            # Find prompts in different possible locations
            prompt_locations = [
                './/filePrompt/promptData/prompt',
                './/announcements/prompt',
                './/prompt/filePrompt/promptData/prompt',
                './/compoundPrompt/filePrompt/promptData/prompt',
                './/promptData/prompt'
            ]
            
            for location in prompt_locations:
                for prompt_elem in module.findall(location):
                    prompt_id = prompt_elem.find('id')
                    prompt_name = prompt_elem.find('name')
                    if prompt_id is not None and prompt_name is not None:
                        prompts_list.append({
                            'ID': prompt_id.text,
                            'Name': prompt_name.text,
                            'Module': module_name,
                            'AudioFile': f"{prompt_name.text}.wav"
                        })
    
    # Remove duplicates based on ID while keeping the first occurrence
    unique_prompts = []
    seen_ids = set()
    for prompt in prompts_list:
        if prompt['ID'] not in seen_ids:
            unique_prompts.append(prompt)
            seen_ids.add(prompt['ID'])
            
    return unique_prompts

def main():
    st.set_page_config(page_title="IVR Prompt Explorer", layout="wide")
    
    st.title("IVR Prompt Explorer")
    
    # Load campaign data
    campaign_data = load_campaign_data()
    
    if campaign_data is not None:
        # Get list of available wav files
        wav_files = [f for f in os.listdir() if f.endswith('.wav')]
        
        # Create campaign dropdown
        campaigns = campaign_data['Campaign(s)'].unique()
        selected_campaign = st.selectbox(
            "Select Campaign",
            options=campaigns,
            help="Choose a campaign to view associated IVR prompts"
        )
        
        if selected_campaign:
            # Get associated IVR(s)
            associated_ivrs = campaign_data[
                campaign_data['Campaign(s)'] == selected_campaign
            ]['IVR Associated with Campaign(s)'].tolist()
            
            for ivr_name in associated_ivrs:
                st.subheader(f"Prompts for {ivr_name}")
                
                # Get IVR content
                try:
                    ivr_path = Path('IVRs') / ivr_name
                    with open(ivr_path, 'r', encoding='utf-8') as f:
                        ivr_content = f.read()
                    
                    # Extract prompts
                    prompts = extract_prompts(ivr_content)
                    
                    if prompts:
                        # Create DataFrame
                        df = pd.DataFrame(prompts)
                        
                        # Display prompts with audio player
                        for index, row in df.iterrows():
                            # Check if the audio file exists
                            if row['AudioFile'] in wav_files:
                                with st.expander(f"{row['Name']}"):
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.write(f"Module: {row['Module']}")
                                        st.write(f"Audio File: {row['AudioFile']}")
                                    
                                    with col2:
                                        # Load and display audio
                                        try:
                                            with open(row['AudioFile'], 'rb') as audio_file:
                                                audio_bytes = audio_file.read()
                                                st.audio(audio_bytes, format='audio/wav')
                                        except Exception as e:
                                            st.write("Audio not available")
                            
                        # Show prompt count
                        st.info(f"Found {len(prompts)} unique prompts in {ivr_name}")
                    else:
                        st.warning(f"No prompts found in {ivr_name}")
                except Exception as e:
                    st.error(f"Could not load IVR file {ivr_name}: {str(e)}")

if __name__ == "__main__":
    main()
