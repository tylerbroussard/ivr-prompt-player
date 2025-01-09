import streamlit as st
import pandas as pd
import os
from pathlib import Path

def load_mapping_file(file):
    """Load and process the campaign mapping CSV file"""
    df = pd.read_csv(file)
    # Split campaigns if they're in a comma-separated format
    df['Associated Campaigns'] = df['Associated Campaigns'].str.split(',')
    return df

def get_unique_campaigns(df):
    """Extract unique campaigns from the dataframe"""
    # Flatten the list of campaigns and remove duplicates
    all_campaigns = []
    for campaigns in df['Associated Campaigns']:
        if isinstance(campaigns, list):
            all_campaigns.extend(campaigns)
    return sorted(list(set(all_campaigns)))

def get_audio_path(prompt_name, audio_dir):
    """Get the path for an audio file based on the prompt name"""
    filename = f"{prompt_name.replace(' ', '_')}.wav"
    return os.path.join(audio_dir, filename)

def create_audio_player(prompt_name, audio_dir):
    """Create an audio player for the given prompt"""
    audio_path = get_audio_path(prompt_name, audio_dir)
    if os.path.exists(audio_path):
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        return st.audio(audio_bytes, format='audio/wav')
    return None

def main():
    st.set_page_config(
        page_title="Campaign Prompt Player",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Campaign Prompt Player")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Settings")
        audio_dir = st.text_input(
            "Audio Directory Path",
            value="./prompts",
            help="Directory containing the prompt audio files (.wav format)"
        )
        
        # Audio file checker
        if st.button("Check Audio Directory"):
            if os.path.exists(audio_dir):
                files = os.listdir(audio_dir)
                wav_files = [f for f in files if f.endswith('.wav')]
                st.success(f"Found {len(wav_files)} .wav files")
            else:
                st.error("Directory not found!")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        mapping_file = st.file_uploader(
            "Upload Campaign-Prompt Mapping CSV",
            type=['csv'],
            help="CSV file with 'Prompt Name' and 'Associated Campaigns' columns"
        )

    if mapping_file:
        # Load the mapping data
        df = load_mapping_file(mapping_file)
        
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
        st.metric("Prompts in Selected Campaign", len(campaign_prompts))
        
        # Display prompts with audio players
        st.markdown("### Campaign Prompts")
        
        for idx, row in campaign_prompts.iterrows():
            prompt_name = row['Prompt Name']
            
            with st.expander(prompt_name, expanded=True):
                cols = st.columns([3, 1])
                
                with cols[0]:
                    st.text("Campaigns: " + ", ".join(row['Associated Campaigns']))
                
                with cols[1]:
                    audio_path = get_audio_path(prompt_name, audio_dir)
                    if os.path.exists(audio_path):
                        create_audio_player(prompt_name, audio_dir)
                    else:
                        st.warning("⚠️ Audio not found")
                        st.text(f"Expected: {os.path.basename(audio_path)}")

if __name__ == "__main__":
    main()