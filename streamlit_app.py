import streamlit as st
import pandas as pd
import os
from pathlib import Path

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
    # Flatten the list of campaigns and remove duplicates
    all_campaigns = []
    for campaigns in df['Associated Campaigns']:
        if isinstance(campaigns, list):
            all_campaigns.extend(campaigns)
    return sorted(list(set(campaign.strip() for campaign in all_campaigns)))

def get_audio_path(prompt_name):
    """Get the path for an audio file based on the prompt name"""
    audio_dir = "./prompts"  # Hardcoded directory path
    
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

    # Load mapping data directly from file
    df = load_mapping_file()
    
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
        st.metric("Prompts in Selected Campaign", len(campaign_prompts))
        
        # Display associated campaigns
        st.markdown("### Campaign Details")
        st.markdown(f"**Selected Campaign:** {selected_campaign}")
        
        # Display prompts with audio players
        st.markdown("### Campaign Prompts")
        
        for idx, row in campaign_prompts.iterrows():
            prompt_name = row['Prompt Name']
            with st.expander(prompt_name, expanded=True):
                audio_path = get_audio_path(prompt_name)
                if os.path.exists(audio_path):
                    create_audio_player(prompt_name)
                else:
                    st.warning("⚠️ Audio not found")
                    st.text(f"Looking for: {os.path.basename(audio_path)}")

if __name__ == "__main__":
    main()
