import streamlit as st
import pandas as pd
import os
from pathlib import Path
import xml.etree.ElementTree as ET

def extract_prompts_from_xml(file_path):
    """Extract prompts and their status from XML content"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
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
            
            # Check if module is disconnected
            is_disconnected = False
            module_id = None
            if module.find('moduleId') is not None:
                module_id = module.find('moduleId').text
                # If all connection IDs are the same as the module ID, it's disconnected
                all_connections = []
                for tag in ['ascendants', 'exceptionalDescendant', 'singleDescendant']:
                    connections = module.findall(f'./{tag}')
                    all_connections.extend([conn.text for conn in connections])
                
                if all_connections and all(conn == module_id for conn in all_connections):
                    is_disconnected = True
            
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
                            # Check if this prompt has announcement settings
                            enabled = announcement_prompts.get(prompt_id.text, {}).get('enabled', None)
                            # If not an announcement prompt, mark as In Use/Not In Use
                            if enabled is None:
                                enabled = not is_disconnected
                            prompts_list.append({
                                'ID': prompt_id.text,
                                'Name': prompt_name.text,
                                'Module': module_name,
                                'Type': 'Announcement' if prompt_id.text in announcement_prompts else 'Play',
                                'Status': ('✅ Enabled' if enabled else '❌ Disabled') if prompt_id.text in announcement_prompts 
                                         else ('✅ In Use' if enabled else '❌ Not In Use')
                            })
        
        return pd.DataFrame(prompts_list)
    except Exception as e:
        st.error(f"Error processing XML file {xml_file}: {str(e)}")
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
    
    # Read IVR files from repository
    ivr_dir = "./IVRs"  # Updated path
    xml_data = pd.DataFrame()
    
    try:
        # Get all XML and five9ivr files from the directory
        ivr_files = []
        for ext in ['*.xml', '*.five9ivr']:
            ivr_files.extend(Path(ivr_dir).glob(ext))
        
        if ivr_files:
            for file_path in ivr_files:
                df = extract_prompts_from_xml(file_path)
                if df is not None:
                    df['Source File'] = file_path.name
                    xml_data = pd.concat([xml_data, df], ignore_index=True)
            
            if not xml_data.empty:
                # Group by prompt ID to combine instances and determine final status
                final_data = []
                for name, group in xml_data.groupby(['ID', 'Name']):
                    prompt_id, prompt_name = name
                    is_announcement = (group['Type'] == 'Announcement').any()
                    
                    # If it's an announcement and enabled anywhere, it's enabled
                    if is_announcement:
                        enabled = group['Enabled'].any()
                        status = '✅ Enabled' if enabled else '❌ Disabled'
                    else:
                        enabled = group['Enabled'].any()
                        status = '✅ In Use' if enabled else '❌ Not In Use'
                    
                    final_data.append({
                        'ID': prompt_id,
                        'Name': prompt_name,
                        'Module': ', '.join(sorted(group['Module'].unique())),
                        'Type': 'Announcement' if is_announcement else 'Play',
                        'Status': status,
                        'Source File': ', '.join(sorted(group['Source File'].unique()))
                    })
                
                final_df = pd.DataFrame(final_data)
                
                st.write("### Prompt Status from IVR Files")
                st.dataframe(
                    final_df[['Name', 'ID', 'Module', 'Type', 'Status', 'Source File']].sort_values('Name'),
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning("No IVR files found in the repository. Please ensure IVR files are in the ./ivr directory.")
    except Exception as e:
        st.error(f"Error reading IVR files: {str(e)}")
        
        if not xml_data.empty:
            st.write("### Prompt Status from IVR Files")
            st.dataframe(
                xml_data[['Name', 'ID', 'Module', 'Type', 'Status', 'Source File']],
                hide_index=True,
                use_container_width=True
            )
    
    # Load mapping data
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
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prompts in Selected Campaign", len(campaign_prompts))
        with col2:
            if not xml_data.empty:
                # Count prompts that are not in use or disabled
                inactive_prompts = len(xml_data[xml_data['Status'].str.contains('❌')])
                st.metric("Inactive Prompts", inactive_prompts)
        
        # Display associated campaigns
        st.markdown("### Campaign Details")
        st.markdown(f"**Selected Campaign:** {selected_campaign}")
        
        # Display prompts with audio players and status
        st.markdown("### Campaign Prompts")
        
        for idx, row in campaign_prompts.iterrows():
            prompt_name = row['Prompt Name']
            status_info = ""
            if not xml_data.empty:
                prompt_status = xml_data[xml_data['Name'] == prompt_name]
                if not prompt_status.empty:
                    status_info = f" ({prompt_status.iloc[0]['Status']})"
            
            with st.expander(f"{prompt_name}{status_info}", expanded=True):
                audio_path = get_audio_path(prompt_name)
                if os.path.exists(audio_path):
                    create_audio_player(prompt_name)
                else:
                    st.warning("⚠️ Audio not found")
                    st.text(f"Looking for: {os.path.basename(audio_path)}")

if __name__ == "__main__":
    main()
