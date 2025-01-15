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

@dataclass
class Prompt:
    """Data class to store prompt information"""
    id: str
    name: str
    module: str
    module_id: str
    type: str
    status: str
    source_location: str  # Track where in the XML the prompt was found
    occurrence_count: int = 1  # Track how many times this prompt appears

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
                
                # Add branch descendants
                for branch in module.findall('.//branches/entry/value/desc'):
                    if branch is not None and branch.text:
                        graph[current_id].add(branch.text)
                
                # Add direct descendants
                for descendant in module.findall('./singleDescendant'):
                    if descendant is not None and descendant.text:
                        graph[current_id].add(descendant.text)
                
                # Add exceptional descendants
                for descendant in module.findall('./exceptionalDescendant'):
                    if descendant is not None and descendant.text:
                        graph[current_id].add(descendant.text)
        
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
                stack.extend(self.graph[current])

    def is_module_reachable(self, module_id: str) -> bool:
        """Check if a module is reachable from IncomingCall"""
        return module_id in self.reachable_modules

class PromptAnalyzer:
    """Class to handle prompt analysis"""
    def __init__(self, module_graph: ModuleGraph):
        self.module_graph = module_graph
        self.prompts = {}  # Dictionary to store unique prompts
        self.prompt_locations = defaultdict(list)  # Track where each prompt appears

    def process_module(self, module: ET.Element) -> None:
        """Process prompts in a module"""
        module_id = module.find('moduleId')
        module_name = module.find('moduleName')
        
        if module_id is None or module_name is None:
            logger.warning(f"Module missing ID or name: {ET.tostring(module, encoding='unicode')[:100]}")
            return
            
        is_reachable = self.module_graph.is_module_reachable(module_id.text)
        
        # Process different types of prompts based on module type
        if module.tag == 'menu':
            self._process_menu_prompts(module, module_name.text, module_id.text, is_reachable)
        else:
            self._process_standard_prompts(module, module_name.text, module_id.text, is_reachable)

    def _process_menu_prompts(self, module: ET.Element, module_name: str, module_id: str, 
                            is_reachable: bool) -> None:
        """Process prompts specific to menu modules"""
        # Track seen prompts within this menu to handle duplicates
        seen_in_menu = set()
        
        # Process main menu prompts
        for prompt in module.findall('.//prompts/prompt/filePrompt/promptData/prompt'):
            self._add_prompt(prompt, module_name, module_id, is_reachable, 'Menu', seen_in_menu)
        
        # Process recoEvents prompts
        for reco_event in module.findall('.//recoEvents'):
            event_type = reco_event.find('event')
            event_count = reco_event.find('count')
            location = f"recoEvents/{event_type.text if event_type is not None else 'unknown'}"
            
            for prompt in reco_event.findall('.//promptData/prompt'):
                self._add_prompt(prompt, module_name, module_id, is_reachable, location, seen_in_menu)

    def _process_standard_prompts(self, module: ET.Element, module_name: str, module_id: str, 
                                is_reachable: bool) -> None:
        """Process prompts in non-menu modules"""
        seen_in_module = set()
        
        # Process all prompt locations
        for prompt in module.findall('.//prompt/filePrompt/promptData/prompt'):
            self._add_prompt(prompt, module_name, module_id, is_reachable, 'Standard', seen_in_module)
            
        # Process announcement prompts
        for prompt in module.findall('.//announcements//prompt'):
            self._add_prompt(prompt, module_name, module_id, is_reachable, 'Announcement', seen_in_module)

    def _add_prompt(self, prompt_elem: ET.Element, module_name: str, module_id: str, 
                   is_reachable: bool, location: str, seen_prompts: Set[str]) -> None:
        """Add or update a prompt in the prompts dictionary"""
        prompt_id = prompt_elem.find('id')
        prompt_name = prompt_elem.find('name')
        
        if prompt_id is None or prompt_name is None:
            logger.warning(f"Prompt missing ID or name in module {module_name}")
            return
            
        prompt_key = (prompt_id.text, prompt_name.text)
        
        # Check if we've seen this prompt in this module
        if prompt_key in seen_prompts:
            self.prompts[prompt_key].occurrence_count += 1
            return
            
        seen_prompts.add(prompt_key)
        
        # Determine prompt status
        status = '✅ In Use' if is_reachable else '❌ Not In Use'
        
        if prompt_key in self.prompts:
            # Update existing prompt if needed
            existing_prompt = self.prompts[prompt_key]
            existing_prompt.occurrence_count += 1
            if is_reachable and existing_prompt.status == '❌ Not In Use':
                existing_prompt.status = '✅ In Use'
        else:
            # Create new prompt
            self.prompts[prompt_key] = Prompt(
                id=prompt_id.text,
                name=prompt_name.text,
                module=module_name,
                module_id=module_id,
                type='Play',
                status=status,
                source_location=location
            )
        
        # Track location
        self.prompt_locations[prompt_key].append(
            f"{module_name}:{location}"
        )

    def get_results(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        results = []
        for prompt in self.prompts.values():
            results.append({
                'ID': prompt.id,
                'Name': prompt.name,
                'Module': prompt.module,
                'ModuleID': prompt.module_id,
                'Type': prompt.type,
                'Status': prompt.status,
                'Occurrences': prompt.occurrence_count,
                'Locations': '; '.join(self.prompt_locations[(prompt.id, prompt.name)])
            })
        return pd.DataFrame(results)

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
        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        return None

def main():
    st.set_page_config(page_title="IVR Prompt Analyzer", layout="wide")
    st.title("IVR Prompt Analyzer")
    
    # Process IVR files
    ivr_dir = "./IVRs"
    results_list = []
    
    try:
        ivr_files = list(Path(ivr_dir).glob('*.five9ivr')) + list(Path(ivr_dir).glob('*.xml'))
        
        if not ivr_files:
            st.warning("No IVR files found in the ./IVRs directory")
            return
            
        for file_path in ivr_files:
            df = analyze_ivr_file(str(file_path))
            if df is not None:
                results_list.append(df)
        
        if not results_list:
            st.error("No valid results found from any IVR files")
            return
            
        # Combine results
        final_results = pd.concat(results_list, ignore_index=True)
        
        # Display results
        st.subheader("Prompt Analysis Results")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.multiselect(
                "Filter by Status",
                options=sorted(final_results['Status'].unique())
            )
        with col2:
            type_filter = st.multiselect(
                "Filter by Type",
                options=sorted(final_results['Type'].unique())
            )
        
        # Apply filters
        filtered_results = final_results
        if status_filter:
            filtered_results = filtered_results[filtered_results['Status'].isin(status_filter)]
        if type_filter:
            filtered_results = filtered_results[filtered_results['Type'].isin(type_filter)]
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Prompts", len(final_results))
        with col2:
            st.metric("In Use", len(final_results[final_results['Status'] == '✅ In Use']))
        with col3:
            st.metric("Not In Use", len(final_results[final_results['Status'] == '❌ Not In Use']))
        
        # Display results table
        st.dataframe(
            filtered_results[['Name', 'Status', 'Type', 'Module', 'Occurrences', 'Locations', 'Source File']]
            .sort_values(['Status', 'Name'])
        )
        
    except Exception as e:
        st.error(f"Error processing IVR files: {str(e)}")
        logger.error("Error in main processing", exc_info=True)

if __name__ == "__main__":
    main()
