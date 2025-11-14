# Detector Scope: Demo Interface

This code runs the demo interface for the detector scope project

## Installation
```
pip install streamlit
```

## Setup
In order to run the interface you must have the following

- A trained detector for both llama and gpt4o
- A trained sae for both detectors
- A csv containing descriptions and log odds for all features for both saes

If those artifacts are not available consult the main project README for how
to generate them.

## Example Usage:
```
streamlit run app.py -- -d <path_to_gpt4o_detector> -ld <path_to_llama_detector> -s <path_to_gpt4o_sae> -ls <path_to_llama_sae> -f <path_to_gpt4o_feature_descriptions> -lf <path_to_llama_feature_descriptions>
```
