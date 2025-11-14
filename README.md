# Interpretable AI Detectors Project

This project aims to leveraging recent advances in mechanistic interpretability,
specifically Sparse Auto-Encoders (SAEs), to build AI detectors that provide
faithful, human-readable explanations for their predictions.

## Installation

Conda:
```
conda create -n detectorscope python=3.11
conda activate detectorscope
pip install -r requirements.txt
```
Venv:
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Set API Key
In addition, before running the code, make sure to specify the API key you
will be using by running
```
$ export GEMINI_API_KEY='your api key'
```
Without this you will get the following error
```
ValueError: Missing key inputs argument! To use the Google AI API, provide (`api_key`) arguments. To use the Google Cloud API, provide (`vertexai`, `project` & `location`) arguments.
```

## Usage

To run the project, pass in a detector to `generate_report.py`. This will begin
an interactive test where you can input text and our scripts will return a LLM
generated report with the top 5 features used by the classifier to get that
output.

Example:
```
$ python generate_report.py -d <path_to_your_detector>
```
Output:
```
> Input Text: Hey man what's up? It's been a while
13492: -0.7107 (Human) -- positive sentiments related to vehicle experiences and reviews
3857: -0.5521 (Human) --  greetings and expressions of goodwill
14305: 0.5215 (GPT4) -- discussions around social structures and governance
10594: -0.4365 (Human) -- conversational phrases expressing frustration and the desire for clarity or reassurance
11582: -0.3888 (Human) -- numerical time-related references, particularly years and months
Generating Report...
The AI detector classified the provided text as being written by a human with 96.67% likelihood. The primary drivers for this classification were several features strongly associated with human conversational patterns. According to the classifier's output, the most influential feature was related to positive sentiments found in vehicle experiences and reviews. The next most significant feature, which directly corresponds to the input text, was the identification of greetings and expressions of goodwill, as seen in the phrase "Hey man what's up".

Further contributing to the human classification were features associated with conversational phrases that express a desire for clarity and numerical time-related references. The latter was likely triggered by the phrase "It's been a while." It is also noted that the model detected one feature associated with GPT-4, related to discussions of social structures and governance. However, the combined weight of the human-associated features was significantly stronger, leading to the final high-confidence determination that the text was human-generated.
```

### Detector Training

To train your own detector for use in our report generation, first run
`process_dataset.py` to cache activations for the SAE on your dataset of choice.

Example Usage:
```
$ python process_dataset.py -h
  -d, --data_path    Remote ID of the dataset on huggingface datasets hub
  -t, --text_column  The name of the column containing the text
  -n, --n_examples   The number of dataset examples to cache activations for (Optional, defaults to full dataset)
  -o, --output_path  The path to the output dataset to save the activations
```

After you finish caching the SAE activations, use `train_detector.py` to
train your interpretable detector.

Usage:
```
$ python train_detector.py -h
  -d, --data_path          Local path to the dataset with cached SAE activations
  -n, --num_examples       Num examples to train on (Optional, defaults to full dataset)
  -l, --learning_rate      Learning Rate (default: 5e-5)
  -b, --batch_size         Batch Size (default: 32)
  -t, --test_set_size      Number of examples to include in the test split
  -e, --eval_every         Evaluate on test data every X batches
  -s, --tensorboard_dir    Parent directory to write tensorboard logs to
  -m, --model_out_prefix   Name for output model
```

Detectors will be saved along with a timestamp. For example, if you add the
flag `-m test_model` the detector will be saved in the
format `test_model_{YYMMDD_HHMM}`

## Citing this work

Add citation details here, usually a pastable BibTeX snippet:

```
@article{publicationname,
      title={Publication Name},
      author={Author One and Author Two and Author Three},
      year={2025},
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
