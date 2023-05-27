---
# South Park Chatbot Model Card

## Model details
**Organization developing the model**
Bobidi

**Model date**
Trained in April 2023

**Model version**
Version 0.1

**Model type**
LLaMA South Path is a model containing South Park character personas, finetuned from a LLaMA model.

**More information**
Please readme in the github repo.

**Citations details**
Please cite the [github repo](https://github.com/bobidi/llama_south_park) if you use the data or code in this repo.

**License**
Code and data are licensed under the Apache 2.0 license.

**Where to send questions or comments about the model**
Questions and comments about this model can be sent via the [GitHub repository](https://github.com/bobidi/llama_south_park) of the project, by opening an issue.

## Intended use
**Primary intended uses**
The primary use of LLaMA South Path is research on persona finetuning.

**Primary intended users**
The primary intended users of the model are researchers in natural language processing and fans of the American animated sitcom South Park.

**Out-of-scope use cases**
This model is not finetuned for obtaining trustworthy information from the model. Thus, this model is not intended for use in production systems. 

## Metrics
This model is subjectively evaluated by several human in a brief manner. This is not intended for comparing with other persona chatbot models.

## Evaluation datasets
The model was evaluated on the self-instruct evaluation set.

## Training dataset
This model is trained on 70k script data, processed from [a script from Kaggle](https://www.kaggle.com/datasets/thedevastator/south-park-scripts-dataset).