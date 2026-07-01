# Graph Representation Learning for LLM

## Getting started

TO DO

## Temporary Notes

##### Datasets

GraphQA for basic representation learning of graph

##### Hierarchical Reasoning Models

These models are good at sudoku, crosswords etc… These data are not sequential and are relatively closer to graph data than to text. Consequently, testing them on graph seems like a promising idea.

##### Meta-Learning

Teach Meta-ICL with diverse graph tasks (like GraphQA tasks at first) expecting that exploring multi graph tasks even with graph prompted in raw text improves the graph structure understanding of the model.

Experiment 1 (Does meta learning improves LLM perfs on raw graph-text tasks ?) : show all tasks at training and see if it beats Talk Like a Graph results for instance 

Experiment 2 (Does meta learning actually enhance the comprehension of the graph structure ?) : show only 5 / 8 graph tasks and see if it helps generalizing on unseen tasks at train (X → X)