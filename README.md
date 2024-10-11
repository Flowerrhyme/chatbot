# Chinese chatbot
## Introduction
This chatbot system includes three response modes: neural network response, knowledge base response, and real-time encyclopedia query. You can specify the response mode via commands, with the default being a combination of knowledge base and neural network responses.

1. Neural Network
The neural network part uses a GRU-based encoder-decoder network with added attention.
The decoding strategies implemented are greedy, beam search, and nucleus filtering.
2. Knowledge Base Query
Knowledge base responses rely on professional Q&A data and conversational responses, matching tags from segmented input text.
3. Real-time Encyclopedia Query
The real-time encyclopedia query uses a web crawler to fetch the summary of corresponding keywords from Baidu Encyclopedia and returns the result.


## Usage
Both the model and corpus are provided. Run web.py and visit http://127.0.0.1:5000/ to access the chat interface.

Start your input with the '@' symbol to perform an encyclopedia search.
Start your input with the '#' symbol to force the use of the neural network.

