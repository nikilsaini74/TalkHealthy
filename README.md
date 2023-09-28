# AI-Powered Fitness Chatbot

## TalkHealthy

TalkHealthy is a web application developed in Python that serves as an AI-powered fitness chatbot. It aims to provide users with personalized answers and guidance on diet, gym, and exercise-related questions. Leveraging the Llama model, the chatbot facilitates natural language processing and understanding of user queries, making interactions more intuitive and efficient.

## Features
- **Llama Model:** The chatbot is built using the Llama model, a state-of-the-art language model, to facilitate natural language processing and improve the understanding of user queries.
- **Semantic Query System:** TalkHealthy leverages a vector database, ChromaDB, to store an extensive collection of data on diet, nutrition, fitness, and exercise. This integration empowers the chatbot with a powerful semantic query system, enabling it to combine user queries with relevant contextual information. As a result, it enhances the accuracy and relevance of the responses provided to the users.

## Technologies Used
- LLM (WizardLM-7B)
- Frontend: Streamlit
- Python 3.10
- Langchain
- Sentence Transformer (all-MiniLM-L6-v2)
- Vector Database: ChromaDB

## How to Run
1. Clone the repository to your local machine
```
git clone https://github.com/nikilsaini74/TalkHealthy.git
```
2. Navigate to the project directory
```
cd TalkHealthy
```
3. Create and activate a virtual environment
```
python -m venv env
source env/bin/activate
```
4. Install the required dependencies
```
pip install -r requirements.txt
```
`LLM model should be placed under models/ directory:`
  
  **Model download URL:** https://huggingface.co/TheBloke/wizardLM-7B-GPTQ  
  
  _llamacpp: WizardLM-7B-uncensored.ggmlv3.q4_K_S.bin_

5. Run the API
```
python main_api.py
```
API is available at http://localhost:8080

6. Run the app
```
streamlit run main_web.py
```
7. Open your browser and navigate to http://localhost:8501 ,enjoy the app and stay fit.

## Contributions
Contributions to TalkHealthy are welcome! If you find any bugs or have suggestions for improvement, please feel free to open issues or submit pull requests.


## License
https://github.com/facebookresearch/llama/blob/main/LICENSE

**Let's make the world a healthier place together with TalkHealthy! 🏋️‍♀️🥗🤖**
