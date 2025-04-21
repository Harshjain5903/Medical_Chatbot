The Medical ChatBot project is designed to enhance healthcare accessibility by creating an intelligent chatbot that can answer basic medical questions and provide emotional support. Utilizing the advanced capabilities of Meta's LLaMA 2 model, this chatbot aims to offer reliable and immediate assistance to users seeking medical information and emotional comfort. By incorporating open-source technologies such as Python, LangChain, Flask, and Pinecone, the project provides a comprehensive guide to building a functional and scalable chatbot.

The chatbot's core functionalities include understanding and responding to user queries related to common medical issues, offering suggestions for next steps, and providing empathetic responses to users in need of emotional support. The project covers everything from setting up the development environment and installing necessary dependencies to configuring the chatbot with the required models and running the application.



# End-to-end-Medical-Chatbot-using-Llama2

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone


