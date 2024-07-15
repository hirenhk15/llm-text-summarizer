# LLM App to summarize any content from the user given text.

import streamlit as st
from enum import Enum
from io import StringIO
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import CharacterTextSplitter

CREATIVITY=0


class ModelType(Enum):
    GROQ='GroqCloud'
    OPENAI='OpenAI'


class LLMModel:
    def __init__(self, model_provider: str) -> None:
        self.model_provider = model_provider

    def load(self, api_key=str):
        try:
            if self.model_provider==ModelType.GROQ.value:
                llm = ChatGroq(temperature=CREATIVITY, model="llama3-70b-8192", api_key=api_key) # model="mixtral-8x7b-32768"
            if self.model_provider==ModelType.OPENAI.value:
                llm = OpenAI(temperature=CREATIVITY, api_key=api_key)
            return llm
        
        except Exception as e:
            raise e


class LLMStreamlitUI:
    def __init__(self) -> None:
        pass

    def validate_api_key(self, key:str):
        if not key:
            st.warning("Please enter your API Key")
            # st.stop()
        else:    
            if (key.startswith("sk-") or key.startswith("gsk_")):
                st.success("Received valid API Key!")
            else:
                st.error("Invalid API Key!")

    def get_api_key(self):
        
        # Get the API Key to query the model
        input_text = st.text_input(
            label="Your API Key",
            placeholder="Ex: sk-2twmA8tfCb8un4...",
            key="api_key_input",
            type="password"
        )

        # Validate the API key
        self.validate_api_key(input_text)
        return input_text
    
    def create(self):
        try:
            # Set the page title for blog post
            st.set_page_config(page_title="Text Summarization App")
            st.markdown("<h1 style='text-align: center;'>Text Summarization App</h1>", unsafe_allow_html=True)

            # Select the model provider
            # st.markdown("## Which model provider you want to choose?")
            option_model_provider = st.selectbox(
                    'Select the model provider',
                    ('GroqCloud', 'OpenAI')
                )

            # Input API Key for model to query
            # st.markdown(f"## Enter Your {option_model_provider} API Key")
            # Input API Key for model to query
            api_key = self.get_api_key()

            txt_input = st.text_area("Enter your text", "", height=200)

            result = []
            with st.form("summarize_form", clear_on_submit=True):
                # col1, col2, col3 = st.columns([1, 2, 1])
                # with col2:
                    
                submitted = st.form_submit_button("Submit")
                # st.markdown('<div class="centered">{}</div>'.format(submitted), unsafe_allow_html=True)

                if submitted:
                    # Split the text into chunks
                    text_splitter = CharacterTextSplitter()
                    texts = text_splitter.split_text(txt_input)
                    docs = [Document(page_content=t) for t in texts]
                    
                    # Load the LLM model
                    llm_model = LLMModel(model_provider=option_model_provider)
                    llm = llm_model.load(api_key=api_key)

                    chain = load_summarize_chain(
                        llm=llm,
                        chain_type="map_reduce"
                    )
                    summary_output = chain.invoke(docs)
                    
                    result.append(summary_output)
                    del api_key
            
            if len(result):
                st.info(summary_output["output_text"])


        except Exception as e:
            st.error(str(e), icon=":material/error:")


def main():
    # Create the streamlit UI
    st_ui = LLMStreamlitUI()
    st_ui.create()


if __name__ == "__main__":
    main()