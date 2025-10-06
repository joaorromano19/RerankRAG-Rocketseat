# RerankRAG-Rocketseat

Este projeto implementa um sistema de busca semântica (RAG – Retrieval-Augmented Generation) com LangChain, OpenAI, Cohere e ChromaDB.
Com ele, é possível carregar um PDF, dividir o texto em partes (chunks), criar embeddings vetoriais e realizar perguntas para obter respostas precisas com base no conteúdo do documento.
Além disso, o projeto utiliza compressão contextual e reranking para refinar os resultados e melhorar a relevância das respostas.

Tecnologias Utilizadas

LangChain — Framework para orquestração de LLMs

OpenAI API — Geração de embeddings e respostas

Cohere API — Reclassificação e compressão contextual

ChromaDB — Banco de vetores para busca semântica

PyPDFLoader — Leitura e extração de conteúdo de PDFs

Instalação
Clone o repositório

git clone https://github.com/seuusuario/seurepositorio.git

cd seurepositorio

Crie um ambiente virtual

python -m venv .venv

.venv\Scripts\activate # Windows

source .venv/bin/activate # Linux/Mac

Instale as dependências

pip install -r requirements.txt

Configuração das Chaves de API

Antes de executar o projeto, defina suas chaves no código:

os.environ["OPENAI_API_KEY"] = "sua_chave_openai_aqui"

os.environ["COHERE_API_KEY"] = "sua_chave_cohere_aqui"

Como Funciona o Código
Carregamento do PDF

pdf = PyPDFLoader(file_path = "Seu PDF", extract_images = False)

pages = pdf.load_and_split()

Divisão dos textos

O texto é dividido em partes menores (chunks) para facilitar o processamento:

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 4000,
    chunk_overlap = 20,
    length_function = len,
    add_start_index = True
)
chunk = text_spliter.split_documents(pages)

Criação do Vetorstore e Retriever

Os textos divididos são transformados em vetores e armazenados em uma base local:

embeddings_model = OpenAIEmbeddings(model = "text-embeddings-3-small")
vectorDB = Chroma(embedding_function = embeddings_model, persist_directory = "naiveDB")
naive_retriever = vectorDB.as_retriever(kwargs = {"k": 10})

Reranking e Compressão Contextual

O CohereRerank é utilizado para reordenar e comprimir os resultados mais relevantes:

rerank = CohereRerank(model = "rerank-v3.5", top_n = 3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor = rerank,
    base_retriever = naive_retriever
)

Criação do Prompt e Execução da Query

Um template é criado para estruturar a interação entre o modelo e o contexto do documento:

TEMPLATE = """
Sua especificação do que o agente de ia é...

Query:
{question}

Context:
{context}
"""


O prompt é combinado com o modelo e o retriever para formar a cadeia de execução:

reg_prompt = ChatPromptTemplate.from_template(TEMPLATE)
compression_retrieval_chain = setup_retrieval | reg_prompt | llm | output_parser
compression_retrieval_chain.invoke("Faça sua pergunta...")
