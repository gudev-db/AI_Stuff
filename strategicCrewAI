import os
import streamlit as st
from crewai import Agent, Task, Process, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import FileReadTool, SerperDevTool, WebsiteSearchTool

# Configuração do ambiente da API
os.environ["OPENAI_API_KEY"] = <CHAVE>
os.environ["SERPER_API_KEY"] = <CHAVE>

file_tool = FileReadTool()
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# vetorizando o pdf para leitura menos onerosa dos livros. vectordb não é local. cria o bd de vetores e faz queries dentro. cria instancia global de recuperador.




# Inicializa o modelo LLM com OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini", #mudei pra 4o-mini
    temperature=0.1
)

# criar função pra llm

# Interface do Streamlit
st.title('Criação de Plano Estratégico e Tático de Marketing')

# Inputs do cliente
cliente_nome = st.text_input('Nome do Cliente:')
site = st.text_input('Site do Cliente:')
ramo_atuacao = st.text_input('Ramo de Atuação:')
intuito_plano = st.text_input('Intuito do Plano Estratégico:')
publico_alvo = st.text_input('Público-Alvo:')
competidores = st.text_input('Competidores:')

# Variável para armazenar o log
log = ""

# Validação de entrada
if st.button('Iniciar Planejamento'):
    if not cliente_nome or not ramo_atuacao or not intuito_plano or not publico_alvo:
        st.write("Por favor, preencha todas as informações do cliente.")
    else:
        log += f"Iniciando o planejamento para o cliente: {cliente_nome}\n"
        log += f"Ramo de Atuação: {ramo_atuacao}\n"
        log += f"Intuito do Plano: {intuito_plano}\n"
        log += f"Público-Alvo: {publico_alvo}\n\n"

        agentes = [
            Agent(
                role="Líder e revisor geral de estratégia",
                goal=f"Revisar toda a estratégia de {cliente_nome} e garantir alinhamento com os objetivos de marca e o público-alvo '{publico_alvo}'. Esse {site} é o site do cliente. Extraia o máximo de informações possíveis dele para que você ajuste o planejamento estratégico com base em entendimento sobre quem é o cliente. Use análise PEST para te guiar.",
                backstory=f"Você é Philip Kotler, você fala em português, renomado estrategista de marketing, está liderando o planejamento de {cliente_nome} no ramo de {ramo_atuacao} com os maiores competidores sendo {competidores}. Use o livro Marketing Management como referência.",
                allow_delegation=False,
                llm=llm,
                tools=[search_tool] #removendo rag
            ),
            Agent(
                role="Criador do posicionamento de marca",
                goal=f"Criar o posicionamento adequado para {cliente_nome}, considerando os objetivos e o público-alvo '{publico_alvo}'. O Posicionamento de marca é em uma única frase. Exemplos de referência seriam 'Just do it' da Nike ou 'Por que você vale muito' da Loreal ou 'Abra a Felicidade' da Coca Cola, considerando o público-alvo em português.",
                backstory=f"Você é Al Ries, você fala em português, responsável por desenvolver o posicionamento de marca de {cliente_nome}. Use o livro Positioning: The Battle for Your Mind como referência.",
                allow_delegation=False,
                llm=llm
            ),
            Agent(
                role="Criador do Golden Circle",
                goal=f"Desenvolver o Golden Circle para {cliente_nome}, considerando o público-alvo '{publico_alvo}', definindo o 'WHY', 'HOW', e 'WHAT'.",
                backstory=f"Você é Simon Sinek, você fala em português, está desenvolvendo o Golden Circle de {cliente_nome}. Use o livro Start With Why como referência.",
                allow_delegation=False,
                llm=llm
            ),
            Agent(
                role="Criador da Brand Persona",
                goal=f"Definir a Brand Persona de {cliente_nome}, garantindo consistência na comunicação e alinhamento com o público-alvo '{publico_alvo}'.",
                backstory=f"Você é Marty Neumeier, você fala em português, está criando a Brand Persona para {cliente_nome}. Use o livro The Brand Gap como referência.",
                allow_delegation=False,
                llm=llm            ),
            Agent(
                role="Criador da Buyer Persona e Público-Alvo",
                goal=f"Definir a buyer persona e o público-alvo de {cliente_nome}, detalhando as características do público-alvo '{publico_alvo}'.",
                backstory=f"Você é Adele Revella, você fala em português, está conduzindo a criação da buyer persona de {cliente_nome}. Use o livro Buyer Persona como referência.",
                allow_delegation=False,
                llm=llm            ),
            Agent(
                role="Criador da Matriz SWOT",
                goal=f"Desenvolver uma análise SWOT para {cliente_nome}, destacando forças, fraquezas, oportunidades e ameaças, considerando o público-alvo '{publico_alvo}'.",
                backstory=f"Você é Michael Porter, você fala em português, está desenvolvendo a análise SWOT de {cliente_nome}. Use o livro Competitive Advantage como referência.",
                allow_delegation=False,
                llm=llm            ),
            Agent(
                role="Criador do Tom de Voz",
                goal=f"Definir o tom de voz de {cliente_nome}, incluindo nuvem de palavras e palavras proibidas que se alinhem ao público-alvo '{publico_alvo}'.",
                backstory=f"Você é Ann Handley, você fala em português, está desenvolvendo a voz da marca para {cliente_nome}. Use o livro Everybody Writes como referência.",
                allow_delegation=False,
                llm=llm            )
        ]
        # Criando tarefas correspondentes aos agentes
        tasks = [
            Task(
                description="Revisar a estratégia geral.",
                expected_output="Revisão completa da estratégia, bem detalhada e concisa, considerando o público-alvo em português. Output máximo de 2000 tokens.",
                agent=agentes[0],
                output_file=r'teste2/1.docx'
            ),
            Task(
                description="Criar o posicionamento de marca.",
                expected_output="Posicionamento de marca em uma única frase. Exemplos de referência seriam 'Just do it' da Nike ou 'Por que você vale muito' da Loreal ou 'Abra a Felicidade' da Coca Cola, considerando o público-alvo em português. Output máximo de 200 tokens.",
                agent=agentes[1],
                output_file=r'teste2/2.docx'
            ),
            Task(
                description="Desenvolver o Golden Circle.",
                expected_output="Golden Circle completo com 'how', 'why' e 'what' resumidos em uma única frase cada, considerando o público-alvo em português. Output máximo de 1000 tokens.",
                agent=agentes[2],
                output_file=r'teste2/3.docx'
            ),
            Task(
                description="Criar a Brand Persona.",
                expected_output="Brand Persona definida, bem detalhada e concisa em português, com nome da persona de uma pessoa normal (fernando, maria, etc) em português, alinhada com o público-alvo. Output máximo de 1500 tokens.",
                agent=agentes[3],
                output_file=r'teste2/4.docx'
            ),
            Task(
                description="Definir a Buyer Persona e o Público-Alvo.",
                expected_output="Buyer Persona e Público-Alvo definidos em português, bem detalhados e concisos, descrevendo características do público-alvo '{publico_alvo}', suas dores, objetivos, objeções, resultados desejados, motivações, canais de comunicação preferidos, classe social, região, interesses e gênero. Output máximo de 1500 tokens.",
                agent=agentes[4],
                output_file=r'teste2/5.docx'
            ),
            Task(
                description="Criar a Matriz SWOT.",
                expected_output="Análise SWOT completa para {cliente_nome}, com forças, fraquezas, oportunidades e ameaças bem detalhadas e concisas em formato de tabela segmentando cada etapa SWOT. Output máximo de 1500 tokens.",
                agent=agentes[5],
                output_file=r'teste2/6.docx'
            ),
            Task(
                description="Definir o Tom de Voz.",
                expected_output="Tom de voz definido, incluindo nuvem de palavras, palavras proibidas e princípios que guiam a comunicação. Output máximo de 1500 tokens.",
                agent=agentes[6],
                output_file=r'teste2/7.docx'
            )
        ]

        log += "Tarefas foram definidas.\n"

        # Criação do processo sem passar os agentes diretamente
        # Criando a equipe e executando o planejamento
        equipe = Crew(
            agents=agentes,
            tasks=tasks,
            process=Process.hierarchical,
            manager_llm=llm,
            verbose=True,
            language = 'portugês brasileiro'
        )

        # Executa a equipe e retorna o resultado
        resultado = equipe.kickoff()
        log += f"Equipe finalizou o planejamento!\nResultado Final:\n{resultado}\n"
        
