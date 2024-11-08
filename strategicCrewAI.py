import os
import streamlit as st
from crewai import Agent, Task, Process, Crew
from langchain_openai import ChatOpenAI
from datetime import datetime

from crewai_tools import (
    FileReadTool,
    WebsiteSearchTool
)

# Configuração do ambiente da API
os.environ["OPENAI_API_KEY"] = <KEY>
os.environ["SERPER_API_KEY"] = <KEY>

# Inicializa o modelo LLM com OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    frequency_penalty=0.5
)


# Função de login
def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return True
    
    st.subheader("Página de Login")

    # Entradas de nome de usuário e senha
    username = st.text_input("Nome de Usuário", type="default")
    password = st.text_input("Senha", type="password")

    # Validação de login
    if st.button("Entrar"):
        if username == "admin" and password == "senha123":  # Aqui você pode trocar pelas credenciais reais
            st.session_state.logged_in = True
            st.success("Login bem-sucedido!")
            return True
        else:
            st.error("Usuário ou senha incorretos.")
            return False
    return False

# Verifique se o login foi feito antes de exibir o conteúdo do aplicativo
if login():
    # Interface do Streamlit
    st.title('Criação de Plano Estratégico e Tático de Marketing')

    # Inputs do cliente
    cliente_nome = st.text_input('Nome do Cliente:')
    site = st.text_input('Site do Cliente:')
    ramo_atuacao = st.text_input('Ramo de Atuação:')
    intuito_plano = st.text_input('Intuito do Plano Estratégico:')
    publico_alvo = st.text_input('Público-Alvo:')
    competidores = st.text_input('Concorrentes:')
    site_competidores = st.text_input('Site dos Concorrentes:')

    # Validação de entrada
    if st.button('Iniciar Planejamento'):
        if not cliente_nome or not ramo_atuacao or not intuito_plano or not publico_alvo:
            st.write("Por favor, preencha todas as informações do cliente.")
        else:
            agentes = [
                Agent(
                    role="Líder e revisor geral de estratégia",
                    goal=f"Revisar toda a estratégia de {cliente_nome} e garantir alinhamento com os objetivos de marca e o público-alvo '{publico_alvo}'.",
                    backstory=f"Você é Philip Kotler, renomado estrategista de marketing, está liderando o planejamento de {cliente_nome} no ramo de {ramo_atuacao}.",
                    allow_delegation=False,
                    llm=llm
                ),
                Agent(
                    role="Analista PEST",
                    goal=f"Realizar a análise PEST para o cliente {cliente_nome}.",
                    backstory=f"Você é Philip Kotler, está liderando a análise PEST para o planejamento estratégico de {cliente_nome}.",
                    allow_delegation=False,
                    llm=llm
                ),
                Agent(
                    role="Criador do posicionamento de marca",
                    goal=f"Criar o posicionamento adequado para {cliente_nome}, considerando o público-alvo '{publico_alvo}'.",
                    backstory=f"Você é Al Ries, responsável por desenvolver o posicionamento de marca de {cliente_nome}.",
                    allow_delegation=False,
                    llm=llm
                ),
                Agent(
                    role="Criador do Golden Circle",
                    goal=f"Desenvolver o Golden Circle para {cliente_nome}, considerando o público-alvo '{publico_alvo}'.",
                    backstory=f"Você é Simon Sinek, está desenvolvendo o Golden Circle de {cliente_nome}.",
                    allow_delegation=False,
                    llm=llm
                ),
                Agent(
                    role="Criador da Brand Persona",
                    goal=f"Definir a Brand Persona de {cliente_nome}, garantindo consistência na comunicação.",
                    backstory=f"Você é Marty Neumeier, está criando a Brand Persona para {cliente_nome}.",
                    allow_delegation=False,
                    llm=llm
                ),
                Agent(
                    role="Criador da Buyer Persona e Público-Alvo",
                    goal=f"Definir a buyer persona e o público-alvo de {cliente_nome}.",
                    backstory=f"Você é Adele Revella, está conduzindo a criação da buyer persona de {cliente_nome}.",
                    allow_delegation=False,
                    llm=llm
                ),
                Agent(
                    role="Criador da Matriz SWOT",
                    goal=f"Desenvolver uma análise SWOT para {cliente_nome} com base nas informaçõe sobre os concorrentes {competidores} com os respectivos sites {site_competidores}.",
                    backstory=f"Você é Michael Porter, está desenvolvendo a análise SWOT de {cliente_nome}.",
                    allow_delegation=False,
                    llm=llm,
                    tools = [WebsiteSearchTool()],
                    max_iter=1000,
                    max_execution_time=1000
                ),
                Agent(
                    role="Criador do Tom de Voz",
                    goal=f"Definir o tom de voz de {cliente_nome}.",
                    backstory=f"Você é Ann Handley, está desenvolvendo a voz da marca para {cliente_nome}.",
                    allow_delegation=False,
                    llm=llm
                )
            ]

            # Criando tarefas correspondentes aos agentes
            tasks = [
                Task(
                    description="Revisar a estratégia geral.",
                    expected_output="Revisão completa da estratégia, bem detalhada e concisa, considerando o público-alvo em português. Output máximo de 2000 tokens.",
                    agent=agentes[0],
                    output_file='output/1.md'
                ),
                Task(
                    description="Criar o posicionamento de marca.",
                    expected_output="Posicionamento de marca em uma única frase. Exemplos de referência seriam 'Just do it' da Nike ou 'Por que você vale muito' da Loreal ou 'Abra a Felicidade' da Coca Cola, considerando o público-alvo em português. Output máximo de 200 tokens.",
                    agent=agentes[2],
                    output_file='output/2.md'
                ),
                Task(
                    description="Desenvolver o Golden Circle.",
                    expected_output="Golden Circle completo com 'how', 'why' e 'what' resumidos em uma única frase cada, considerando o público-alvo em português. Output máximo de 1000 tokens.",
                    agent=agentes[3],
                    output_file='output/3.md'
                ),
                Task(
                    description="Criar a Brand Persona.",
                    expected_output="Brand Persona definida, bem detalhada e concisa em português, com nome da persona de uma pessoa normal (fernando, maria, etc) em português, alinhada com o público-alvo. Output máximo de 1500 tokens.",
                    agent=agentes[4],
                    output_file='output/4.md'
                ),
                Task(
                    description="Definir a Buyer Persona e o Público-Alvo.",
                    expected_output=f"Buyer Persona e Público-Alvo definidos em português, bem detalhados e concisos, descrevendo características do público-alvo '{publico_alvo}', suas dores, objetivos, objeções, resultados desejados, motivações, canais de comunicação preferidos, classe social, região, interesses e gênero. Output máximo de 1500 tokens.",
                    agent=agentes[5],
                    output_file='output/5.md'
                ),
                Task(
                    description="Criar a Matriz SWOT.",
                    expected_output=f"Análise SWOT completa para {cliente_nome}, com forças, fraquezas, oportunidades e ameaças bem detalhadas e concisas em formato de tabela segmentando cada etapa SWOT. Output máximo de 1500 tokens.",
                    agent=agentes[6],
                    output_file='output/6.md'
                ),
                Task(
                    description="Definir o Tom de Voz.",
                    expected_output="Tom de voz definido, incluindo nuvem de palavras, palavras proibidas e princípios que guiam a comunicação. Output máximo de 1500 tokens.",
                    agent=agentes[7],
                    output_file='output/7.md'
                ),
                Task(
                    description="Análise PEST.",
                    expected_output="Análise PEST com pelo menos 5 pontos em cada etapa PEST.",
                    agent=agentes[1],
                    output_file='output/8.md'
                )
            ]

            # Processo do Crew
            equipe = Crew(
                agents=agentes,
                tasks=tasks,
                process=Process.hierarchical,
                manager_llm=llm,
                verbose=True,
                language='português brasileiro'
            )

            # Executa as tarefas do processo
            resultado = equipe.kickoff()

            # Exibir resultados no Streamlit
            for idx, task in enumerate(tasks):
                # Verifica se o arquivo foi gerado corretamente
                output_file_path = f"output/{idx + 1}.md"
                if os.path.isfile(output_file_path):
                    with open(output_file_path, 'r', encoding='utf-8') as file:
                        file_content = file.read()
                        st.markdown(f"**Resultado da Tarefa {idx + 1}:**\n\n{file_content}", unsafe_allow_html=True)
                else:
                    st.write(f"Arquivo para a Tarefa {idx + 1} não encontrado ou não gerado.")


        # Executa a equipe e retorna o resultado
        resultado = equipe.kickoff()
        log += f"Equipe finalizou o planejamento!\nResultado Final:\n{resultado}\n"
        
