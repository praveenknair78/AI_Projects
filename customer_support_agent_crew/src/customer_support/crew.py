import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import PDFSearchTool

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


@CrewBase
class customer_support():
    """Customer Support crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    doc_srch_tool = PDFSearchTool(
        pdf='Plan_document.pdf',
        config=dict(
            llm=dict(
                provider="google",
                config=dict(
                    model="gemini/gemini-2.0-flash",
                    api_key=GEMINI_API_KEY,
                    # temperature=0.5,
                    # top_p=1,
                    # stream=true,
                ),
            ),
            embedder=dict(
                provider="google",
                config=dict(
                    model="models/embedding-001",
                    task_type="retrieval_document",
                    # title="Embeddings",
                ),
            ),
        )
    )

    @agent
    def support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['support_agent'],
            tools=[self.doc_srch_tool],
            verbose=True
        )

    @agent
    def support_quality_assurance_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['support_quality_assurance_agent'],
            verbose=True
        )

    @task
    def inquiry_resolution(self) -> Task:
        return Task(
            config=self.tasks_config['inquiry_resolution'],
        )

    @task
    def quality_assurance_review(self) -> Task:
        return Task(
            config=self.tasks_config['quality_assurance_review'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Coder crew"""


        return Crew(
            agents=self.agents, 
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )