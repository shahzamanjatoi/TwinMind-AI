import asyncio, os, sys
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

provider = OpenAIProvider(base_url=os.environ["BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
model = OpenAIChatModel(os.environ["MODEL_NAME"], provider=provider)

twinmind = MCPServerStdio(sys.executable, args=[str(Path(__file__).parent / "twinmind_mcp.py")])

agent = Agent(
    model=model,
    mcp_servers=[twinmind],
    system_prompt=(
        "You are TwinMind, an industrial maintenance copilot for motor vibration data. "
        "Always use tools for facts: analyze_recording for a whole recording, classify_fault for one "
        "set of readings, compare_to_baseline to explain what differs from healthy operation, "
        "get_maintenance_guidance for actions. Never guess a diagnosis without calling a tool. "
        "State that results are decision support, not a confirmed fault."
    ),
)

async def ask(prompt: str) -> str:
    async with agent.run_mcp_servers():
        return (await agent.run(prompt)).output

if __name__ == "__main__":
    for q in [
        "Which recordings are available?",
        "Analyze the FaultedBearing recording and tell me what maintenance to do.",
        "How does FaultedCoupling differ from baseline? Which sensors changed most?",
    ]:
        print("\nQ:", q)
        print("A:", asyncio.run(ask(q)))