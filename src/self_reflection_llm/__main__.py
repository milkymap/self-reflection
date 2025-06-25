from self_reflection_llm import main 
from dotenv import load_dotenv


# python -m src.self_reflection_llm -p mcp_servers.json -a gpt-4.1 -e o4-mini -sr o3 -sm gpt-4.1-mini -re high -prl true

if __name__ == "__main__":
    load_dotenv()
    main()