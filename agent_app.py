import sys
sys.path.append('.')

from src.agent.agent import initialize_agent, run_agent

def main():
    agent = initialize_agent()
    
    print("Research Agent ready. Type 'quit' to exit.\n")
    
    while True:
        try:
            question = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        
        if not question:
            continue
        if question.lower() == "quit":
            print("Goodbye!")
            break
        
        print("\nAgent thinking...\n")
        answer = run_agent(agent, question)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()