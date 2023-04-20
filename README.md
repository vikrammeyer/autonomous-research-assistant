# AutoResearchGPT

## Problem

- How to keep up with the flood of AI research?
	- Number of new papers submitted to Arxiv daily is far too much for any researcher to keep up with
- What if we had a research assistant whose job was to keep up to date on the current developments in AI?
	- Would free up valuable time for deep thinking and research

## Approach
- Use autonomous agents powered by LLMs to act as personal research assistants
	- The technology powering this new wave of AI research can also be used to help ride the wave better
	- Beneficial relationship
		- Autonomous agents are good for searching through and summarizing vasts amounts of existing research since deep learning excels at finding patterns in data
		- Gives human researchers more time for thinking deeply about solving problems (things autonomous agents cannot currently do)
- Create an autonomous agent using a LLM at its core
	- can fulfill a user defined task by creating subtasks with a LLM query and fulfilling those subtasks with other LLM queries and calls to external tools
- Base Architecture of Autonomous Agent:
	- LLMs as the core reasoning module that can request tools
	    - Arxiv API
	    - Read/Write File
	- VectorStores act as "memory" by storing text embeddings of requests and results as intermediate steps towards the objective
	- Everything is connected with LangChain

## Method

- AutoGPT's Core LLM: GPT-3.5-turbo
- Chain together multiple calls to LLM storing intermediate results in a vectorstore and querying the vectorestore to add relevant results to the context of another LLM call
- Each pass through the LLM will generate a "thought" which is a json response of the form:

```json
{
    "thoughts": {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user"
    },
    "command": {
        "name": "command name",
        "args": {
            "arg name": "value"
        }
    }
}
```

## Example

Task: write a summary on the latest trends state in natural language processing research

Initial Prompt to GPT3.5-turbo:

```
System: You are Koda, Research Assistant
Your decisions must always be made independently
            without seeking user assistance. Play to your strengths
            as an LLM and pursue simple strategies with no legal complications.
            If you have completed all your tasks,
            make sure to use the "finish" command.

GOALS:

1. write a summary on the latest trends state in natural language processing research


Constraints:
1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. No user assistance
4. Exclusively use the commands listed in double quotes e.g. "command name"

Commands:
1. Arxiv: useful for when you need to gather summaries and titles of research papers. Input: a query for research papers of the format 'cat:cs.<subject category> and <topic>' where subject category can be CL (for natural language processing), CV (for computer vision and pattern recognition), or LG (for general machine learning). Output: the top 3 paper titles and summaries matching the query., args json schema: {"query": {"title": "Query", "type": "string"}}
2. write_file: Write file to disk, args json schema: {"file_path": {"title": "File Path", "description": "name of file", "type": "string"}, "text": {"title": "Text", "description": "text to write to file", "type": "string"}}
3. read_file: Read file from disk, args json schema: {"file_path": {"title": "File Path", "description": "name of file", "type": "string"}}
4. finish: use this to signal that you have finished all your objectives, args: "response": "final response to let people know you have finished your objectives"

Resources:
1. Internet access for searches and information gathering.
2. Long Term memory management.
3. GPT-3.5 powered Agents for delegation of simple tasks.
4. File output.

Performance Evaluation:
1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

You should only respond in JSON format as described below
Response Format:
{
    "thoughts": {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user"
    },
    "command": {
        "name": "command name",
        "args": {
            "arg name": "value"
        }
    }
}
Ensure the response can be parsed by Python json.loads
System: The current time and date is Mon Apr 17 12:31:58 2023
System: This reminds you of these events from your past:
[]


Human: Determine which next command to use, and respond using the format specified above:
```

Example Initial Response:
``` json
{
    "thoughts": {
        "text": "I think I should start by gathering some research papers on the latest trends in natural language processing.",
        "reasoning": "I need to have a good understanding of the current state of research in natural language processing before I can write a summary on the latest trends.",
        "plan": "- Use the Arxiv command to gather summaries and titles of research papers on natural language processing.\n- Save the summaries and titles to a file for future reference.",
        "criticism": "I need to make sure that I am not just gathering papers for the sake of it, but that I am actually reading and understanding the content of each paper.",
        "speak": "I will use the Arxiv command to gather summaries and titles of research papers on natural language processing."
    },
    "command": {
        "name": "Arxiv",
        "args": {
            "query": "cat:cs.CL and latest trends in natural language processing"
        }
    }
}
```

Recall the task was:  write a summary on the latest trends state in natural language processing research
File contents AutoResearchGPT output: 

```
Latest Trends in Natural Language Processing Research:

1. Natural language generation
2. Pre-trained language models
3. Multilingual models

Notes:
- Natural language generation is becoming increasingly popular in applications such as chatbots and virtual assistants.
- Pre-trained language models such as GPT-3 are achieving state-of-the-art performance on a wide range of natural language processing tasks.
- Multilingual models are becoming more important as the demand for natural language processing in non-English languages increases.
```

### Prompting Tool Usage

- Objective: write a detailed summary on the latest trends in natural language processing and computer vision research
- does breakdown the external paper search into 2 separate Arxiv API requests, one for CV another for NLP
	- but also continues on afterwards making multiple weird API queries
- Experimented with prompting the LLM with a description of the Arxiv tool that included the documentation for the API's query interface 
	- Idea: since the query is not very much natural language, we provided in the tool's description how to craft the queries according to the API
	- Was able to adapt its queries to the format but suprisingly this gave worse results

## Analysis

#### Limitations

- AutoGPT can be a repetitive in its steps to complete a task
- Might try to use tools not available to it 
	- read file tool if you give it write file tool
	- nonexistant Google Scholar tool
	
- Arxiv API is fairly limited and not tuned the best towards natural language queries
  - doesn't let us sort by topic query *and* date so a different tool would need to find relevant papers to search for

#### Takeaways

- This project is a step forward towards personal research assistants
- Demonstrated we can integrate APIs into a LLMs workflow even if it has not been trained to use that API specifically
    - More structured APIs (i.e. inputs are not in natural language) require additional prompting describing the interface. 

#### Next Steps

- Adding more tools to the agent!
- Twitter: impactful/important papers are highly liked and retweeted
- SemanticSearch API provides more information on papers like citations and references so it could be used by an agent for automated literature review. downside is it doesn't have all the papers of arxiv 

#### Societal Impact of Autonomous Agents

- Have huge potential to augment the productivity of humans
- Free humans up from "doing" to have more time for "thinking"
- will lower the barriers of entry for a person to capitalize on their idea
	- 1-person startup that can do the work of a 50-person startup using autonomous agents for marketing,
- Will likely decrease the need for many jobs like personal assistants, secretaries, social media managers, etc.
	- Much cheaper than humans and more efficient!
- Depending on how AI becomes regulated, autonomous agents can potentially 
	- increase inequality by concentrating power in the hands of those with the most data and compute to create the best LLMs
	- decrease inequality by giving everyone easy access to productivity enhancing autonomous agents similar to the smart phone revolution

## Resources

- [The Complete Beginner Guide to Autonomous Agents](https://www.mattprd.com/p/the-complete-beginners-guide-to-autonomous-agents)
- [(github repo) BabyAGI ](https://github.com/yoheinakajima/babyagi/blob/main/babyagi.py)
- [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/)
- [AutoGPT with LangChain](https://python.langchain.com/en/latest/use_cases/autonomous_agents/marathon_times.html#autogpt-as-a-research-data-munger)
- [Video Overview]( )
