# Certification mentor to help you study for Anthropic's CCAR-F exam.

This is an AI bot that helps you study for Anthropic's exam. Still not great, but getting there 😉

Using `uv`, follow the steps below to start the application:

1. Create a `.env` file in the project root with your OpenAI key:

```
OPENAI_API_KEY=sk-...
```

2. From the project root, install the dependencies and start the exam:

```bash
uv sync
uv run certification-mentor
```

## Current facing challenges

- How to initially provide all the context to the LLM so that it generates questions about only the context that I want?
  - what it is doing now: the LLM chooses a category/concept and queries that from the index
- How to enforce the LLM to use a specific set of tools on every interaction?
  - it is adding the question to the "database" right after it is generated. The LLM does not choose this
- Is it possible to split the LLM's process? Like, allow/enable it to have more than 1 answer/response for every user prompt?
