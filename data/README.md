# Data

The dataset is not included because it belongs to the course (ECON 424, University of Waterloo). It was a **simulated** labour market, not real people.

Each row was one job candidate, with:

- free-text hiring materials (hiring-team discussions and chatbot interview transcripts)
- the wage offered, or "not hired"

Demographics were **not** given as columns; they had to be inferred from the text.

To run the script, save the file here as `wage_simulation_data.csv` and put your OpenAI key in a `.env` file in the repo root:

```
OPENAI_API_KEY=<your key>
```
