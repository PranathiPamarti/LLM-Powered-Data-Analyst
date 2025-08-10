# LLM-Powered Data Analyst API

A FastAPI-based data analysis API powered by a Large Language Model (LLM).  
It dynamically generates Python code to query and analyze SQLite databases based on natural language questions. The API returns numeric results, summary statistics, and visualizations encoded as base64 images.

---

## Features

- Connects to SQLite databases and extracts data using Pandas.
- Uses an LLM (via AIPipe) to dynamically generate Python code for data analysis.
- Handles small datasets with missing or null values gracefully.
- Computes summary statistics: totals, medians, correlations.
- Generates plots with matplotlib, returning base64-encoded PNG images.
- Easy deployment on cloud platforms like Render or Heroku.

---

## Project Structure

```

LLM-Powered-Data-Analyst/
│
├── .env                   # Environment variables, e.g. AIPIPE\_TOKEN
├── LICENSE                # MIT License file
├── README.md              # This file
├── requirements.txt       # Python dependencies
│
├── main.py                # FastAPI app entrypoint
├── llm\_handler.py         # Module to generate code using LLM
├── db\_utils.py            # Database utility functions
├── data\_cleaning.py       # Data cleaning helpers
├── safe\_exec.py           # Safe execution of generated code
├── mapping.py             # Helper functions (e.g., keyword mapping)
├── scraping.py            # Web scraping related code (if any)
├── view\_image.py          # Utility for image display/testing
│
├── sample-sales.csv       # Sample dataset CSV (gitignored)
├── scraped\_raw\_table\_1.csv # Raw scraped data CSV (gitignored)
├── scraped\_cleaned\_table\_1.csv # Cleaned scraped data CSV (gitignored)
├── chroma\_db/             # Vector DB or other data storage (gitignored)
│
└── notes.txt              # Notes file (gitignored)
└── questions.txt          # Questions for testing (gitignored)

````

---

## Basic Workflow

1. **User sends natural language questions** to the FastAPI endpoint.
2. **The API loads metadata and sample data** from the SQLite database.
3. **The LLM is prompted** (via AIPipe) with the questions, database schema, and sample data.
4. **LLM generates Python code** that queries the database, computes statistics, and creates plots as needed.
5. **The generated code is executed safely**, producing results and base64 encoded images.
6. **The API returns a structured JSON response** containing numeric answers and image data.
7. Clients can decode images and display them alongside numeric insights.

---
````markdown
## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/PranathiPamarti/LLM-Powered-Data-Analyst.git
   cd LLM-Powered-Data-Analyst
````

2. Create and activate a Python virtual environment:

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your AIPipe API token:

   ```env
   AIPIPE_TOKEN=your_token_here
   ```

5. Run the FastAPI application locally:

   ```bash
   uvicorn main:app --reload
   ```
