# Research Project: Evaluating the Change of Foundational Skills in Tech Co-op Job Postings

## Overview
This project explores foundational skills in job postings for Engineering and IT roles at a single Ontario university. The research utilizes natural language processing (NLP) techniques, including TF-IDF, to analyze postings from 2013 to 2022, identifying trends and patterns in skill requirements before and after COVID-19.

## Objectives
- Analyze foundational skills from job postings across a decade.
- Identify trends in skill demand over time.
- Compare skill requirements pre- and post-COVID.
- Explore the potential of TF-IDF and other NLP techniques in foundational skills research.

## Methodology
1. **Data Collection:**
   - Job postings were gathered and preprocessed to extract skill-related content.
   - Time periods were divided into pre- and post-COVID for comparative analysis.

2. **NLP Techniques:**
   - **TF-IDF:** To quantify term importance across job postings.
   - **Visualization:** Trend analysis using tables, charts, and arrows to highlight significant findings.

3. **Analysis:**
   - Extracted key skills from postings.
   - Highlighted evolving demands for foundational skills.

## Results
- Skills like collaboration, communication, and adaptability remain consistently in demand.
- Technical skills have seen varied demand, with some skills like programming languages increasing post-COVID.
- Emerging trends reflect a growing emphasis on remote work adaptability.

## Tools and Technologies
- **Python:** For data analysis and NLP tasks.
- **Jupyter Notebooks:** For iterative development and visualization.
- **Libraries:** pandas, scikit-learn, nltk, matplotlib, seaborn.

## Future Work
- Expand the analysis to other universities or regions.
- Integrate advanced NLP techniques like word embeddings for deeper insights.
- Develop a dynamic dashboard for real-time skill trend tracking.

## Repository Structure
```
|-- credentials/           # API keys or service credentials (secured)
|-- data/                  # Raw and processed data files
|-- docs/                  # Project documentation and references
|-- notebooks/             # Jupyter notebooks for analysis
|-- references/            # Papers, articles, and related research
|-- reports/               # Outputs, visualizations, and summaries
|-- src/                   # Python scripts for data preprocessing and analysis
|-- secrets/               # Secure storage for sensitive data
|-- environment.yml        # Conda environment configuration
|-- Makefile               # Automation for setup and tasks
|-- README.md              # Project documentation
```

## Makefile Commands
- **Set up the environment:**
  ```bash
  make setup_env
  ```
  Creates a new Conda environment from `environment.yml` and installs a Jupyter kernel for the environment.

- **Update the environment:**
   ```bash
   make update_env
   ```
   Update the existing Conda environment using `environment.yml`.

- **Clean the environment:**
  ```bash
  make clean_env
  ```
  Removes the Conda environment.

- **Help:**
  ```bash
  make help
  ```
  Shows a list of available commands and their descriptions.

## How to Run
1. **Set up the environment:**
   ```bash
   make setup_env
   conda activate soft-skills-env
   ```

2. **Run the notebooks:**
   Open the `notebooks/` directory and execute Jupyter notebooks for specific analyses.

3. **View results:**
   Processed outputs and visualizations are available in the `reports/` directory.

## Contributions
- [Jeremie Fraeys]: Principal Investigator and Developer

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- The examination committee for their valuable feedback.
- Open-source contributors to Python libraries used in this project.

For more details, refer to the [thesis document available on the University of Guelph Atrium Repository](https://atrium.lib.uoguelph.ca/items/79f32fa4-2e18-4c50-90ac-39b773a36fd4).

For questions or feedback, contact [mailto:jfraeysd@uoguelph.ca].


