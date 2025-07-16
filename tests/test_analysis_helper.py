import pandas as pd
import pytest

from src import analysis_helper


@pytest.fixture
def sample_data():
    """Fixture to create a sample DataFrame for testing."""
    data = {
        "program": ["Computer Science", "Software Engineering", "Data Science"],
        "job_description": [
            "Python, Machine Learning, Algorithms, AI",
            "Java, Software Engineering, Algorithms, AI",
            "Python, Data Science, Algorithms, AI",
        ],
    }
    return pd.DataFrame(data)


def test_calculate_term_frequencies(sample_data):
    """Test for calculate_term_frequencies."""
    terms = ["Python", "Machine Learning", "Algorithms"]
    result = analysis_helper.calculate_term_frequencies(
        sample_data, "job_description", terms=terms
    )
    assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
    assert "Python" in result.columns, "Expected 'Python' column in the output"
    assert result["Python"].sum() > 0, "Expected non-zero frequency for 'Python'"


def test_calculate_document_frequencies(sample_data):
    """Test for calculate_document_frequencies."""
    terms = ["Python", "Machine Learning", "Algorithms"]
    result = analysis_helper.calculate_document_frequencies(
        sample_data, "job_description", terms=terms
    )
    assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
    assert "Python" in result.columns, "Expected 'Python' column in the output"
    assert result["Python"].sum() > 0, "Expected non-zero frequency for 'Python'"


def test_calculate_posting_counts_by_skills(sample_data):
    """Test for calculate_posting_counts_by_skills."""
    terms = ["Python", "Machine Learning", "Algorithms"]
    result = analysis_helper.calculate_posting_counts_by_skills(
        sample_data, "job_description", terms=terms
    )
    assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
    assert result.shape[0] > 0, "Expected rows in the output"
    assert result.shape[1] > 0, "Expected columns in the output"


def test_count_document_per_group_with_term_occurrences(sample_data):
    """Test for count_document_per_group_with_term_occurrences."""
    terms = ["Python", "Machine Learning", "Algorithms"]
    grouped_df = sample_data.groupby("program")
    result = analysis_helper.count_document_per_group_with_term_occurrences(
        grouped_df, "job_description", terms=terms
    )
    assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
    assert "Python" in result.columns, "Expected 'Python' column in the output"


def test_create_term_table(sample_data):
    """Test for create_term_table."""
    df = {
        "Python": {
            "Computer Science": {1: 1, 2: 0},
            "Software Engineering": {1: 1, 2: 0},
        }
    }
    result = analysis_helper.create_term_table(df, "Python")
    assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
    assert result.shape[0] == 2, "Expected 2 rows for the groups"
    assert result.shape[1] == 3, "Expected columns for occurrence counts"


def test_create_all_skill_tables(sample_data, tmp_path):
    """Test for create_all_skill_tables."""
    df = {
        "Python": {
            "Computer Science": {1: 1, 2: 0},
            "Software Engineering": {1: 1, 2: 0},
        }
    }
    file_path = tmp_path / "skills.xlsx"
    analysis_helper.create_all_skill_tables(df, file_path)
    assert file_path.exists(), f"File {file_path} should be created"


def test_count_document_per_group_with_total_term_occurrences(sample_data):
    """Test for count_document_per_group_with_total_term_occurrences."""
    terms = ["Python", "Machine Learning", "Algorithms"]
    grouped_df = sample_data.groupby("program")
    result = analysis_helper.count_document_per_group_with_total_term_occurrences(
        grouped_df, "job_description", terms=terms
    )
    assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
    assert result.shape[0] > 0, "Expected rows in the output"


def test_print_documents_with_max_occurrences(sample_data):
    """Test for print_documents_with_max_occurrences."""
    terms = ["Python", "Machine Learning", "Algorithms"]
    grouped_df = sample_data.groupby("program")
    # Use print and capture output in a string buffer
    import sys
    from io import StringIO

    captured_output = StringIO()
    sys.stdout = captured_output
    analysis_helper.print_documents_with_max_occurrences(
        grouped_df, "job_description", "program", terms=terms
    )
    sys.stdout = sys.__stdout__

    assert "Max occurrences" in captured_output.getvalue(), (
        "Expected output mentioning 'Max occurrences'"
    )


if __name__ == "__main__":
    pytest.main()
