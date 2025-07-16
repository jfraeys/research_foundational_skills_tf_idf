from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

tqdm.pandas()


# def calculate_term_frequencies(
#     groupby_df: pd.DataFrame,
#     text_column: str,
#     terms: list[str] = [],
#     ngram_range: tuple = (1, 2),
# ):
#     # Initialize Counter
#     term_frequencies: Counter = Counter()
#     total_terms_in_group = 0

#     for document in groupby_df[text_column].tolist():
#         # Tokenize the document into words
#         words = document.split()

#         # Create n-grams based on the specified ngram_range
#         ngrams = [
#             tuple(words[i : i + n])
#             for n in range(ngram_range[0], ngram_range[1] + 1)
#             for i in range(len(words) - n + 1)
#         ]

#         # Total number of terms in the document
#         total_terms_in_group += len(words)

#         # Update counts based on the occurrences in the document
#         for term in terms:
#             term_count = ngrams.count(tuple(term.split()))
#             term_frequencies[term] += term_count

#     term_frequencies = {
#         term: term_frequencies[term] / total_terms_in_group for term in terms
#     }

#     # Convert the Counter to a DataFrame
#     term_frequencies_df = pd.DataFrame(
#         list(term_frequencies.items()), columns=["keyword", "term_frequency"]
#     )

#     # Pivot the DataFrame to get term frequencies as a pivot table
#     term_frequencies_pivot = term_frequencies_df.pivot_table(
#         index=None, columns="keyword", values="term_frequency", fill_value=0
#     )
#     term_frequencies_pivot.index.name = None


#     return term_frequencies_pivot
def calculate_term_frequencies(
    groupby_df: pd.DataFrame,
    text_column: str,
    terms: list[str] = [],
    ngram_range: tuple = (1, 2),
):
    """
    Calculate term frequencies for specified terms in a text column.

    Args:
        groupby_df (pd.DataFrame): Input DataFrame.
        text_column (str): Column containing text data.
        terms (list[str]): List of terms to calculate frequencies for.
        ngram_range (tuple): Range of n-grams (min_n, max_n).

    Returns:
        pd.DataFrame: Pivot table of term frequencies.
    """
    # Create CountVectorizer with specified n-gram range
    vectorizer = CountVectorizer(vocabulary=terms, ngram_range=ngram_range)

    # Transform the text column into a term-document matrix
    term_matrix = vectorizer.fit_transform(groupby_df[text_column])

    # Sum term frequencies across all documents
    term_frequencies = term_matrix.sum(axis=0)

    # Normalize term frequencies by the total number of terms
    total_terms_in_group = term_matrix.sum()
    normalized_frequencies = (term_frequencies / total_terms_in_group).A1

    # Create a DataFrame for term frequencies
    term_frequencies_df = pd.DataFrame(
        {
            "keyword": vectorizer.get_feature_names_out(),
            "term_frequency": normalized_frequencies,
        }
    )

    # Pivot the DataFrame
    term_frequencies_pivot = term_frequencies_df.pivot_table(
        index=None, columns="keyword", values="term_frequency", fill_value=0
    )
    term_frequencies_pivot.index.name = None

    return term_frequencies_pivot


# def calculate_document_frequencies(
#     groupby_df: pd.DataFrame, text_column: str, terms: list, ngram_range: tuple = (1, 2)
# ):
#     # Initialize dictionary to store document frequencies for each term
#     document_frequencies: dict[str, float] = {term: 0 for term in terms}

#     # Count the number of documents in the group of dataframes
#     num_documents = len(groupby_df)

#     # Iterate through each document in the group
#     for document in groupby_df[text_column].tolist():
#         # Tokenize the document into n-grams based on the specified ngram_range
#         words = document.split()
#         ngrams = [
#             tuple(words[i : i + n])
#             for n in range(ngram_range[0], ngram_range[1] + 1)
#             for i in range(len(words) - n + 1)
#         ]

#         # Check if each term is present in the document and increment count if found at least once
#         for term in terms:
#             term_tuple = set(
#                 term.split()
#             )  # Convert term to a set for efficient membership checking
#             if any(term_tuple == set(ngram[: len(term_tuple)]) for ngram in ngrams):
#                 document_frequencies[term] += 1

#     # Divide the counts by the number of documents to get document frequencies
#     document_frequencies = {
#         term: count / num_documents for term, count in document_frequencies.items()
#     }

#     # Convert the document frequencies dictionary to a DataFrame
#     document_frequencies_df = pd.DataFrame(
#         list(document_frequencies.items()), columns=["keyword", "document_frequency"]
#     )

#     # Pivot the DataFrame to get document frequencies as a pivot table
#     document_frequencies_pivot = document_frequencies_df.pivot_table(
#         index=None, columns="keyword", values="document_frequency", fill_value=0
#     )
#     document_frequencies_pivot.columns.name = None
#     document_frequencies_pivot.index.name = None


#     return document_frequencies_pivot
def calculate_document_frequencies(
    groupby_df: pd.DataFrame,
    text_column: str,
    terms: list[str],
    ngram_range: tuple = (1, 2),
):
    """
    Calculate document frequencies for specified terms in a text column.

    Args:
        groupby_df (pd.DataFrame): Input DataFrame.
        text_column (str): Column containing text data.
        terms (list[str]): List of terms to calculate document frequencies for.
        ngram_range (tuple): Range of n-grams (min_n, max_n).

    Returns:
        pd.DataFrame: Pivot table of document frequencies.
    """
    # Create CountVectorizer with binary=True to count document occurrences (not term frequency)
    vectorizer = CountVectorizer(vocabulary=terms, ngram_range=ngram_range, binary=True)

    # Transform the text column into a binary term-document matrix
    term_matrix = vectorizer.fit_transform(groupby_df[text_column])

    # Sum binary presence across all documents
    document_frequencies = term_matrix.sum(axis=0)

    # Normalize document frequencies by the total number of documents
    num_documents = len(groupby_df)
    normalized_frequencies = (document_frequencies / num_documents).A1

    # Create a DataFrame for document frequencies
    document_frequencies_df = pd.DataFrame(
        {
            "keyword": vectorizer.get_feature_names_out(),
            "document_frequency": normalized_frequencies,
        }
    )

    # Pivot the DataFrame
    document_frequencies_pivot = document_frequencies_df.pivot_table(
        index=None, columns="keyword", values="document_frequency", fill_value=0
    )
    document_frequencies_pivot.index.name = None

    return document_frequencies_pivot


# def calculate_posting_counts_by_skills(
#     groupby_df: pd.DataFrame, text_column: str, terms: list, ngram_range: tuple = (1, 2)
# ):
#     # Initialize a dictionary to store posting counts for each program and number of foundational skills
#     program_posting_counts = {}

#     # Get the total number of programs for progress tracking
#     total_programs = len(groupby_df)

#     # Iterate through each group in the grouped DataFrame
#     for group_name, group_data in tqdm(
#         groupby_df, total=total_programs, desc="Calculating posting counts"
#     ):
#         # Initialize a dictionary to store posting counts for each number of foundational skills
#         posting_counts: dict[int, int] = {}

#         # Iterate through each document in the program group
#         for document in group_data[text_column].tolist():
#             # Tokenize the document into n-grams based on the specified ngram_range
#             words = document.split()
#             ngrams = [
#                 tuple(words[i : i + n])
#                 for n in range(ngram_range[0], ngram_range[1] + 1)
#                 for i in range(len(words) - n + 1)
#             ]

#             # Initialize the count of foundational skills in the document
#             num_skills = 0

#             # Check if each term is present in the document and increment count if found at least once
#             for term in terms:
#                 term_tuple = set(
#                     term.split()
#                 )  # Convert term to a set for efficient membership checking
#                 if any(term_tuple == set(ngram[: len(term_tuple)]) for ngram in ngrams):
#                     num_skills += 1

#             # Increment the count of postings containing the number of foundational skills
#             if num_skills in posting_counts:
#                 posting_counts[num_skills] += 1
#             else:
#                 posting_counts[num_skills] = 1

#         # Store the program and its posting counts by number of foundational skills in the dictionary
#         program_posting_counts[group_name] = posting_counts

#     # Convert the dictionary to a DataFrame
#     df = pd.DataFrame.from_dict(program_posting_counts, orient="index").fillna(0)

#     # Convert the DataFrame to integer type
#     df = df.astype(int)


#     return df
def calculate_posting_counts_by_skills(
    groupby_df: pd.DataFrame,
    text_column: str,
    terms: list[str],
    ngram_range: tuple = (1, 2),
):
    """
    Calculate posting counts for each group based on the number of foundational skills.

    Args:
        groupby_df (pd.DataFrame): Input DataFrame grouped by a column (e.g., program name).
        text_column (str): Column containing text data.
        terms (list[str]): List of foundational skills (terms).
        ngram_range (tuple): Range of n-grams (min_n, max_n).

    Returns:
        pd.DataFrame: DataFrame of posting counts by group and number of skills.
    """
    # Create CountVectorizer to find terms in the text
    vectorizer = CountVectorizer(vocabulary=terms, ngram_range=ngram_range, binary=True)

    # Initialize an empty dictionary to store results
    program_posting_counts = {}

    # Iterate through groups and calculate counts
    for group_name, group_data in tqdm(
        groupby_df, total=len(groupby_df), desc="Calculating posting counts"
    ):
        # Transform the text column into a binary term-document matrix
        term_matrix = vectorizer.fit_transform(group_data[text_column])

        # Count the number of skills (non-zero elements per row)
        num_skills_per_doc = term_matrix.sum(axis=1).A1

        # Count the frequency of each number of skills in the group
        posting_counts = pd.Series(num_skills_per_doc).value_counts().to_dict()

        # Store the results for the group
        program_posting_counts[group_name] = posting_counts

    # Convert the dictionary to a DataFrame
    posting_counts_df = pd.DataFrame.from_dict(
        program_posting_counts, orient="index"
    ).fillna(0)

    # Convert to integer type
    posting_counts_df = posting_counts_df.astype(int)

    return posting_counts_df


# def count_document_per_group_with_term_occurrences(
#     grouped_df: pd.DataFrame, text_column: str, terms: list, ngram_range: tuple = (1, 2)
# ):
#     # Initialize a dictionary to store document counts per group with term occurrences
#     group_document_counts = {}

#     # Get the total number of groups for progress tracking
#     total_groups = len(grouped_df)

#     # Iterate through each group in the grouped DataFrame
#     for group, group_data in tqdm(
#         grouped_df, total=total_groups, desc="Counting documents per group"
#     ):
#         # Initialize a dictionary to store term occurrences per document for the group
#         document_term_occurrences = {
#             document_id: {term: 0 for term in terms}
#             for document_id, _ in group_data[text_column].items()
#         }

#         # Iterate through each document in the group
#         for document_id, document_text in group_data[text_column].items():
#             # Tokenize the document into n-grams based on the specified ngram_range
#             words = document_text.split()
#             ngrams = [
#                 tuple(words[i : i + n])
#                 for n in range(ngram_range[0], ngram_range[1] + 1)
#                 for i in range(len(words) - n + 1)
#             ]

#             # Count the occurrences of each term in the document
#             for term in terms:
#                 term_count = ngrams.count(tuple(term.split()))
#                 document_term_occurrences[document_id][term] = term_count

#         # Count the number of documents per group that contain 1, 2, 3, ... occurrences of each term
#         max_term_occurrences = max(
#             max(doc_counts.values(), default=0)
#             for doc_counts in document_term_occurrences.values()
#         )
#         group_document_counts[group] = {
#             term: {
#                 count: sum(
#                     1
#                     for doc_counts in document_term_occurrences.values()
#                     if doc_counts[term] == count
#                 )
#                 for count in range(max_term_occurrences + 1)
#             }
#             for term in terms
#         }

#     # Convert the dictionary to a DataFrame
#     df_counts = pd.DataFrame.from_dict(group_document_counts, orient="index")


#     return df_counts
def count_document_per_group_with_term_occurrences(
    grouped_df: pd.DataFrame,
    text_column: str,
    terms: list[str],
    ngram_range: tuple = (1, 2),
):
    """
    Count the number of documents per group with specific term occurrences.

    Args:
        grouped_df (pd.DataFrame): Input DataFrame grouped by a column.
        text_column (str): Column containing text data.
        terms (list[str]): List of terms to search for.
        ngram_range (tuple): Range of n-grams (min_n, max_n).

    Returns:
        pd.DataFrame: DataFrame with groups as rows, terms as columns,
                      and term occurrence counts as nested dictionaries.
    """
    # Create CountVectorizer to find terms in the text
    vectorizer = CountVectorizer(vocabulary=terms, ngram_range=ngram_range)

    # Initialize a dictionary to store results
    group_document_counts = {}

    # Iterate through groups
    for group_name, group_data in tqdm(
        grouped_df, total=len(grouped_df), desc="Counting term occurrences per group"
    ):
        # Transform the text column into a term-document matrix
        term_matrix = vectorizer.fit_transform(group_data[text_column])

        # Convert to a DataFrame for easier manipulation
        term_df = pd.DataFrame(
            term_matrix.toarray(), columns=vectorizer.get_feature_names_out()
        )

        # Count occurrences of each term per document
        term_occurrences = term_df.apply(pd.Series.value_counts, axis=0).fillna(0)

        # Flatten the counts into a dictionary with nested term counts
        group_term_counts = {
            term: term_occurrences[term].to_dict() for term in term_df.columns
        }

        # Store the group's results
        group_document_counts[group_name] = group_term_counts

    # Convert the dictionary to a DataFrame
    df_counts = pd.DataFrame.from_dict(group_document_counts, orient="index")

    return df_counts


# def create_term_table(df: dict, term: str):
#     # Check if the term is a valid key in the dictionary
#     if term not in df:
#         raise ValueError(
#             "Invalid term: '{}' is not a key in the dictionary.".format(term)
#         )

#     # Extract the groups (index) and the data
#     groups: list[str] = list(df[term].keys())  # Assuming the keys represent the index
#     occurrences = df[term].values

#     # Find the length of the longest diet without trailing zeros
#     longest_length = 0
#     for diet in occurrences:
#         non_zero_indices = [i for i, count in diet.items() if count != 0]
#         if non_zero_indices:
#             longest_length = max(longest_length, max(non_zero_indices) + 1)

#     # Create a DataFrame to hold the occurrences
#     term_df = pd.DataFrame(0, index=groups, columns=range(longest_length))

#     # Iterate over each group's term data
#     for occurrence, diet in zip(groups, occurrences):
#         non_zero_indices = [i for i, count in diet.items() if count != 0]
#         for index in non_zero_indices:
#             term_df.loc[occurrence, index] = diet[index]


#     return term_df
def create_term_table(df: dict, term: str) -> pd.DataFrame:
    """
    Create a term table DataFrame from a dictionary of term occurrences.

    Args:
        df (dict): Dictionary where keys are terms, and values are dictionaries
                   of group occurrences.
        term (str): The term to extract and create the table for.

    Returns:
        pd.DataFrame: DataFrame where rows are groups, columns are term occurrence counts,
                      and cell values represent the number of documents with those occurrences.
    """
    # Check if the term is a valid key in the dictionary
    if term not in df:
        raise ValueError(f"Invalid term: '{term}' is not a key in the dictionary.")

    # Extract group-level data for the specified term
    term_data = df[term]

    # Determine the maximum term occurrence count across all groups
    max_occurrence = (
        max(max(diet.keys(), default=0) for diet in term_data.values) + 1
    )  # +1 to account for inclusive range

    # Create the DataFrame with groups as rows and occurrence counts as columns
    term_df = (
        pd.DataFrame.from_dict(
            {
                group: {
                    count: term_data[group].get(count, 0)
                    for count in range(max_occurrence)
                }
                for group in term_data
            },
            orient="index",
        )
        .fillna(0)
        .astype(int)
    )

    return term_df


def create_all_skill_tables(df: dict, filename: str) -> None:
    """
    Create an Excel file with a sheet for each skill term, containing its occurrence table.

    Args:
        df (dict): A dictionary where keys are terms and values are dictionaries
                   of group occurrences.
        filename (str): The name of the Excel file to save the tables to.

    Returns:
        None
    """
    # Use Pandas Excel writer with XlsxWriter engine
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        # Iterate over each term and its data in the dictionary
        for term, data in df.items():
            # Create term table for the current term
            term_df = create_term_table(df, term)

            # Write the DataFrame to an Excel sheet named after the term
            term_df.to_excel(writer, sheet_name=term, index=True, header=True)

    print(f"Skill tables have been written to {filename}.")


# def count_total_term_occurrences_per_group(grouped_df: pd.DataFrame, text_column: str, terms: list, ngram_range: tuple = (1, 2)):
#     # Initialize a dictionary to store total term occurrences per group
#     total_term_occurrences_per_group = {}

#     # Get the total number of groups for progress tracking
#     total_groups = len(grouped_df)

#     # Iterate through each group in the grouped DataFrame
#     for group, group_data in tqdm(grouped_df.iterrows(), total=total_groups, desc="Counting total term occurrences per group"):
#         # Initialize a dictionary to store term occurrences for the group
#         term_occurrences = {}

#         # Iterate through each document in the group
#         for document_text in group_data[text_column]:
#             # Tokenize the document into n-grams based on the specified ngram_range
#             words = document_text.split()
#             ngrams = [tuple(words[i:i+n]) for n in range(ngram_range[0], ngram_range[1] + 1) for i in range(len(words) - n + 1)]

#             # Count the occurrences of each term in the document
#             for term in terms:
#                 term_count = ngrams.count(tuple(term.split()))
#                 term_occurrences[term] = term_occurrences.get(term, 0) + term_count

#         # Get the total occurrence count for all terms in the group
#         total_occurrences = sum(term_occurrences.values())

#         # Store the total term occurrences for the group
#         total_term_occurrences_per_group[group] = total_occurrences

#     # Convert the dictionary to a DataFrame
#     df_total_term_occurrences = pd.DataFrame.from_dict(total_term_occurrences_per_group, orient='index', columns=["Total Occurrences"])

#     return df_total_term_occurrences


def count_document_per_group_with_total_term_occurrences(
    grouped_df: pd.DataFrame, text_column: str, terms: list, ngram_range: tuple = (1, 2)
):
    # Initialize a dictionary to store document counts per group with total term occurrences
    group_document_counts = {}

    # Get the total number of groups for progress tracking
    total_groups = len(grouped_df.groups)

    # Define a function to count total term occurrences for each document in a group
    def count_total_term_occurrences(document_text):
        total_terms = 0

        # Tokenize the document into n-grams based on the specified ngram_range
        words = document_text.split()
        ngrams = [
            tuple(words[i : i + n])
            for n in range(ngram_range[0], ngram_range[1] + 1)
            for i in range(len(words) - n + 1)
        ]

        # Count the total occurrences of any term in the document
        total_terms = sum(ngrams.count(tuple(term.split())) for term in terms)

        return total_terms

    # Apply the function to each document in each group and store the results in the dictionary
    for group_name, group_data in tqdm(
        grouped_df,
        total=total_groups,
        desc="Counting documents per group with total term occurrences",
    ):
        total_term_occurrences = group_data[text_column].apply(
            count_total_term_occurrences
        )
        group_document_counts[group_name] = total_term_occurrences

    # Convert the dictionary to a DataFrame
    df_total_term_occurrences = pd.DataFrame.from_dict(group_document_counts)

    # Count the number of documents in each group that have 1, 2, 3, ..., terms
    occurrence_counts = (
        df_total_term_occurrences.apply(lambda x: x.value_counts())
        .fillna(0)
        .astype(int)
        .T
    )

    occurrence_counts.columns = occurrence_counts.columns.astype(int)
    column_range = range(
        occurrence_counts.columns.min(), occurrence_counts.columns.max() + 1
    )
    occurrence_counts = occurrence_counts.reindex(columns=column_range, fill_value=0)

    return occurrence_counts


def print_documents_with_max_occurrences(
    grouped_df: pd.DataFrame,
    text_column: str,
    doc_id_column: str,
    terms: list,
    ngram_range: tuple = (1, 2),
):
    # Initialize a dictionary to store documents with the greatest occurrences of each term per group
    group_documents_max_occurrences = {}

    # Get the total number of groups for progress tracking
    total_groups = len(grouped_df)

    # Initialize a dictionary to store term occurrences per document for the group
    document_term_occurrences = {
        group: {
            document_id: {term: 0 for term in terms} for document_id in group_data.index
        }
        for group, group_data in grouped_df
    }

    # Iterate through each group in the grouped DataFrame
    for group, group_data in tqdm(
        grouped_df, total=total_groups, desc="Finding documents with max occurrences"
    ):
        # Iterate through each document in the group
        for document_id in group_data.index:
            # Tokenize the document into n-grams based on the specified ngram_range
            words = group_data.loc[document_id, text_column].split()
            ngrams = [
                tuple(words[i : i + n])
                for n in range(ngram_range[0], ngram_range[1] + 1)
                for i in range(len(words) - n + 1)
            ]

            # Count the occurrences of each term in the document
            for term in terms:
                term_count = ngrams.count(tuple(term.split()))
                document_term_occurrences[group][document_id][term] = term_count

        # Find the document with the greatest number of occurrences for each term in this group
        max_occurrences_document = {
            term: max(
                document_term_occurrences[group].keys(),
                key=lambda doc_id: document_term_occurrences[group][doc_id][term],
            )
            for term in terms
        }

        # Store the document with the greatest number of occurrences for each term in the group
        group_documents_max_occurrences[group] = max_occurrences_document

    # Print the documents with the greatest occurrences of each term for each group
    for group, max_occurrences_document in group_documents_max_occurrences.items():
        print(f"Group: {group}")
        group_data = grouped_df.get_group(group)
        for term, document_id in max_occurrences_document.items():
            doc_id_value = group_data.loc[document_id, doc_id_column]
            document_text = group_data.loc[document_id, text_column]
            occurrence = document_term_occurrences[group][document_id][term]
            print(
                f"Term: {term}, Doc ID: {doc_id_value}, Occurrence: {occurrence}, Text: {document_text}"
            )


def plot_line_graph(posting_counts_df: pd.DataFrame, title: str):
    # Create a line graph for each program
    for program in posting_counts_df.index:
        plt.plot(
            posting_counts_df.columns, posting_counts_df.loc[program], label=program
        )

    # Add labels and title
    plt.xlabel("Number of Foundational Skills")
    plt.ylabel("Number of Postings")
    plt.title(title)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def plot_posting_counts(document_counts_tf):
    # Print the skills with their indices
    print("Skills with their indices:")
    for index, skill in enumerate(document_counts_tf.columns):
        print(f"Index {index}: {skill}")

    # Iterate over each program in the DataFrame
    for program in document_counts_tf.index:
        # Extract the dictionary of skill occurrences for the current program
        skill_occurrences = document_counts_tf.loc[program]

        # Convert the dictionary into a DataFrame with columns for each skill count
        program_df = pd.json_normalize(skill_occurrences).fillna(0)

        # Plot the posting counts for the current program with the specified title
        plot_line_graph(program_df, title=f"Posting Counts for Program in {program}")


def calculate_term_counts(
    groupby_df: pd.DataFrame, text_column: str, terms: list, ngram_range: tuple = (1, 2)
):
    # Initialize Counter
    term_counts = Counter({term: 0 for term in terms})

    for document in groupby_df[text_column].tolist():
        # Tokenize the document into words
        words = document.split()

        # Create n-grams based on the specified ngram_range
        ngrams = [
            tuple(words[i : i + n])
            for n in range(ngram_range[0], ngram_range[1] + 1)
            for i in range(len(words) - n + 1)
        ]

        # Update counts based on the occurrences in the document
        for term in set(terms):
            term_count = ngrams.count(tuple(term.split()))
            term_counts[term] += term_count

    # Convert the Counter to a DataFrame
    term_counts_df = pd.DataFrame(
        list(term_counts.items()), columns=["keyword", "term_count"]
    )

    # Pivot the DataFrame to get term counts as a pivot table
    term_counts_pivot = term_counts_df.pivot_table(
        index=None, columns="keyword", values="term_count", fill_value=0
    )

    return term_counts_pivot
