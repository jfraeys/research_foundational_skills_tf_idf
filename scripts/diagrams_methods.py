from diagrams import Cluster, Diagram, Edge
from diagrams.programming.flowchart import (Action, Document, Merge,
                                            MultipleDocuments)

graph_attr = {
    "bgcolor": "transparent"
}

with Diagram("exploration.py pre-processing", show=False, graph_attr=graph_attr):
    input_file = MultipleDocuments("Raw data")
    output_data_file = MultipleDocuments("processed data")
    output_datatype = MultipleDocuments("processed data")
    
    with Cluster("processing actions"):
        first_action = Action("Removing columns")
        last_action = Action("Process")
        first_action - Edge(color="black") \
        >> Action("Editing columns names") >> Edge(color="black") \
        >> Action("Convert column types") >> Edge(color="black") \
        >> last_action

        
    input_file >> first_action
    last_action >> [
        output_data_file,
        output_datatype,
        Action("Exploration")
    ]