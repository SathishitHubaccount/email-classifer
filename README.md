Email Classification & Entity Masking Service
This project implements a pipeline that:

Classifies emails into one of four categories:

Incident

Request

Change

Problem

Detects and masks sensitive entities such as names, emails, and other personal information within the email text.

Features
Uses state-of-the-art Named Entity Recognition (NER) to identify entities and their positions.

Masks detected entities in the email text with descriptive placeholders (e.g., [PERSON], [EMAIL]).

Classifies the input email into predefined categories using a machine learning classification model.

Returns a detailed JSON response including original text, masked email, list of masked entities with positions, and email category.

Input / Output Format
Input JSON
json
Copy
Edit
{
  "input_email_body": "string containing the email"
}
Output JSON
json
Copy
Edit
{
  "input_email_body": "string containing the email",
  "list_of_masked_entities": [
    {
      "position": [start_index, end_index],
      "classification": "entity_type",
      "entity": "original_entity_value"
    }
    // Additional entities ...
  ],
  "masked_email": "string containing the masked email",
  "category_of_the_email": "string containing the class"
}
How It Works
Named Entity Recognition (NER):
The system scans the email for named entities like person names, email addresses, locations, and masks them with placeholders.

Entity Masking:
Replaces each detected entity in the email text with a tag corresponding to the entity type, ensuring privacy and anonymization.

Email Classification:
Classifies the email into one of the four categories based on its content.

Response:
Returns the original email, masked email, the list of masked entities with their positions and classifications, and the predicted email category.

Requirements

fastapi
uvicorn
transformers
torch
pydantic
pandas
numpy

