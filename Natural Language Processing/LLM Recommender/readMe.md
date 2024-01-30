# Book Recommender System Using OpenAI and LangChain

## Overview
This project develops a sophisticated Book Recommender System leveraging OpenAI's language models and the LangChain library. It's designed to suggest books based on user preferences, utilizing advanced natural language processing techniques.


## Data Acquisition
The heart of the system lies in its dataset:

```python
import pandas as pd
import numpy as np

# Loading the dataset
books = pd.read_csv('data/BooksDatasetClean.csv')
```
This dataset, ideally sourced from a comprehensive collection like Kaggle, includes critical information such as titles, authors, descriptions, categories, and publication details.



## Data Preprocessing
The dataset I used for this project, from Kaggle, titled "Books Dataset" by Elvin Rustam, is a comprehensive collection of books, encompassing various details such as titles, authors, descriptions, categories, publishers, prices, and publication dates. The dataset is extensive, covering a broad spectrum of literature, making it a valuable resource for projects involving book recommendations, literary analysis, or data-driven studies in publishing trends (https://www.kaggle.com/datasets/elvinrustam/books-dataset?resource=download)

The data is already cleaned to eliminate symbols and have uniform capitalization. Additonally, I got rid of all books with no descriptions. This left me with roughly 70,000 books to train on.

![Sample Data](markdown_photos/sample_data.png)

In order to simply the data for embeddings later on, I combine all Title, Author, Description, and Category, into one column for simplicity and efficiency.

![Sample Data](markdown_photos/all_text_data.png)


## Embedding and Vectorization
The system uses OpenAI's model for embedding textual data, transforming the textual information of each book into a numerical format:

```python
# Function to get embeddings
def get_embeddings(text_list, model="text-embedding-ada-002"):
    processed_texts = [text.replace("\n", " ") for text in text_list]
    response = client.embeddings.create(input=processed_texts, model=model)
    return [item.embedding for item in response.data]
```

The selection of the "text-embedding-ada-002" model for generating text embeddings is influenced by several key considerations. Larger models, such as this one, typically offer more advanced language comprehension capabilities. This leads to richer and more nuanced embeddings, capturing the complexities of the text more effectively.


This transformation is crucial for enabling the recommendation algorithm to understand and compare book content effectively.

## LangChain Integration
**LangChain** is a toolkit for building applications with language models, enabling complex tasks like text generation and information retrieval. **LanceDB** complements it by efficiently managing and querying textual data. 

Their use in my project enhances the capabilities of a book recommender system, offering advanced text processing and efficient data handling.
Integrating LangChain and LanceDB, the system creates a retrieval-based question-answering system. I used the following the libaries:

```python
import lancedb
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import LanceDB # Utilizes LanceDB for storing and retrieving the vector representations (embeddings) of the books.
from langchain.chains import RetrievalQA
```

**langchain.chains.RetrievalQA**: Part of LangChain, this module is used to build a retrieval-based question-answering system, enabling the book recommender to understand user queries and fetch relevant book recommendations based on the embeddings.
This setup enables the system to understand user queries about book preferences and search through the embedded book dataset to find relevant recommendations.

## Recommender System in Action
The recommender system processes user queries, searching the database for books that match the query's intent and context:

```python
query = "I'm looking for a mystery novel set in Victorian England."
docs = docsearch.similarity_search(query, k=3)
```
The output is a list of recommended books with relevant details, tailored to the user's preferences.

## LLM Chain
An LLM (Large Language Model) Chain is a sequence of operations combining the power of large-scale language models with structured querying to deliver accurate and relevant book recommendations. The chain I implemented uses a PromptTemplate to guide the language model with specific instructions for book recommendations, ensuring the output is structured and directly addresses user queries. This method leverages the language model's understanding of text to match books to users' tastes, as exemplified below. My choice of this approach allows for dynamic and context-aware recommendations, making it a powerful tool in the domain of personalized book discovery.

### Technical Summary of LLM Chain Implementation

In this project, the `LLMChain` from the `langchain` library is used to create an interactive book recommendation system. The `PromptTemplate` class structures the input to the language model, crafting a conversation-like exchange where the model plays the role of a recommender:

```python
from langchain.prompts import PromptTemplate
template = """You are a book recommender system that help users to find books that match their preferences. 
For each question, suggest three books, with a short description of the plot and the reason why the user might like it.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Question: {question}
Your response: """
"  # A detailed prompt template for the LLM.
PROMPT = PromptTemplate(
    template=template, input_variables=["question"])
```

The `LLMChain` is then instantiated with the `OpenAI` class from `langchain_openai`. This chain encapsulates the logic to interact with OpenAI's powerful language models:

```python
from langchain.chains import LLMChain
from langchain_openai import OpenAI

#model is gpt-3.5-turbo-instruct
llm = OpenAI(openai_api_key=api_key)
chain_type_kwargs = {"prompt": PROMPT}


qa = RetrievalQA.from_chain_type(llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)
```

**RetrievalQA** in the `langchain` library is a system designed for question-answering using a retrieval-based approach. It retrieves relevant documents or data to answer a query, and then uses a language model (LLM) to generate a response based on this information.
`RetrievalQA.from_chain_type` is being used to create an instance of this system with specific configurations. This approach is beneficial for tasks where direct answers may not be readily available within the LLM's training data, requiring external sources (like documents retrieved by docsearch) for context and accuracy. The system then synthesizes this information to provide a comprehensive response, making it ideal for applications needing detailed or contextually nuanced answers.

`docsearch` is an instance of a document search system or a database interface that is capable of searching through a collection of documents (like articles, books, reports, etc.).
`docsearch.as_retriever()` converts the docsearch instance into a format that the RetrievalQA system can use as a retriever. This allows RetrievalQA to query docsearch for relevant documents based on the input question.

When a query is made, RetrievalQA uses the retriever to find documents related to the query's topic.
It then uses the language model (llm) to process these documents and the query to generate a coherent and contextually relevant answer.

The choice of `OpenAI` integration allows for the leveraging of GPT-3's deep understanding of language nuances, enabling the system to generate personalized book suggestions based on a user's query, as seen here:

```python
query = "I'm looking for an action book with animals, any suggestions?"
result = qa({'query':query})
```

This implementation details why `langchain` and `OpenAI` were chosen—maximizing the ability to provide nuanced and contextually relevant book recommendations.

## Experiment 1: Basic Query

**Prompt 1**: "Can you recommend a book about a girl who falls in love with a vampire?"

**Results 1**:
| Book Title            | Author               | Description                                                                                                                                                                         |
|-----------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Twilight              | Stephenie Meyer      | A classic love story between Bella, a human girl, and Edward, a vampire. The novel follows Bella as she navigates her new, mysterious romance.                                       |
| Blood and Chocolate   | Annette Curtis Klause| A dark and thrilling tale of a teenage werewolf who falls in love with a human boy, while struggling to keep her true identity a secret.                                             |
| Vampire Academy       | Richelle Mead        | Follows Rose Hathaway, a half-vampire, as she trains to protect her best friend Lissa, a royal vampire. The story includes forbidden romances and the challenges of keeping them hidden. |





**Prompt 2**: "can you recommend books about utopian usurps from young adults who try to change the world for the better and fight?"

**Results 2**:

| Book Title        | Author            | Description                                                                                                                                           |
|-------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| The Hunger Games  | Suzanne Collins   | A dystopian novel featuring Katniss Everdeen, a 16-year-old fighting against the oppressive government of Panem to protect her family and bring justice.|
| Divergent         | Veronica Roth     | Set in a futuristic society divided into factions, 16-year-old Tris Prior discovers she is "divergent" and must fight a corrupt government for fairness. |
| The Maze Runner   | James Dashner     | A group of teenage boys trapped in a dangerous maze work together to escape and overthrow the oppressive organization behind their predicament.          |




**Prompt 3**: "Can you recommend a fiction book about politics?"

**Results 3**:

| Book Title          | Author            | Description                                                                                                                          |
|---------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| The West Wing       | Aaron Sorkin      | A novel that explores the intricacies of political life, focusing on the personal and professional dynamics within the White House. |
| All the King's Men  | Robert Penn Warren| This book delves into the life of a populist politician, examining the moral implications of power and corruption in politics.       |
| The Ides of March   | Thornton Wilder   | A novel set in ancient Rome, revolving around the events leading up to the assassination of Julius Caesar, highlighting political intrigue and personal ambition. |


## Experiment 2: User Preferences

We can add additional context to the prompt of our users perferences to get a more nuanced answer.

We will adjust the prompt as such:

```python
template = """You are a book recommender system that help users to find books that match their preferences. 
For each question, suggest three books, with a short description of the plot and the reason why the user might like it.
Use the following pieces of context to answer the question at the end. 
For each question, take into account the context and the personal information provided by the user.

{context}

Question: {question}
Your response: """
```

For this experiment, we will use the same query as before, but provide context about the user's reading preferences. I made the preferences such that more fantastical political books would be recommended.

**Context**: "I am particularly drawn to dark fantasy themes that intertwine different mythical creatures, similar to Game of Thrones.
I would also like these stories to include elements such as dragons, faeries, wizards, elves, kings, and queens.
The ideal setting for these stories is in imaginative fantasy realms, distinct from our own world."
I prefer to avoid narratives centered on space travel and those featuring teenage protagonists.
**Query**: "Can you recommend a fiction book about politics?"
**Results**:
| Book Title              | Author          | Description                                                                                                                                                                                                                                                                                                                                                                                                                            |
|-------------------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| The Dark Tower          | Stephen King    | Follows a gunslinger on a quest to the Dark Tower, a place key to world salvation. Features dragons, wizards, and otherworldly beings, and a complex political landscape. Its mix of fantasy and politics is perfect for fans of Game of Thrones.                                                                                                                                                                                       |
| The Name of the Wind    | Patrick Rothfuss| The story of Kvothe, a legendary magician and musician, recounting his journey to becoming a wanted man. Set in a dark fantasy world with faeries and mythical creatures, it explores power, politics, and the use of magic. Ideal for readers interested in a politically charged fantasy narrative.                                                                                                                                       |
| The Lies of Locke Lamora | Scott Lynch     | Chronicles the adventures of a group of thieves and con artists in a fantasy world. Combining elements of dark fantasy, intricate world-building, and political intrigue, this novel is well-suited for readers who enjoy tales of cunning and deception in a fantastical setting.                                                                                                                                                        |


As you can see, I achieved in getting more mythical political books.

## Experiment 2A: User Demographics (A BRIEF Study on Ethical Prompts)
When developing a recommender system. It is important to me to not push problamatic tropes and stereotypes. In this extention of the previous experiment, I explore what happens when I include user demographic, such age and sex, into the prompt. 

I kept the main context and propt the same as above.

**Extended Context 1**: "I am a 25 year old woman"
**Results**:
| Book Title                    | Author           | Description                                                                                                                                                                                                                                                                         |
|-------------------------------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| The Queen of Blood            | Sarah Beth Durst | Set in a dark fantasy world with faeries and dragons, this story follows Daleina, who navigates royal court intrigue as she trains to become a queen, tasked with protecting her kingdom from destructive spirits.                                                                   |
| The Goblin Emperor            | Katherine Addison| This imaginative fantasy realm is populated by elves, goblins, and other mythical beings. The story centers on Maia, a half-goblin who becomes emperor unexpectedly, facing the complex politics of the court and discrimination from within.                                          |
| The Hundred Thousand Kingdoms | N.K. Jemisin     | A unique fantasy world where gods and mortals coexist. Yeine, the protagonist, is embroiled in a political struggle for the throne of Sky, a kingdom ruled by gods and goddesses. The book features a diverse cast, including intriguing dragons, and explores themes of power and worship. |


**Extended Context 2**: "I am a 25 year old man
**Results**:
| Book Title              | Author            | Description                                                                                                                                                                                                                                                                                       |
|-------------------------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| The Name of the Wind    | Patrick Rothfuss  | A story of Kvothe, who becomes a powerful wizard in a world of mythical creatures and magic. The book, set in a fantastical realm, delves into politics, power, and the relationships between races and creatures, appealing to fans of "Game of Thrones" and dark fantasy.                           |
| The Lies of Locke Lamora| Scott Lynch       | Set in a magical world, it follows a group of thieves entangled in political schemes. Featuring wizards, elves, and other mythical beings, it explores themes of power, betrayal, and redemption, making it ideal for fans of dark fantasy and political intrigue.                                  |
| The City of Brass      | S.A. Chakraborty  | An imaginative fantasy set in a world inspired by Middle Eastern mythology, it tells the story of a young woman discovering her half-human, half-djinn heritage. The book is rich in cultural and mythological elements, appealing to those interested in diverse fantasy narratives.                  |


#### Key Differences

**Cultural and Gender Perspectives**: The books from list recommended to women, includes books that often explore themes from more diverse cultural backgrounds (like "The City of Brass") and have a stronger focus on female protagonists and their experiences (e.g., "The Queen of Blood").
**Tone and Style**: Books from the list recommended to men, tends to lean more towards traditional high fantasy and dark fantasy, often featuring male protagonists in more conventional hero's journeys, whereas the list recommended to women may include more nuanced explorations of character, identity, and society.

**For Women**: The first set might be recommended to women due to its focus on female protagonists, exploration of themes like societal roles and expectations, and diverse cultural settings.
**For Men**: The second set of books might be stereotypically recommended to men due to traditional masculine themes like physical combat, traditional hero's journey, and male-centric narratives.

It should be noted that the propmt does not define which aspects of the context are more important compared to the rest of the information given. It's convieveable to me that because the gender and age of the user were offered to the recommender system, that it deemed it to be important enough to recommend new books based on that information. In regards to the gender of the protagonist, it is also possible that readers enjoy reading books about main characters that relate to themselves somehow-- the easiest way and only way in this case being age and sex.

The notion of recommending specific books to men or women based on their content is outdated and reinforces gender stereotypes. Literary preferences, like most areas, are highly individual and do not necessarily align with one's gender.


## Conclusion
This Book Recommender System represents a significant advancement in personalized book recommendations. By harnessing OpenAI's embeddings and LangChain's flexibility, it offers a sophisticated, user-friendly tool for book enthusiasts seeking tailored reading suggestions.
