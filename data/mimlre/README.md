Active Learning Data
====================

This folder contains the dataset associated with the active learning paper:

    INSERT CITATION HERE!!!

Annotated Sentences
-------------------
A CSV file contains the entire set of annotated sentences, bzip'd, in:

    annotated_sentences.csv.bz2

The columns of this file are as follows:

  * key: The sentence key, used in cross-referencing this sentence with other
    resources
  * relation: The annotated relation, as determined by Turkers
  * confidence: A metric of Turker confidence. It is defined as the percentage of
    Turkers which chose this relation over the NO_RELATION option.
  * jsDistance: The Jensen Shannon divergence of this example, as determined
    by the committee of relation extractors during example selection.
    The larger this distance, the more uncertain we were about this example.
  * sentence: The plain-text gloss of the sentence. Note that this string is encoded as
    HTML, exactly as it was shown to the Turkers. This also means that commas are escaped,
    for easier CSV reading.
  * entity: The plain-text gloss of the entity.
  * entityCharOffsetBegin: The character offset of the beginning of the entity, 
    in the original string.
  * entityCharOffsetEnd: The character offset of the end of the entity.
  * slotValue: The plain-text gloss of the slot value.
  * slotValueCharOffsetBegin: The character offset of the beginning of the slot value, 
    in the original string.
  * slotValueCharOffsetEnd: The character offset of the end of the slot value.
