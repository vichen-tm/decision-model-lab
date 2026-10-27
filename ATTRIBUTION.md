# Attribution and distribution

| Source | Original reference | Use here | Distribution |
|---|---|---|---|
| AG News | [Dataset and card](https://huggingface.co/datasets/fancyzhx/ag_news), [author corpus](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) | News-category experiments and error audit | The dataset card lists license as unknown; article input text is excluded from public Git |
| ProofWriter | [Paper](https://aclanthology.org/2021.findings-acl.317/), [source archive V2020.12.3](https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip) | Facts/rules/statement experiments; OWA depth-5 train | No confirmed standalone license in our archive audit; source input text is kept in local private evidence |
| First decision benchmark | [Repository](https://github.com/vichen-tm/decisions-model-banchmark) | Reused scoring/journal/runner and historical baseline results | Copied core retained under `decision_benchmark`; local source file hashes are in release inventory |
| OpenRouter Decisions | [OpenRouter](https://openrouter.ai/) | Inference provider interface and recorded usage | This repository supplies an adapter, not the provider or model weights |

Source corpus IDs and theory/question IDs are in the corpus index. The full ProofWriter archive hash used was `bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26`; the OWA depth-5 train member hash was `d0501d29506159e9b59f3cb9dcff10b4eb44f4d94dd52f9ee0bf93e5054a262f`.

The public package contains our code, instructions, numerical outputs, prompt templates and source identifiers. The code license does not grant rights to third-party dataset texts or model services. Local `private/` corpus archives are excluded from Git and publication archives. They preserve exact inputs for the author; obtaining benchmark source texts separately is necessary to repeat their API requests.
