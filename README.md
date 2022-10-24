# ComFact
This is the source code for paper "ComFact: A Benchmark for Linking Contextual Commonsense Knowledge".


## Getting Started
Start with creating a **python 3.6** venv and installing **requirements.txt**.


## ComFact Datasets
Our **ComFact dataset** can be downloaded from [this link](https://drive.google.com/file/d/1nbQiASv32WTGVo5TQHatJbxBlz2HtMRP/view?usp=sharing), please place data/ under this root directory.

Pretrained Glove embeddings can be downloaded from [this link](https://drive.google.com/file/d/17a-sYMpS1hBYpavlq3tliZG7MKLWJHxB/view?usp=sharing), please place glove/ under the data/ directory and unzip glove.6B.zip in it.

**Data portions**:
- Persona-Atomic data portion: persona/
- Mutual-Atomic data portion: mutual/
- Roc-Atomic data portion: roc/
- Movie-Atomic data portion: movie/

## Data Preprocessing
```
python data_preprocessing_main.py
```

## Training and Evaluation
**Prepare directory**:
```
mkdir pred
mkdir runs
```

**Training**:
```
bash train_baseline.sh
```
Parameters:
- language model ${lm}: "deberta-large" | "deberta-base" | "roberta-large" | "roberta-base" | "bert-large" | "bert-base" | "distilbert-base" | "lstm"
- data portion ${portion}: "persona" | "mutual" | "roc" | "movie" | "all" (training on the union of all four data portions)
- context window ${window}: "nlg" (half window without future context) | "nlu" (full context window)
- linking task ${task}: "fact_full" (direct setting) | "head" (head entity linking, sub-task in pipeline setting) | "fact_cut" (fact linking of relevant head entities, sub-task in pipeline setting)
- evaluation set ${eval_set}: "val" (validation set) | "test" (testing set)


**Evaluating direct setting or sub-tasks in pipeline setting**:
```
bash run_baseline.sh
```
parameters refer to Training.


**Fine-grained analysis** on fact linking results (after evaluating by run_baseline.sh):
```
python evaluate_linking.py --model ${lm} --window ${window} --portion ${portion} --linking ${task}
```
parameters refer to Training, ${task} should be **fact_full** | **fact_cut**


**Evaluating full pipeline setting**:
```
bash run_baseline_pipeline.sh
```
parameters refer to Training.


**Evaluating head entity linkers in fact linking**:
```
bash run_baseline_head_linker.sh
```
parameters refer to Training.


**Cross evaluation**:
```
bash cross_evaluation.sh
```
Parameters:
- source data portion providing training set ${source_portion}: "persona" | "mutual" | "roc" | "movie" | "all"
- target data portion providing validation or testing set ${target_portion}: "persona" | "mutual" | "roc" | "movie" | "all"

others refer to Training.


**Plot heatmap** for cross evaluation (lm: roberta-large, window: nlg, task: fact_full):
```
python plot_cross_evaluation.py
```

## Downstream Dialogue Response Generation (CEM)
Setup [NLG evaluation toolkits](https://github.com/Maluuba/nlg-eval)
```
pip install git+https://github.com/Maluuba/nlg-eval.git@master
nlg-eval --setup
```

Download **CEM data** from [this link](https://drive.google.com/file/d/1p_70KLQzoqW92YexDyVlhKB4k9Mikv4E/view?usp=sharing) and place data/ under CEM/ directory.

**Original** preprocessed CEM data: ED/dataset_preproc.p

We also include our preprocessed CEM data **with ComFact refined knowledge**: ED/dataset_preproc_link.p

**Prepare directory**:
```
mkdir CEM/saved
mkdir CEM/vectors
```
Copy glove.6B.zip from data/glove/ to CEM/vectors/ directory.

### Knowledge Refinement (For producing dataset_preproc_link.p, Optional)

**Training fact linker** for CEM knowledge refinement:
```
python preprocessing_rel_tail_link_x.py
bash train_baseline_rel_tail_link_x.sh
```

**Extracting CEM data and preprocessing** for knowledge refinement:
```
python cem_data_extract.py
python preprocess_cem_link.py
```
The extracted data will be placed in data/cem/rel_tail/nlg/test/${split}_data.json, where ${split}: "train" | "val" | "test"

**Knowledge refining** by fact linker, *i.e.*, **labeling** the relevance of knowledge in the extracted CEM data:
```
bash run_baseline_cem_link_x.sh
python label_cem.py
```

**Write back** to the CEM data form:
```
python cem_data_back.py
```

### Dialogue Modeling
Switch to the **CEM folder**:
```
cd CEM
```

**Training CEM** dialogue model:
```
python main.py --model cem --dataset ${dataset} --save_path ${save} --model_path ${save} --cuda
```
Parameters:
- data source ${dataset}: **dataset_preproc.p** (original CEM dataset) | **dataset_preproc_link.p** (CEM dataset with ComFact refined knowledge),
- ${save}: your directory for saving the model and results.

**Testing CEM** dialogue model:
```
python main.py --test --model cem --dataset ${dataset} --save_path ${save} --model_path ${save} --cuda
```

**NLG Evaluation**:

Move the obtained results.txt from your result saving directory to results/ directory, rename the file to ${name}.txt, then run:
```
python src/scripts/evaluate.py --results ${name}
```
Parameters:
- ${name}: name of the results file, *e.g.*, CEM_link

We include the dialogue generation results: **CEM_ori.txt** (original CEM) and **CEM_link.txt** (CEM trained with ComFact refined knowledge) under results/ directory.
