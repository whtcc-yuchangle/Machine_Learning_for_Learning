# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a personal machine learning study project — a curated series of Jupyter notebooks with companion markdown documentation, progressing from foundational Python data libraries through ML algorithms toward deep learning.

## Repository Conventions

**File naming:** `0Nth_TopicName.ipynb` and `0Nth_TopicName.md` (e.g., `01th_numpy.ipynb` + `01th_numpy.md`). The two-digit prefix defines the learning order.

**Paired files per topic:**
- `.ipynb` — executable Jupyter notebook with runnable code cells and inline outputs
- `.md` — detailed reference document covering principles, formulas, API tables, and best practices for the same topic

**Data and assets:**
- Data files use the topic prefix: `02th_grade.csv`, `02th_salary.xlsx`
- Generated images live in `03th_img/` (Matplotlib outputs)
- Raw data files are committed directly (no `data/` directory convention in practice, despite `.gitignore`)

## Learning Progression

| Stage | Topics | Focus |
|-------|--------|-------|
| 01th | NumPy | Array ops, math, indexing |
| 02th | Pandas | Series/DataFrame, CSV/Excel I/O, web scraping |
| 03th | Matplotlib | Line, scatter, bar, histogram, box, pie, radar, 3D, heatmap |
| 04th | KNN | Manual implementation + sklearn `KNeighborsClassifier` |
| 05th | Decision Tree | Gini/entropy, sklearn `DecisionTreeClassifier`, visualization |
| Future | Scikit-learn, supervised/unsupervised learning, PyTorch, CNNs, RNNs | (planned) |

## Environment

- **Python**: 3.11+
- **Core dependencies**: `numpy pandas matplotlib jupyter openpyxl requests lxml seaborn scikit-learn`

```bash
pip install numpy pandas matplotlib jupyter openpyxl requests lxml seaborn scikit-learn
jupyter notebook
```

There is no `requirements.txt` — dependencies are documented in the README.

## Adding a New Topic

When adding a new numbered topic:
1. Create `0Nth_TopicName.ipynb` with runnable code cells following the established structure (imports → principles → manual implementation → library usage → real dataset case study)
2. Create `0Nth_TopicName.md` as a companion reference with sections: Overview, Learning Objectives, Content Structure (chapters with formulas and API tables), Summary, and Reference Materials
3. Update the README learning path with the new topic entry
4. Place any topic-specific data files in the repo root with the matching `0Nth_` prefix
