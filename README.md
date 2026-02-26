# The Architecture of Machine Learning

A multi-paradigm Streamlit reference hub for ML concepts — theory, visual breakdowns,
and runnable step-by-step implementations.

## Paradigms

| Paradigm                  | Modules | Path                             |
|---------------------------|---------|----------------------------------|
| 🤖 Generative AI          | 15      | `generative_ai/topics/`          |
| 📈 Supervised Learning    | 9       | `supervised/topics/`             |
| 🔵 Unsupervised Learning  | 8       | `unsupervised/topics/`           |
| 🎮 Reinforcement Learning | 6       | `reinforcement_learning/topics/` |

## Learning Paths

```
SUPERVISED
Linear Regression → Logistic Regression → SVM → Decision Trees
→ Random Forests → Gradient Boosting → KNN → Naive Bayes → Neural Networks

UNSUPERVISED
K-Means → DBSCAN → Hierarchical Clustering → PCA
→ t-SNE/UMAP → Autoencoders → GMM → Anomaly Detection

REINFORCEMENT LEARNING
MDP → Q-Learning → DQN → Policy Gradient → PPO → Actor-Critic

GENERATIVE AI
Tokenization → Language Modeling → Transformer LLMs → Pretraining
→ Fine-Tuning → Alignment → Prompt Engineering → RAG → AI Agents
→ Multi-Agent → Inference Optimization → Generative Models
→ Multimodal AI → Architecture Innovations → Evaluation & Safety
```

## Running the App

```bash
pip install streamlit
streamlit run app.py
```

## Adding a New Module

1. Copy `topic_template.py`
2. Rename with a numeric prefix (e.g. `10_new_topic.py`)
3. Drop into the relevant paradigm's `topics/` folder
4. Fill in `THEORY`, `OPERATIONS`, and `VISUAL_HTML`
5. The app auto-discovers it — no registration needed
