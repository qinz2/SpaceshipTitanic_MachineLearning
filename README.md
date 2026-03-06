# Spaceship Titanic Machine Learning Project

Kaggle Spaceship Titanic 竞赛的完整机器学习解决方案。

**模型性能**: F1-Score 0.8033 (±0.0098) | Accuracy 0.8059 (±0.0089)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

从 [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic) 下载 `train.csv` 和 `test.csv`，放入 `spaceship-titanic_data/` 文件夹。

### 3. 修改路径

打开每个 Python 脚本，将数据路径修改为你的实际路径：

```python
train_path = "your_path/spaceship-titanic_data/train.csv"
test_path = "your_path/spaceship-titanic_data/test.csv"
```

### 4. 运行代码

**一键运行（推荐）：**
```bash
run_all.bat
```

**或逐步运行：**
```bash
python code/01_data_exploration_and_preprocessing.py
python code/02_missing_value_analysis_and_processing.py
python code/04_data_preprocessing_and_model_selection.py
python code/05_cross_validation_and_hyperparameter_tuning.py
python code/06_model_evaluation_and_feature_analysis.py
python code/07_generate_test_predictions.py
```

### 5. 提交结果

运行完成后，在 `result/submission.csv` 找到提交文件，上传到 Kaggle。

## 项目结构

```
├── code/                          # 代码文件夹
│   ├── 01_data_exploration_and_preprocessing.py
│   ├── 02_missing_value_analysis_and_processing.py
│   ├── 04_data_preprocessing_and_model_selection.py
│   ├── 05_cross_validation_and_hyperparameter_tuning.py
│   ├── 06_model_evaluation_and_feature_analysis.py
│   └── 07_generate_test_predictions.py
├── spaceship-titanic_data/        # 数据文件夹（需自行下载）
├── result/                        # 结果文件夹（自动生成）
├── README.md
├── requirements.txt
└── run_all.bat
```

## 技术特点

- 完整的机器学习流程：数据探索 → 特征工程 → 模型训练 → 评估优化
- 5折交叉验证 + 网格搜索超参数优化
- 使用 Hist Gradient Boosting 模型
- 15+ 可视化图表
- 详细的代码注释

## 常见问题

**Q: FileNotFoundError?**  
A: 检查并修改脚本中的数据路径。

**Q: 运行时间太长?**  
A: 第5步（交叉验证）需要10-30分钟，可以跳过直接使用第4步的模型。

**Q: 如何提升性能?**  
A: 尝试更多特征工程、模型集成、调整超参数范围。

---

*Good luck! 🚀*
