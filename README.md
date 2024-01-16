
<img align="right" src="logo.jpg" alt="" width="150"/>

# DatSciEO
A group project for the "Data Science in Earth Observation" module (WS 23/24, TUM)

<br>
<br>

**Project tasks**:

- Use Sentinel-2 multi-spectral imagery to derive the dominant tree species for forests in Germany
- Dominant tree species shall be classified by a machine-learning approach trained through the 
provided reference data

**How to start**:

- Create an environment by typing `conda create --name <name> python=3.10`
- Activate the environment by typing `conda activate <name>`
- Run the command `pip install -r requirements.txt`

## Results
| Model        | Dataset   | Accuracy |
|--------------|-----------|------------|
| SVM (Handeul) | <br> - split 0.8/0.2 <br> - mode?      | 47.22 %       |
| KNN (Yi) | <br> - split ? <br> - mode?      | 40 %       |
| MLP (Yi) | <br> - split ? <br> - mode?      | 22 %       |
| Random Forest Classifier <br> with hyperparameter optimization (Chris) | <br> - split 0.7/0.3 <br> - 57928 samples <br> - data augmentation included <br> - no autumn bands <br> - no nan-values  | 44 %       |
