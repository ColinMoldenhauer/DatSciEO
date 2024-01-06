
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
| SVM (Handeul) | top10 <br> - split 0.8/0.2 <br> - mode?      | 44.4 %       |
| DRNN (Handeul) | top10 <br> - split 0.8/0.2 <br> - mode?      | 36.44 %       |
| FCNN (Handeul) | top10 <br> - split 0.8/0.2 <br> - mode?      | 38.54 %       |
| KNN (Yi) | top10 <br> - split ? <br> - mode?      | 40 %       |
| MLP (Yi) | top10 <br> - split ? <br> - mode?      | 22 %       |
