# [CarVision](https://image-classify.vercel.app/predictions)

## Equipe 👷‍♂️

- Bruno Henrique Miranda de Oliveira (25459333)
- Raul Semicek Coelho (25891651)
- Bruno Vinicius Martins Faria (25806939)
- Nicolas Fernandes (26387085)

## Tema 🚗

- Sistema de classificação de carros

## Objetivos 🎯
- Criar um sistema que classifique marcas de carros

## Banco de dados 📝
- <b>URL</b>: [stanford-cars-dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
- <b>Número de classes/marcas</b>: 196
- <b>Número de imagens</b>: 16.185

### Agradecimentos ao Banco de Dados
[Dados provenientes do Laboratório de IA da Universidade de Stanford](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

3D Object Representations for Fine-Grained Categorization

Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei

4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.

## Revisão da literatura 📖

### Artigo 1
[Reconhecimento de silhueta de automóveis para carros autônomos utilizando aprendizado de máquina](https://wiki.sj.ifsc.edu.br/images/e/e8/TCC1_Tamara_Arrigoni.pdf)

### Objetivos 
<p>Os objetivos do artigo foi desenvolver um sistema para investigar e aplicar técnicas e algoritmos de aprendizado de máquina, analisar o desempenho das técnicas de acordo com um conjunto de dados pré-definido e definir, a partir dos resultados de testes, as técnicas que mais se adéquam ao objetivo geral do projeto.<p/>

### Banco de dados utilizado
<p>Para o desenvolvimento deste trabalho uma base de dados disponível no repositório de aprendizado de máquina da Universidade da Califórnia, Irvine, foi utilizada<p/>

### Modelos utilizados
Foram utilizados alguns modelos para verificar a acurácia do sistema:
- Modelo baseado em Redes Neurais Artificiais
- Modelo baseado em Máquina de Vetores de Suporte
- Modelos baseados em Árvores de Decisão (C5.0/CART)
- Modelo baseado em Random Forest
- Modelo baseado em K-Nearest Neighbors (k-NN)
- Modelo baseado em Naive Bayes
- Modelo baseado em K-means

### Resultados obtidos
A acurácia de cada modelo foi dada por:
- Redes Neurais Artificiais: 83,33%
- Máquina de Vetores de Suporte: 79,17%
- Árvores de Decisão(C5.0): 72,62%
- Árvores de Decisão(CART): 69,05%
- Random Forest: 71,43%
- K-Nearest Neighbors(k-NN): 74,40%
- Naive Bayes: 62,50%
- K-means: 41,07%

  <hr>

### Artigo 2
[Técnicas de PDI e Inteligência Artificial Aplicadas ao Reconhecimento de placas de Carro nos Padrões Brasileiros](https://www.researchgate.net/profile/Edson-Cavalcanti-Neto/publication/262956949_TECNICAS_DE_PDI_E_INTELIGENCIA_ARTIFICIAL_APLICADAS_AO_RECONHECIMENTO_DE_PLACAS_DE_CARRO_NOS_PADROES_BRASILEIROS/links/0c96053970df6e1d49000000/TECNICAS-DE-PDI-E-INTELIGENCIA-ARTIFICIAL-APLICADAS-AO-RECONHECIMENTO-DE-PLACAS-DE-CARRO-NOS-PADROES-BRASILEIROS.pdf)

### Objetivos 
Este trabalho tem como objetivo o desenvolvimento de técnicas de Processamento Digital de Imagem e Inteigência Computacional com o foco em sistema de detecção e reconhecimento de placas de veículos nos padrões brasileiros.

### Banco de dados utilizado
Não informado

### Modelos utilizados
- Filtros operadores gradiente (Roberts, Prewwit, Sobel)
- Filtro de Canny
- Transformada de Hough
- RNA MLP (Rede Neural Artificial / Multi Layer Perceptron)

### Resultados obtidos
<p>No teste do sistema foi utilizado 700 vídeos e a partir destes foram extraídas 12000 imagens.</p>
<p>A etapa do PDI desta forma obteve uma taxa de acerto de 97.02%.</p>
<p>A etapa de reconhecimento uma taxa de acerto de 97.7% em relação aos números das placas e 96.4% em relação as letras.</p>
<p>No processo inteiro obteve-se <b>97.3%</b> de acerto na extração e reconhecimento dos números e <b>96.16%</b> nos caracteres</p>

<hr>

### Artigo 3
[TÉCNICAS DE CLASSIFICAÇÃO DE IMAGENS PARA ANÁLISE DE COBERTURA VEGETAL](https://www.scielo.br/j/eagri/a/NfyXzM9DZsx5g3gnCQgCSsg/?lang=pt)

### Objetivos 
O artigo defende e explica as técnicas de sensoriamento remoto para subsidiar decisões na área agrosilvopastoral, com ênfase na compreensão dos processos envolvidos e na comunicação dos resultados de forma acessível.
### Banco de dados utilizado
Imagens retiradas com sensores e satélites
### Modelos utilizados
Modelo de classificação supervisionada;
Modelo de classificação não-supervisionada;
Modelo de classificação híbrida;

### Resultados obtidos
Melhora na análise de ocupação do solo
<hr>

### Artigo 4

[CLASSIFICAÇÃO DE IMAGENS DERMATOSCÓPICAS COM MACHINE LEARNING](https://bdtd.ucb.br:8443/jspui/bitstream/tede/2802/2/JulioCezarDissertacao2020.pdf)

### Objetivos 
O artigo tem como objetivo detalhar o desenvolvimento de um aplicativo utilizando machine learning para auxiliar no diagnóstico precoce de melanoma. Para alcançar os objetyivos do trabalho deve ser considerado os seguintes objetivos.

1. Realizar uma revisão de literatura sobre o emprego de machine learning na identificação de melanoma utilizando imagens dermatoscópicas;
2. Identificar bancos de imagens dermatoscópicas para treinar e validar o modelo de machine learning;
3. Desenvolver um modelo de machine learning para classificar imagens dermatoscópicas em duas categorias: Melanoma e Não Melanoma;
4. Desenvolver uma aplicação para viabilizar o uso do modelo;
5. Propor um protocolo de utilização do classificador pelos Dermatologistas.

### Banco de dados utilizado

Identificar bancos de imagens dermatoscópicas para treinar e validar o modelo de machine learning;

### Modelos utilizados

Modelo de machine learning

### Resultados obtidos

Foi possível obter um modelo de machine learning capaz de classificar imagens dermatoscópicas para detecção de melanoma. O modelo obteve a precisão de 94% no processo de treinamento e validação.

## Materiais e Métodos ⚡
### Metodologia
**Pré-processamento de dados:**
  - Carregamento das anotações de treinamento e validação a partir de arquivos CSV.
  - Definição das classes a serem usadas no treinamento e filtro das anotações com base nessas classes.
  - Utilização do ImageDataGenerator para pré-processamento e aumentação de dados, incluindo redimensionamento, normalização, e várias técnicas de aumentação (como espelhamento horizontal e vertical, rotação, zoom e ajustes de brilho).

**Visualização dos dados:**
  - Exibição de algumas imagens transformadas dos conjuntos de dados de treinamento para verificar se as transformações estão corretas.

**Definição do modelo:**
  - Utilização da arquitetura VGG16 pré-treinada no conjunto de dados ImageNet, removendo a parte superior do modelo.
  - Adição de camadas densas, normalização por lotes e dropout para construir a nova cabeça de classificação.
  - Definição de algumas camadas da VGG16 como não treináveis, dependendo do nível de fine-tuning especificado.

**Definição da função de perda e otimizador:**
  - Utilização da função de perda categorical_crossentropy.
  - Utilização do otimizador Adam com uma taxa de aprendizado ajustada.

**Treinamento do modelo:**
  - Laço de treinamento por várias épocas, onde cada época inclui:
      - Laço de treinamento sobre os lotes de dados de treinamento.
      - Avaliação do desempenho do modelo no conjunto de dados de validação após cada época.
      - Utilização de callbacks, como PlotLossesCallback, ModelCheckpoint, EarlyStopping e ReduceLROnPlateau, para monitorar o desempenho do modelo, salvar o melhor modelo, parar o treinamento cedo se necessário, e ajustar a taxa de aprendizado.

**Avaliação do modelo:**
  - Avaliação do modelo treinado no conjunto de dados de treinamento e validação para calcular as métricas de perda e precisão
  - Carregamento do melhor modelo salvo e cálculo da precisão final e da matriz de confusão no conjunto de dados de teste

**Salvamento do modelo:**
  - Salvamento do modelo treinado com melhor desempenho no conjunto de dados de teste.

**Salvamento do modelo:**
  - Realização de predições no conjunto de dados de validação usando o melhor modelo salvo.
  - Cálculo e exibição da matriz de confusão e do relatório de classificação para avaliar o desempenho do modelo em termos de precisão, recall e F1-score para cada classe.

## Resultados 🏁

### Sem modelo pré-treinado
  - Configurações do modelo:
    - Tamanho da imagem de entrada: (224, 224)
    - Formato da entrada: (224, 224, 3)
    - Tamanho do lote: 32
    - Número de épocas: 100
    - Número de classes: 196

  - Métricas de desempenho:
    - Precisão no conjunto de treinamento: 0.008
    - Precisão no conjunto de validação: 0.010
    - Perda no conjunto de treinamento: 5.270
    - Perda no conjunto de validação: 5.295
  
  - Precisão (Accuracy):
    - O gráfico de precisão mostra que a precisão no treinamento aumentou ligeiramente nas primeiras épocas, mas estabilizou em torno de 0.008.
    - A precisão no conjunto de validação permaneceu constante em 0.010 ao longo do treinamento.

  - Perda (Loss):
    - O gráfico de perda mostra que a perda no treinamento diminuiu continuamente ao longo das épocas, começando em 5.282 e diminuindo para 5.270.
    - A perda no conjunto de validação aumentou levemente ao longo das épocas, começando em 5.282 e aumentando para 5.295.

  Os resultados indicam que o modelo sem pré-treinamento não conseguiu aprender adequadamente as características dos dados, resultando em uma baixa precisão e uma alta perda tanto no treinamento quanto na validação.


### Com modelo pré-treinado VGG16 (10 classes)
  - Configurações do modelo:
    - Tamanho da imagem de entrada: (224, 224)
    - Formato da entrada: (224, 224, 3)
    - Tamanho do lote: 32
    - Número de épocas: 100
   
  - Métricas de desempenho:
    - Precisão no conjunto de treinamento: 0.654
    - Precisão no conjunto de validação: 0.608
    - Perda no conjunto de treinamento: 1.109
    - Perda no conjunto de validação: 1.290

### Com modelo pré-treinado VGG16 (20 classes)
  - Configurações do modelo:
    - Tamanho da imagem de entrada: (224, 224)
    - Formato da entrada: (224, 224, 3)
    - Tamanho do lote: 32
    - Número de épocas: 100
   
  - Métricas de desempenho:
    - Precisão no conjunto de treinamento: 0.909
    - Precisão no conjunto de validação: 0.703
    - Perda no conjunto de treinamento: 0.339
    - Perda no conjunto de validação: 0.832
