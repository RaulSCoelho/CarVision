# Sistema de Classificação de Carros

## Equipe

- Bruno Henrique Miranda de Oliveira (25459333)
- Raul Semicek Coelho (25891651)
- Bruno Vinicius Martins Faria (25806939)
- Nicolas Fernandes (26387085)

## Tema

- Sistema de classificação de carros

## Objetivos
- Criar um sistema que classifique marcas de carros

## Banco de dados:
- **URL**: [stanford-cars-dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
- **Número de classes/marcas**: 196
- **Número de imagens**: 16.185

## Metodologia
### Pré-processamento de dados:
  - Carregamento do conjunto de dados de treinamento e teste usando ImageFolder.
  - Definição de transformações de imagem para pré-processamento, como redimensionamento, normalização e aumentação de dados (como espelhamento horizontal e rotação aleatória para dados de treinamento).

### Visualização dos dados:
  - Exibição de algumas imagens transformadas do conjunto de dados de treinamento para verificar se as transformações estão corretas.

### Definição do modelo:
  - Utilização da arquitetura ResNet-18 pré-treinada no conjunto de dados ImageNet, substituindo a camada de classificação final para se adequar ao número de classes do conjunto de dados de carros.
  - Movendo o modelo para o dispositivo apropriado (CPU ou GPU).

### Definição da função de perda e otimizador:
  - Utilização da função de perda CrossEntropyLoss.
  - Utilização do otimizador SGD (Gradiente Descendente Estocástico) com momento e decaimento de peso.

### Treinamento do modelo:
  - Laço de treinamento por várias épocas, onde cada época inclui:
      - Laço de treinamento sobre os lotes de dados de treinamento.
      - Computação da perda, retropropagação e atualização dos pesos do modelo.
      - Avaliação do desempenho do modelo no conjunto de dados de teste após cada época.
      - Salvamento de um checkpoint do modelo se a precisão no conjunto de dados de teste melhorar.

### Avaliação do modelo:
  - Laço de avaliação sobre os lotes de dados de teste para calcular a precisão final do modelo.

### Salvamento do modelo:
  - Salvamento do modelo treinado com melhor desempenho no conjunto de dados de teste.
