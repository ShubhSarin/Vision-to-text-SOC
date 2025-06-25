# Vision-to-text-SOC

## NumPy, Pandas and Matplotlib

### NumPy Fundamentals

#### Array Operations and Initialization
- Creating arrays using `np.array()`
- Random array generation with `np.random.random()`
- Special arrays: `np.zeros()` and `np.ones()`
- Understanding array shapes and dimensions
- Transposing arrays using `.T`
- Reshaping arrays with `.reshape()`
- Matrix multiplication using `.dot()`
- Broadcasting between arrays of different shapes

#### Array Indexing & Slicing
- Basic indexing techniques
- Slicing syntax: `array[start:end:step]`
- Accessing rows and columns in 2D arrays
- Sub-array extraction
- Vectorized operations for performance

### Pandas and Data Analysis

#### Census 2011 Analysis
- Loading and handling CSV data
- Calculating literacy rates across demographics
- Urban vs rural population analysis
- Employment statistics by state
- Data manipulation and aggregation

### Data Visualization with Matplotlib

#### Visualization Techniques
- Bar plots for literacy rates
- Comparative visualizations
  - Male vs female literacy
  - Urban vs rural demographics
- Multi-variable plotting for employment sectors
- State-wise statistical analysis

#### Key Findings
- Urban-rural literacy disparities
- Gender-based literacy differences
- Employment sector distribution
- Population demographic correlations
- Employment and literacy rate relationships

### Best Practices
- Using assertions for dimension checking
- Efficient array operations
- Statistical analysis methods
- Data visualization techniques
- Performance optimization through vectorization

## Deep Learning with PyTorch

### Assignment 2: MNIST Digit Classification

#### Dataset and Data Loading
- Loading MNIST dataset using `torchvision.datasets`
- Data transformation with `ToTensor()`
- Creating train/validation split (80-20)
- Using DataLoader for batch processing

#### Neural Network Architecture
- Feedforward Neural Network implementation
- Layer structure:
  - Input layer (28×28 = 784 neurons)
  - Hidden layers (1024 → 1024 → 512)
  - Output layer (10 classes)
- ReLU activation functions
- CrossEntropyLoss for classification

#### Training and Evaluation
- SGD optimizer with learning rate 0.007
- Training loop with batch processing
- Validation accuracy monitoring
- Model evaluation on test set
- Visualization of predictions

### Assignment 3: Computer Vision Tasks

#### Part 1: Pizza Classification
- Binary classification (Pizza vs Not Pizza)
- CNN Architecture:
  - 5 Convolutional layers with BatchNorm
  - MaxPooling layers
  - Fully connected layers (256 → 32 → 2)
  - Dropout for regularization
- Data preprocessing:
  - Image resizing to 128×128
  - Data augmentation
  - Train/validation split

#### Part 2: CIFAR-10 with Transfer Learning
- Implementation using ResNet50
- Custom architecture:
  - Additional conv layers before ResNet
  - Modified fully connected layers
  - Dropout for regularization
- Training optimizations:
  - Adam optimizer
  - Learning rate adjustment
  - Data augmentation with random flips

### Deep Learning Best Practices
- Model architecture design
- Batch normalization usage
- Dropout for regularization
- Learning rate selection
- Data augmentation techniques
- Transfer learning implementation
- GPU acceleration
- Model evaluation and visualization
