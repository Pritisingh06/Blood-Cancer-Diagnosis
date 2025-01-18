### Prerequisites
Ensure you have Python installed along with the required packages. You can install the dependencies using:
```bash
pip install -r requirements.txt
```

### Dataset
Place the dataset in the `data/raw` directory. The dataset should be organized by class folders. For example:
```
data/raw/
├── class_1/
├── class_2/
```

### How to Run
1. Preprocess the data and load datasets using `src/preprocess.py`.
2. Train the model using `src/train.py`:
   ```bash
   python src/train.py
   ```
3. View the training metrics in the `outputs/` folder.
4. The trained model will be saved in the `saved_models/` folder.

### Visualization
The training and validation accuracy/loss curves are saved as `training_history.png` in the `outputs/` folder.

### Results
- **Maximum Validation Accuracy**: The best accuracy achieved during training is displayed in the console after training.

## Contributing
Feel free to fork the repository, make improvements, and submit pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- TensorFlow and Keras documentation.
- Google Colab for hosting and running the project.
